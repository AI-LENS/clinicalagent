"""
LLM interaction layer for the clinicalagent framework.

This module handles all communication with OpenAI-compatible LLM APIs, including:

- **Streaming structured responses**: The LLM outputs JSON conforming to a Pydantic schema
  (e.g., ``AgentResponse``). This module incrementally parses the JSON as it streams in,
  yielding validated Pydantic objects at each parseable checkpoint.
- **Non-streaming structured responses**: For one-shot calls (e.g., conversation summarization),
  the full response is awaited and parsed.
- **Simulated streams**: For testing/debugging, responses can be read from local files instead
  of hitting a real API, activated by setting ``base_url`` to ``"file:/path/to/file"``.

Key design choice: The module uses ``partialjson`` to parse incomplete JSON during streaming.
This allows the agent to yield partial ``AgentResponse`` objects (e.g., with ``msg_to_user``
populated but ``action`` still incomplete), enabling real-time text streaming to the user
while the LLM is still generating.
"""

import logging
from pathlib import Path

from openai import AsyncOpenAI
from partialjson.json_parser import JSONParser
from pydantic import BaseModel

from .settings import settings
from .types import History, OpenAiClientConfig

logger = logging.getLogger(__name__)

parser = JSONParser()
"""Module-level partial JSON parser instance shared across all stream parsing operations."""


class StructuredStreamParser[T: BaseModel]:
    """
    Incrementally parses a streaming JSON response into a Pydantic model.

    As the LLM streams JSON tokens, each chunk is appended to an internal buffer.
    After each chunk, the parser attempts to parse the buffer as (possibly incomplete)
    JSON using ``partialjson``, then validates the result against the target Pydantic schema.

    This enables yielding partial but valid ``AgentResponse`` objects during streaming —
    for example, the ``msg_to_user`` field may be available before ``action`` is complete.

    Type Parameter:
        T: The target Pydantic model class (e.g., ``AgentResponse[SomeToolUnion]``).

    Attributes:
        schema: The Pydantic model class to validate parsed JSON against.
        buffer: Accumulated raw JSON string from all chunks received so far.
    """

    def __init__(self, schema: type[T]):
        self.schema = schema
        self.buffer = ""

    def feed(self, chunk: str) -> T | None:
        """
        Append a new chunk to the buffer and attempt to parse the accumulated JSON.

        Args:
            chunk: A new string fragment from the LLM stream.

        Returns:
            A validated Pydantic model instance if the buffer is parseable, or None
            if the JSON is still too incomplete to validate.
        """
        self.buffer += chunk
        try:
            parsed = parser.parse(self.buffer)
            result = self.schema.model_validate(parsed)
            return result
        except Exception as e:
            logger.debug(f"Stream parsing error: {e}\nBuffer: {self.buffer}")
            return None


async def stream_agent_response[T: BaseModel](
    history: History,
    schema: type[T],
    client_config: OpenAiClientConfig | None = None,
):
    """
    Stream a structured LLM response, yielding incremental Pydantic model instances.

    This is the primary LLM call used by the agent loop. It sends the conversation history
    to an OpenAI-compatible API with structured output (``text_format=schema``), then
    incrementally parses the streaming JSON response into validated Pydantic objects.

    Supports two modes:
        1. **Live API**: Connects to the configured OpenAI-compatible endpoint.
        2. **File-based simulation**: If ``base_url`` starts with ``"file:"``, reads from a
           local file instead. Useful for deterministic testing without API calls.

    Args:
        history: The conversation context to send to the LLM.
        schema: The Pydantic model class the LLM's JSON output should conform to
            (typically ``AgentResponse[T]`` or ``AgentResponseThoughtful[T]``).
        client_config: Optional API connection config. If None, falls back to global settings
            from ``clinicalagent-config.yaml``.

    Yields:
        Incrementally parsed ``T`` instances as more JSON becomes available from the stream.
        Each yielded object is a more complete version of the response.

    Raises:
        AssertionError: If no ``client_config`` is provided and ``llm_model_name`` or
            ``llm_base_url`` is not set in the global settings.
    """
    if client_config is None:
        assert settings.llm_model_name is not None, (
            "llm_model_name must be set in `local-agent-config.yaml`"
        )
        assert settings.llm_base_url is not None, (
            "llm_base_url must be set in `local-agent-config.yaml`"
        )
        client_config = OpenAiClientConfig(
            llm_model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            extra_kw=settings.llm_api_extra_kw,
        )
    stream_parser = StructuredStreamParser(schema)
    last_yielded: T | None = None

    if client_config.base_url.startswith("file:"):
        logger.debug(
            f"Using simulated agent stream from file. {client_config.base_url[5:]}"
        )
        async for chunk in simulated_agent_stream(
            Path(client_config.base_url[5:]),
            history,
            newline_delimited=client_config.llm_model_name == "newline",
        ):
            parsed = stream_parser.feed(chunk)
            if parsed is not None and parsed != last_yielded:
                last_yielded = parsed
                yield parsed
        return
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=client_config.api_key,
    )

    async with client.responses.stream(
        model=client_config.llm_model_name,
        input=history.compact(),  # type: ignore
        text_format=schema,
        extra_body=client_config.extra_kw,
    ) as stream:
        async for event in stream:
            if (
                event.type == "response.output_text.delta"
                or event.type == "response.refusal.delta"
            ):
                parsed = stream_parser.feed(event.delta)
                if parsed is not None and parsed != last_yielded:
                    last_yielded = parsed
                    yield parsed


async def simulated_agent_stream(
    path: Path, history: History, newline_delimited: bool = True
):
    """
    Simulate an LLM stream by reading pre-recorded responses from a local file.

    This is activated when ``OpenAiClientConfig.base_url`` starts with ``"file:"``.
    The file can contain a single response or multiple responses separated by
    ``\\n<---xxx--->\\n`` delimiters. When multiple responses exist, the correct one
    is selected based on how many assistant messages are already in the history
    (i.e., the Nth assistant turn reads the Nth response from the file).

    Chunking modes:
        - **Newline-delimited** (default): Each line is yielded as a separate chunk.
          Activated when ``newline_delimited=True`` or ``llm_model_name != "newline"``.
        - **Token-delimited**: Uses a regex tokenizer to split text into token-like chunks.
          Requires the ``regex`` package (install via ``pip install clinicalagent[debug]``).

    Args:
        path: Path to the file containing pre-recorded JSON responses.
        history: Current conversation history (used to determine which response to read
            in multi-response files).
        newline_delimited: If True, split by newlines. If False, split by regex tokens.

    Yields:
        String chunks simulating streaming output.

    Raises:
        AssertionError: If the file is empty or doesn't have enough responses for the
            current conversation turn.
        ImportError: If ``newline_delimited=False`` and the ``regex`` package is not installed.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 0, "Simulated stream file is empty"
    messages = content.split("\n<---xxx--->\n")

    if len(messages) > 1:
        current_msg_idx = sum(1 for msg in history.root if msg.role == "assistant")
        assert current_msg_idx < len(messages), (
            f"Not enough messages in simulated stream file {path} for current history. "
            f"Current message index: {current_msg_idx}, total messages in file: {len(messages)}"
        )
        msg_content = messages[current_msg_idx]
    else:
        msg_content = content

    if newline_delimited:
        for line in msg_content.splitlines(keepends=True):
            yield line
    else:
        try:
            import regex
        except ImportError:
            raise ImportError(
                "The 'regex' package is required for non-newline delimited simulated streams. "
                "regex package can be installed along with clinicalagent via 'pip install clinicalagent[debug]'"
            )

        _unused_pat = regex.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
        )
        for piece in regex.findall(_unused_pat, msg_content):
            yield piece


async def structured_agent_response[T](
    history: History,
    schema: type[T],
    client_config: OpenAiClientConfig | None = None,
) -> T:
    """
    Make a non-streaming LLM call and return a fully parsed structured response.

    Unlike ``stream_agent_response``, this waits for the complete response before
    returning. Used for auxiliary tasks like conversation summarization where
    streaming is not needed.

    Args:
        history: The conversation context to send to the LLM.
        schema: The Pydantic model class the response should conform to.
        client_config: Optional API connection config. Falls back to global settings if None.

    Returns:
        A fully parsed and validated instance of ``T``.

    Raises:
        AssertionError: If the response could not be parsed, or if required settings are missing.
    """
    if client_config is None:
        assert settings.llm_model_name is not None, (
            "llm_model_name must be set in `local-agent-config.yaml`"
        )
        assert settings.llm_base_url is not None, (
            "llm_base_url must be set in `local-agent-config.yaml`"
        )
        client_config = OpenAiClientConfig(
            llm_model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            extra_kw=settings.llm_api_extra_kw,
        )
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=client_config.api_key,
    )

    response = await client.responses.parse(
        model=client_config.llm_model_name,
        input=history.model_dump(),
        text_format=schema,
        extra_body=client_config.extra_kw,
    )

    assert response.output_parsed is not None, "Expected parsed response to be present"
    return response.output_parsed
