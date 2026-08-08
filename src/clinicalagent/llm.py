import hashlib
import logging
import re
from pathlib import Path
from typing import cast

from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from partialjson.json_parser import JSONParser
from pydantic import BaseModel

from .settings import settings
from .types import History, OpenAiClientConfig

logger = logging.getLogger(__name__)

parser = JSONParser()

# OpenAI's Structured Outputs `json_schema.name` is capped at 64 characters.
# Pydantic model names (including dynamically-generated ones) can exceed
# that, which otherwise 400s the request before a single token streams back.
_SCHEMA_NAME_MAX_LEN = 64
_SCHEMA_NAME_HASH_LEN = 8

# Markdown code fences (```json ... ``` or ``` ... ```) sometimes wrap the
# JSON payload. Fence markers aren't valid JSON syntax, so stripping them
# before scanning is safe and leaves clean-JSON streams untouched.
_CODE_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)

# Candidate JSON start characters, per partialjson.JSONParser.parsers.
_JSON_START_CHARS = "{["


def cap_schema_name(name: str, max_len: int = _SCHEMA_NAME_MAX_LEN) -> str:
    """Deterministically cap a structured-output schema name at `max_len` chars.

    Names within the limit pass through unchanged. Longer names are
    truncated and suffixed with a short stable hash of the *full original
    name*, so the result is deterministic for a given input and distinct
    for distinct inputs (even ones sharing a long common prefix).
    """
    if len(name) <= max_len:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:_SCHEMA_NAME_HASH_LEN]
    suffix = f"_{digest}"
    truncated = name[: max_len - len(suffix)]
    return f"{truncated}{suffix}"


def _schema_text_format(
    schema: type[BaseModel],
) -> ResponseFormatTextJSONSchemaConfigParam:
    """Build the `text.format` param for a schema, with its name capped.

    `type_to_text_format_param` always returns the `json_schema` variant for
    a Pydantic model (it asserts `type == "json_schema"` internally); the
    cast just gives the narrower, name-bearing TypedDict back to the caller.
    """
    text_format = cast(
        ResponseFormatTextJSONSchemaConfigParam, type_to_text_format_param(schema)
    )
    text_format["name"] = cap_schema_name(text_format["name"])
    return text_format


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s)


class StructuredStreamParser[T: BaseModel]:
    def __init__(self, schema: type[T]):
        self.schema = schema
        self.buffer = ""

    def feed(self, chunk: str) -> T | None:
        self.buffer += chunk
        cleaned = _strip_code_fences(self.buffer)

        # Fast path (regression): buffer parses as-is. This covers both
        # clean JSON from char 0 (complete or still-accumulating) and any
        # buffer that happens to already be valid/partial JSON.
        result = self._try_parse(cleaned)
        if result is not None:
            return result

        # If the buffer already starts with a JSON opener (ignoring leading
        # whitespace), a failed parse here just means it's incomplete so
        # far — normal streaming, not a failure. Don't go hunting for a
        # different candidate; more of *this* JSON value is still arriving.
        if cleaned.lstrip()[:1] in _JSON_START_CHARS:
            return None

        # Prose-tolerant path: the buffer doesn't start with JSON at all
        # (there may be no `{`/`[` yet — not a failure, just not parseable
        # yet). Scan forward for candidate JSON start positions and try
        # each in turn, so a `{` that turns out to be inside quoted prose
        # text doesn't get permanently locked onto — parsing just falls
        # through to the next candidate.
        for idx, ch in enumerate(cleaned):
            if ch in _JSON_START_CHARS:
                result = self._try_parse(cleaned[idx:])
                if result is not None:
                    return result
        return None

    def _try_parse(self, candidate: str) -> T | None:
        try:
            parsed = parser.parse(candidate)
            return self.schema.model_validate(parsed)
        except Exception as e:
            logger.debug(f"Stream parsing error: {e}\nCandidate: {candidate}")
            return None


async def stream_agent_response[T: BaseModel](
    history: History,
    schema: type[T],
    client_config: OpenAiClientConfig | None = None,
):
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

    stream = await client.responses.create(
        model=client_config.llm_model_name,
        input=history.compact(),  # type: ignore
        stream=True,
        text={"format": _schema_text_format(schema)},
        extra_body=client_config.extra_kw,
    )

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
    """Simulates streaming by reading a file and yielding its content in chunks."""
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
