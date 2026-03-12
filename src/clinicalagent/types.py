"""
Core type definitions for the clinicalagent framework.

This module defines all the fundamental data structures used throughout the agent system:

- **Sentinel / Control Flow**: ``TERMINATE`` — a sentinel value returned by environments to signal
  the agent loop should stop.
- **MCP Compatibility**: ``McpTool``, ``McpToolAnnotations`` — Pydantic models that mirror the
  Model Context Protocol tool schema, enabling interop with any MCP server.
- **Tool System**: ``TypedTool``, ``BaseToolModel``, ``BaseToolMeta`` — type-safe tool definitions
  where each tool binds a Pydantic input model to an MCP tool descriptor. Tools can be marked
  as "terminating" to signal the agent should stop after calling them.
- **Conversation History**: ``Message``, ``History``, ``MessageFlag`` — an ordered, ID-validated
  message sequence with role-based access and flag-based metadata (e.g., marking system instructions,
  tool results, or conversation summaries).
- **Agent Responses**: ``AgentResponse``, ``AgentResponseThoughtful`` — structured LLM outputs
  that optionally include a thought trace alongside the user-facing message and tool action.
- **Streaming**: ``TextStream``, ``ToolCallStream`` — lightweight wrappers for incremental
  text deltas and completed tool calls emitted during streaming.
- **Configuration**: ``OpenAiClientConfig`` — connection parameters for the OpenAI-compatible API.
"""

import logging
from enum import Enum
from typing import Any, Generic, Literal, Self, TypeVar, overload

from json_schema_to_pydantic import create_model as json_schema_to_pydantic_model
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Sentinel

logger = logging.getLogger(__name__)

TERMINATE = Sentinel("TERMINATE")
"""Sentinel value used by ``Environment.on_agent_message_completed()`` to signal the agent loop should stop."""


class McpToolAnnotations(BaseModel):
    """
    Optional annotations for an MCP tool, providing hints about its behavior.

    These annotations help the agent (and callers) understand side effects before invoking a tool.
    They mirror the MCP specification's tool annotation schema.

    Attributes:
        title: Human-readable display name for the tool.
        readOnlyHint: If True, the tool only reads data and has no side effects.
        destructiveHint: If True, the tool may delete or irreversibly modify data.
        idempotentHint: If True, calling the tool multiple times with the same input produces the same result.
        openWorldHint: If True, the tool interacts with external/open-world systems (e.g., network APIs).
    """

    title: str | None = None
    readOnlyHint: bool | None = None
    destructiveHint: bool | None = None
    idempotentHint: bool | None = None
    openWorldHint: bool | None = None
    model_config = ConfigDict(extra="allow")


class McpTool(BaseModel):
    """
    Pydantic model representing an MCP (Model Context Protocol) tool definition.

    This is the base representation of a tool that the agent can invoke. It mirrors the
    MCP tool schema so that tools defined in any MCP server can be directly deserialized
    into this model.

    Attributes:
        name: Unique identifier for the tool (used in tool call requests).
        title: Optional human-readable display name.
        description: Natural-language description of what the tool does; included in the
            system prompt so the LLM knows when/how to use the tool.
        inputSchema: JSON Schema dict describing the tool's expected input arguments.
        outputSchema: Optional JSON Schema dict describing the tool's output format.
        icons: Optional list of icon references for UI rendering.
        annotations: Optional behavioral hints (read-only, destructive, etc.).
        meta: Internal metadata dict (aliased from ``_meta`` in serialized form).
            Used by clinicalagent to store framework-specific flags such as ``["TERMINATING"]``.
    """

    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None = None
    icons: list | None = None
    annotations: McpToolAnnotations | None = None
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    model_config = ConfigDict(extra="allow")


class CallToolRequestParams[T: BaseModel](BaseModel):
    """
    Parameters for a tool call request, as produced by the LLM.

    When the agent decides to invoke a tool, its structured output is parsed into this model.
    The type parameter ``T`` is the Pydantic model representing the tool's input schema,
    ensuring type-safe access to the tool's arguments.

    Attributes:
        tool_name: The name of the tool to invoke (must match a tool registered in the environment).
        arguments: The parsed, validated arguments for the tool call (typed as ``T``).
    """

    tool_name: str
    arguments: T


class ClinicalAgentError(Exception):
    """Base exception class for all clinicalagent-specific errors."""

    pass


class MaxAgentIterationsExceededError(ClinicalAgentError):
    """
    Raised when the agent loop exceeds the configured maximum number of iterations.

    This is a safety mechanism to prevent infinite loops. The iteration limit is set via
    ``Agent(max_iterations=...)`` or the ``max_agent_iterations`` setting in config.
    """

    pass


class MessageFlag(str, Enum):
    """
    Flags attached to messages in the conversation history to indicate their semantic role.

    These flags are used internally by the environment and agent to identify special messages
    (e.g., system prompts, tool results) during context engineering and history management.

    Values:
        is_system_instruction: Marks the system prompt prepended at the start of context.
            Removed and re-rendered on each iteration to reflect updated tool lists and remaining iterations.
        is_system_response: Marks messages generated by the system (not the LLM or user),
            such as continuation prompts when the agent fails to make a tool call.
        is_tool_result: Marks messages containing the output of a tool execution.
        is_summary: Marks messages that are LLM-generated summaries of earlier conversation,
            used when history exceeds ``max_history_length`` and must be compressed.
    """

    is_system_instruction = "is_system_instruction"
    is_system_response = "is_system_response"
    is_tool_result = "is_tool_result"
    is_summary = "is_summary"


class Message(BaseModel):
    """
    A single message in the conversation history.

    Messages are the fundamental unit of the agent's context. Each message has a role
    (system, user, or assistant), text content, and optional metadata flags.

    Attributes:
        id: Auto-assigned positional index within the History. Managed by ``History.ensure_valid_ids()``;
            you generally don't set this manually.
        role: The speaker role — ``"system"`` for system prompts, ``"user"`` for human/tool messages,
            ``"assistant"`` for LLM responses.
        content: The text content of the message.
        flags: Internal metadata flags (see ``MessageFlag``). Excluded from JSON schema generation
            so they don't leak into LLM-visible context.
    """

    id: SkipJsonSchema[int | None] = None
    role: Literal["system", "user", "assistant"]
    content: str
    flags: SkipJsonSchema[list[str]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class History(RootModel[tuple[Message, ...]]):
    """
    An ordered, ID-validated sequence of ``Message`` objects forming the conversation context.

    History is backed by a tuple (immutable per-entry, but the tuple reference is replaced on
    mutation via ``add_message``). Message IDs are automatically re-assigned to match their
    positional index whenever the history is created or modified.

    This model is a Pydantic ``RootModel``, so it can be constructed from a plain list of dicts:
        >>> history = History.model_validate([
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ])

    Key methods:
        - ``add_message()``: Append or insert a message (re-indexes all IDs automatically).
        - ``compact()``: Return a minimal list-of-dicts representation suitable for LLM input.
    """

    root: tuple[Message, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Validate and ensure message IDs are correct after initialization."""
        return self.ensure_valid_ids()

    def ensure_valid_ids(self, raise_warning: bool = True) -> Self:
        """
        Re-assign each message's ``id`` to match its positional index in the history.

        Called automatically after construction and after ``add_message()``. Logs a warning
        if any message had an ID that didn't match its position (indicates external tampering
        or deserialization from a reordered source).

        Args:
            raise_warning: If True, log warnings for mismatched IDs. Set to False during
                internal mutations (e.g., ``add_message``) where reindexing is expected.

        Returns:
            Self, for chaining.
        """
        for idx, message in enumerate(self.root):
            if raise_warning and message.id is not None and message.id != idx:
                logger.warning(
                    f"Message ID {message.id} was incorrect.\nMessage: {message}"
                )
            message.id = idx
        return self

    def compact(self) -> list[dict[Literal["role", "content"], str]]:
        """
        Return a minimal list-of-dicts representation of the history.

        This format contains only ``role`` and ``content`` fields — no IDs, flags, or extra
        metadata. It is the format passed directly to the OpenAI API as the ``input`` parameter.

        Returns:
            List of dicts with ``"role"`` and ``"content"`` keys.
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.root]

    @overload
    def add_message(self, msg: Message, *, index: int | None = None) -> None: ...

    @overload
    def add_message(
        self,
        *,
        role: Literal["system", "user", "assistant"],
        content: str,
        flags: list[str] | None = None,
        index: int | None = None,
    ) -> None: ...

    def add_message(
        self,
        msg: Message | None = None,
        *,
        role: Literal["system", "user", "assistant"] = "user",
        content: str = "",
        flags: list[str] | None = None,
        index: int | None = None,
    ) -> None:
        """
        Add a message to the history, either by passing a ``Message`` object or keyword arguments.

        Supports two calling conventions (enforced via ``@overload``):
            1. ``history.add_message(msg)`` — pass a pre-built Message.
            2. ``history.add_message(role="user", content="...")`` — build inline.

        The message is inserted at ``index`` (defaults to end). All message IDs are
        re-assigned after insertion to maintain positional consistency.

        Args:
            msg: A pre-built Message object. If provided, ``role``/``content``/``flags`` are ignored.
            role: Message role (only used when ``msg`` is None).
            content: Message text (only used when ``msg`` is None).
            flags: Optional list of ``MessageFlag`` values (only used when ``msg`` is None).
            index: Position to insert at. Defaults to appending at the end.

        Raises:
            IndexError: If ``index`` is out of bounds for the current history length.
        """
        if index is None:
            index = len(self.root)
        if not (0 <= index <= len(self.root)):
            raise IndexError(
                f"Index out of bounds for adding message to history of length {len(self.root)}."
            )
        if msg is None:
            msg = Message(
                id=index,
                role=role,
                content=content,
                flags=flags or [],
            )
        else:
            msg.id = index
        self.root = self.root[:index] + (msg,) + self.root[index:]
        self.ensure_valid_ids(raise_warning=False)


class AgentResponse[T: BaseModel](BaseModel):
    """
    Structured response produced by the LLM during each agent iteration.

    This is the core schema that the LLM's output is parsed into. It contains an optional
    user-facing message and an optional tool call action. The agent loop yields partial
    ``AgentResponse`` objects as they are streamed from the LLM.

    Type Parameter:
        T: The union of all tool input models available in the current environment.
            This ensures ``action.arguments`` is typed to the correct tool schema.

    Attributes:
        msg_to_user: Text message intended for the end user. May be None if the agent
            is only performing a tool call with no accompanying message.
        action: If the agent decides to invoke a tool, this contains the tool name and
            parsed arguments. None if the agent is only responding with text.
        turn_completed: Internal flag set to True by the agent loop after the full LLM
            response has been received. Not included in the JSON schema sent to the LLM.
    """

    msg_to_user: str | None = None
    action: CallToolRequestParams[T] | None = None
    turn_completed: SkipJsonSchema[bool] = False


class AgentResponseThoughtful[T: BaseModel](AgentResponse[T]):
    """
    Extended agent response that includes the LLM's internal reasoning trace.

    Used when ``Agent(disable_thought=False)``. The ``thought`` field captures the agent's
    chain-of-thought reasoning before it produces the user message or tool call. This is
    valuable for auditing, debugging, and regulatory transparency in clinical settings.

    Attributes:
        thought: The agent's internal reasoning/thought process. Visible to developers
            but typically not shown to end users.
    """

    thought: str | None = None


class Token(BaseModel):
    """Simple wrapper for a single token string. Used in tokenization utilities."""

    token: str


class OpenAiClientConfig(BaseModel):
    """
    Connection configuration for an OpenAI-compatible API endpoint.

    This config is used both for the main agent LLM calls and for auxiliary calls
    (e.g., history summarization). It supports any OpenAI-compatible provider
    (OpenAI, OpenRouter, vLLM, Ollama, etc.) via the ``base_url`` field.

    Attributes:
        llm_model_name: Model identifier (e.g., ``"gpt-4o"``, ``"google/gemini-2.5-flash"``).
        base_url: API base URL. Can also be ``"file:/path/to/file"`` for simulated/test streams.
        api_key: API authentication key. May be None for local/unauthenticated endpoints.
        extra_kw: Additional keyword arguments passed to the API call (e.g., ``temperature``,
            ``max_tokens``). These are sent as ``extra_body`` in the OpenAI SDK.
    """

    llm_model_name: str
    base_url: str
    api_key: str | None
    extra_kw: dict = {}


T_co = TypeVar("T_co", bound=type[BaseModel], covariant=True)
"""Covariant type variable for the tool's input model type (a class, not an instance)."""


class TypedTool(McpTool, Generic[T_co]):
    """
    A type-safe extension of ``McpTool`` that binds a Pydantic model to the tool's input schema.

    ``TypedTool`` is the bridge between MCP's JSON Schema-based tool definitions and Python's
    type system. It guarantees that when the LLM produces a tool call, the arguments are parsed
    and validated against the correct Pydantic model.

    Construction:
        You can create a TypedTool in two ways:
        1. **From a BaseToolModel subclass** (preferred): Pass ``input_model=MyToolModel``.
           The ``inputSchema`` is auto-generated from the model's JSON schema.
        2. **From an MCP tool definition**: Pass ``inputSchema={...}`` (a JSON Schema dict).
           A Pydantic model is auto-generated from the schema using ``json-schema-to-pydantic``.

    Termination:
        If the tool's ``_meta`` contains ``{"clinicalagent": ["TERMINATING"]}``, the tool is
        marked as terminating. When the agent calls a terminating tool, the agent loop stops.

    Attributes:
        input_model: The Pydantic model class for this tool's input arguments.
            Excluded from JSON schema generation and serialization.

    Example:
        >>> class SearchTrials(BaseToolModel, terminating=False):
        ...     '''Search for clinical trials by condition.'''
        ...     condition: str
        ...     phase: str = ""
        >>> tool = SearchTrials.convert_to_tool()  # Returns TypedTool[type[SearchTrials]]
    """

    model_config = ConfigDict(frozen=True)
    input_model: SkipJsonSchema[T_co] = Field(exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_input_schema(cls, data: Any) -> Any:
        """
        Pre-construction validator that synchronizes ``input_model`` and ``inputSchema``.

        Logic:
            - If ``input_model`` is provided, its JSON schema replaces any existing ``inputSchema``.
            - If only ``inputSchema`` is provided, a Pydantic model is auto-generated from it.
            - If the tool's ``_meta`` marks it as TERMINATING, the ``__is_terminating__`` flag
              is set on the input model class.

        Raises:
            ValueError: If neither ``input_model`` nor ``inputSchema`` is provided.
        """
        if isinstance(data, dict):
            input_model = data.get("input_model")
            input_schema = data.get("inputSchema")

            if input_model is None and input_schema is None:
                raise ValueError("Either input_model or inputSchema must be provided.")
            elif input_model is not None:
                if input_schema is not None:
                    logger.warning(
                        "Ignoring inputSchema since input_model is provided."
                    )
                data["inputSchema"] = input_model.model_json_schema()
            elif input_model is None:
                assert input_schema is not None
                data["input_model"] = json_schema_to_pydantic_model(
                    input_schema, base_model_type=BaseToolModel
                )

            if (
                "meta" in data
                and data["meta"]
                and ("clinicalagent" in data["meta"])
                and ("TERMINATING" in data["meta"]["clinicalagent"])
            ):
                data["input_model"].__is_terminating__ = True

        return data

    @classmethod
    def from_tool(cls, tool: McpTool):
        """
        Create a ``TypedTool`` from a plain ``McpTool`` instance.

        This is used when converting tools fetched from an MCP server (which arrive as
        ``McpTool`` objects) into ``TypedTool`` instances with auto-generated input models.

        Args:
            tool: An McpTool instance (e.g., from ``client.list_tools()``).

        Returns:
            A new TypedTool with an auto-generated input_model based on the tool's inputSchema.
        """
        return cls.model_validate(tool.model_dump())


class BaseToolMeta(ModelMetaclass):
    """
    Metaclass for ``BaseToolModel`` that supports the ``terminating`` class keyword.

    When defining a tool model, you can mark it as terminating by passing
    ``terminating=True`` as a class keyword argument:

        class FinalAnswer(BaseToolModel, terminating=True):
            answer: str

    This sets the ``__is_terminating__`` class attribute, which the agent loop checks
    to decide whether to stop after the tool is called.
    """

    __is_terminating__: bool = False

    def __new__(mcls, name, bases, attrs, **kwargs):
        terminating = kwargs.pop("terminating", False)
        cls = super().__new__(mcls, name, bases, attrs, **kwargs)
        if terminating:
            cls.__is_terminating__ = True
        return cls


class BaseToolModel(BaseModel, metaclass=BaseToolMeta):
    """
    Base class for defining tool input schemas as Pydantic models.

    Subclass this to define a tool's input arguments. The class docstring becomes the
    tool's description, and the class name becomes the tool's name.

    Features:
        - ``convert_to_tool()``: Class method that creates a ``TypedTool`` from this model.
        - ``terminating=True`` keyword: Marks the tool as a terminating tool (see ``BaseToolMeta``).

    Example:
        >>> class CheckEligibility(BaseToolModel):
        ...     '''Check patient eligibility for a clinical trial.'''
        ...     trial_id: str
        ...     patient_age: int
        ...     diagnosis: str
        ...
        >>> tool = CheckEligibility.convert_to_tool()
        >>> tool.name
        'CheckEligibility'
    """

    @classmethod
    def convert_to_tool(cls) -> TypedTool[type[Self]]:
        """
        Convert this tool model class into a ``TypedTool`` instance.

        Uses the class name as the tool name, the class docstring as the description,
        and auto-generates the input schema from the Pydantic model. If the class was
        defined with ``terminating=True``, the tool's ``_meta`` will include the
        ``TERMINATING`` flag.

        Returns:
            A ``TypedTool`` instance ready to be registered in an environment.
        """
        if cls.__is_terminating__:
            meta = {"clinicalagent": ["TERMINATING"]}
        else:
            meta = {}
        tool = TypedTool(
            name=cls.__name__,
            description=cls.__doc__ or "",
            input_model=cls,
            _meta=meta,
        )  # type: ignore
        return tool


class TextStream(BaseModel):
    """
    A streaming event containing an incremental text delta from the agent.

    Yielded by ``Agent.stream()`` as the LLM generates text. Each ``TextStream`` contains
    only the *new* text since the last event, making it suitable for real-time display.

    Attributes:
        type: Discriminator literal, always ``"text"``.
        delta: The new text chunk (not cumulative — only the incremental addition).
    """

    type: Literal["text"] = "text"
    delta: str


class ToolCallStream[T: BaseModel](BaseModel):
    """
    A streaming event indicating the agent has completed a tool call.

    Yielded by ``Agent.stream()`` after a full agent turn that includes a tool action.
    This event is emitted *after* the LLM response is fully received, not during streaming.

    Type Parameter:
        T: The tool input model type, matching the tool's Pydantic schema.

    Attributes:
        type: Discriminator literal, always ``"tool_call"``.
        tool_call: The completed tool call with name and parsed arguments.
    """

    type: Literal["tool_call"] = "tool_call"
    tool_call: CallToolRequestParams[T]
