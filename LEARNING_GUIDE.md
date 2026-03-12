# ClinicalAgent — Learning Guide for Beginners

This guide walks you through the entire `clinicalagent` codebase **in the order you should read it** — starting from the simplest concepts and building up to the full agent loop. Each section explains what the code does, why it's written that way, and what Python concepts you need to know.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Prerequisites — Python Concepts You'll Need](#2-prerequisites--python-concepts-youll-need)
3. [Reading Order](#3-reading-order)
4. [File-by-File Walkthrough](#4-file-by-file-walkthrough)
   - [Phase 1: Configuration](#phase-1-configuration--settingspy)
   - [Phase 2: Data Types](#phase-2-data-types--typespy)
   - [Phase 3: LLM Communication](#phase-3-llm-communication--llmpy)
   - [Phase 4: History Management](#phase-4-history-management--history_utilspy)
   - [Phase 5: Environment](#phase-5-the-environment--environmentpy)
   - [Phase 6: Utilities](#phase-6-utilities--utilspy)
   - [Phase 7: The Agent Loop](#phase-7-the-agent-loop--agentpy)
   - [Phase 8: The Example](#phase-8-the-working-example--examplessimplepy)
5. [The Complete Flow — Trace a Request End-to-End](#5-the-complete-flow--trace-a-request-end-to-end)
6. [Exercises](#6-exercises)

---

## 1. The Big Picture

`clinicalagent` is a framework for building **AI agents** — programs that can:
1. Receive a question from a user
2. Think about what tools to use (search databases, check eligibility, etc.)
3. Call those tools
4. Use the results to formulate an answer
5. Repeat steps 2–4 until the answer is complete

### The Core Loop (memorize this!)

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ 1. Build     │    │ 2. Send to   │    │ 3. Parse  │ │
│  │    Context    │───▶│    LLM       │───▶│  Response │ │
│  │ (get_context) │    │ (streaming)  │    │           │ │
│  └──────────────┘    └──────────────┘    └─────┬─────┘ │
│         ▲                                      │       │
│         │                                      ▼       │
│  ┌──────┴───────┐                       ┌───────────┐  │
│  │ 5. Add tool  │◀──────────────────────│ 4. Handle │  │
│  │    result to │   (if tool was called)│  Response  │  │
│  │    history   │                       │           │  │
│  └──────────────┘                       └─────┬─────┘  │
│                                               │        │
│                                    ┌──────────▼─────┐  │
│                                    │ TERMINATE?     │  │
│                                    │ Yes → Stop     │  │
│                                    │ No  → Loop ↑   │  │
│                                    └────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### The Architecture (only 3 main pieces)

| Component | File | Responsibility |
|-----------|------|----------------|
| **Agent** | `agent.py` | Runs the loop (context → LLM → handle → repeat) |
| **Environment** | `environment.py` | Defines tools, builds prompts, executes tools, decides when to stop |
| **LLM Layer** | `llm.py` | Sends messages to the AI model, parses streaming JSON responses |

Everything else is supporting types and utilities.

---

## 2. Prerequisites — Python Concepts You'll Need

Before reading the code, make sure you understand these concepts. If any are unfamiliar, look them up first.

### Must-Know (used everywhere)

| Concept | Where it appears | Quick explanation |
|---------|-----------------|-------------------|
| **Pydantic `BaseModel`** | Every file | Classes that auto-validate data. You define fields with types, and Pydantic checks them at runtime. |
| **Type hints** (`str`, `int`, `list[str]`, `X \| None`) | Every file | Annotations telling Python (and your IDE) what type a variable should be. `X \| None` means "X or nothing". |
| **`async`/`await`** | `agent.py`, `llm.py`, `environment.py` | Lets Python do other work while waiting for slow operations (like API calls). |
| **`async for`** | `agent.py`, `llm.py` | Iterates over values that arrive one at a time from an async source (like a streaming API). |
| **`yield`** | `agent.py`, `llm.py` | Makes a function a "generator" — it produces values one at a time instead of returning all at once. |
| **`ABC` / `@abstractmethod`** | `environment.py` | Abstract Base Class — defines an interface that subclasses *must* implement. |
| **Inheritance** (`class Child(Parent)`) | `environment.py`, `types.py` | A class that builds on another class, inheriting its methods and adding new ones. |

### Good-to-Know (used in specific places)

| Concept | Where | Quick explanation |
|---------|-------|-------------------|
| **Generics** (`class Foo[T: Bar]`) | `types.py`, `agent.py` | Python 3.13 syntax. `T` is a placeholder type that gets filled in when you use the class. Like a template. |
| **`@overload`** | `agent.py`, `types.py` | Lets you define different type signatures for the same function depending on the arguments. |
| **`Enum`** | `types.py` (`MessageFlag`) | A class with a fixed set of named constants. |
| **`RootModel`** | `types.py` (`History`) | A Pydantic model where the *entire model* is a single value (like a list or tuple), not a dict of fields. |
| **Metaclass** | `types.py` (`BaseToolMeta`) | A "class of a class" — controls how classes themselves are created. Advanced Python. |
| **`Sentinel`** | `types.py` (`TERMINATE`) | A unique object used as a signal value. Like `None` but you can distinguish it from actual `None`. |
| **Jinja2 templates** | `environment.py` | A templating language (like f-strings on steroids) that renders text with variables and logic. |
| **`Literal`** | `types.py` | A type hint that says "this value must be exactly this string", e.g., `Literal["user"]`. |

---

## 3. Reading Order

Read the files in this order. Each file builds on the previous ones:

```
1. settings.py        ← Configuration (simplest file, standalone)
2. types.py           ← All data structures (depends on nothing internal)
3. llm.py             ← LLM communication (depends on types + settings)
4. history_utils.py   ← History summarization (depends on types + llm)
5. utils.py           ← Utility functions (depends on types)
6. environment.py     ← Environment logic (depends on everything above)
7. agent.py           ← The agent loop (depends on environment + llm)
8. examples/simple.py ← A working example (uses the whole framework)
```

---

## 4. File-by-File Walkthrough

### Phase 1: Configuration — `settings.py`

**What this file does**: Loads configuration from YAML files, `.env` files, or environment variables.

**Start here because**: It's the simplest file and introduces Pydantic Settings, which is used throughout.

```python
# Line 1-6: Import pydantic-settings, a library that auto-loads config from files/env vars
from pydantic_settings import (
    BaseSettings,                    # Base class for settings models
    PydanticBaseSettingsSource,      # Type for config sources
    SettingsConfigDict,             # Configuration for the settings class itself
    YamlConfigSettingsSource,       # Reads settings from YAML files
)
```

```python
# Line 9: Settings inherits from BaseSettings (not regular BaseModel!)
# BaseSettings automatically reads values from env vars, .env files, etc.
class Settings(BaseSettings):
    # Lines 10-17: Each field is a configuration value.
    # The type hint (str | None) tells Pydantic what to expect.
    # The default value (= None, = 7) is used if no config source provides it.

    llm_model_name: str | None = None   # e.g., "gpt-4o" — which AI model to use
    llm_base_url: str | None = None     # e.g., "https://api.openai.com/v1" — API endpoint

    llm_api_key: str | None = None      # API authentication key
    llm_api_extra_kw: dict = {}         # Extra params like {"temperature": 0.7}

    max_agent_iterations: int = 7       # Safety limit: agent stops after 7 loops max
    max_history_length: int | None = 11 # Summarize history after 11 messages
    reduce_history_by: int = 7          # Summarize the first 7 messages when triggered
```

```python
    # Lines 20-26: model_config tells Pydantic WHERE to look for settings
    model_config = SettingsConfigDict(
        yaml_file="clinicalagent-config.yaml",  # Look for this YAML file
        yaml_file_encoding="utf-8",
        env_file=".env",                         # Also look for a .env file
        env_file_encoding="utf-8",
        extra="ignore",                          # Don't crash on unknown keys in config
    )
```

```python
    # Lines 28-43: This controls the PRIORITY ORDER of config sources.
    # If the same setting is in both YAML and env vars, which wins?
    # Answer: whichever is FIRST in this tuple.
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings,
                                    env_settings, dotenv_settings,
                                    file_secret_settings):
        return (
            init_settings,                              # 1st: programmatic overrides (highest priority)
            YamlConfigSettingsSource(settings_cls),     # 2nd: clinicalagent-config.yaml
            dotenv_settings,                            # 3rd: .env file
            env_settings,                               # 4th: OS environment variables
            file_secret_settings,                       # 5th: Docker/K8s secret files
        )

# Line 46: Create ONE global instance. This is loaded at import time.
# Every other module does: from .settings import settings
settings = Settings()
```

**Key takeaway**: When you see `settings.max_agent_iterations` anywhere in the code, it's reading from this global object, which was populated from your config files.

---

### Phase 2: Data Types — `types.py`

**What this file does**: Defines ALL the data structures used throughout the framework. This is the longest file — take it one class at a time.

#### Part A: The TERMINATE Sentinel (Line 35)

```python
from typing_extensions import Sentinel

TERMINATE = Sentinel("TERMINATE")
```

**What is a Sentinel?** It's a unique object — not `True`, not `False`, not `None`. It exists solely so you can check `if result is TERMINATE`. The environment returns this value to tell the agent loop "stop now."

**Why not just use `None`?** Because `None` could mean "no result" from a tool call. `TERMINATE` is unambiguous — it can ONLY mean "stop the loop."

#### Part B: MCP Tool Definitions (Lines 39-91)

```python
class McpToolAnnotations(BaseModel):
    title: str | None = None
    readOnlyHint: bool | None = None      # True = this tool only reads data
    destructiveHint: bool | None = None    # True = this tool might delete things
    idempotentHint: bool | None = None     # True = calling twice = same as calling once
    openWorldHint: bool | None = None      # True = calls external systems
    model_config = ConfigDict(extra="allow")  # Allow unknown fields without crashing
```

```python
class McpTool(BaseModel):
    name: str                                    # "search_trials"
    description: str | None = None               # "Search for clinical trials..."
    inputSchema: dict[str, Any]                  # JSON Schema: {"type": "object", "properties": {...}}
    meta: dict[str, Any] | None = Field(
        alias="_meta", default=None               # Note: serialized as "_meta" in JSON
    )
    # The "meta" field stores framework-specific flags.
    # If meta = {"clinicalagent": ["TERMINATING"]}, this tool stops the agent when called.
```

**What is MCP?** Model Context Protocol — an open standard for connecting AI models to tools. Think of `McpTool` as a "tool description card" that tells the LLM what a tool does and what arguments it expects.

#### Part C: Tool Call Parameters (Lines 94-108)

```python
class CallToolRequestParams[T: BaseModel](BaseModel):
    tool_name: str    # Which tool to call, e.g., "search_trials"
    arguments: T      # The arguments, typed to the tool's schema
```

The `[T: BaseModel]` is a **generic type parameter**. Think of it like a variable for types:
- `CallToolRequestParams[SearchTrialsInput]` means `arguments` is of type `SearchTrialsInput`
- `CallToolRequestParams[CheckEligibilityInput]` means `arguments` is of type `CheckEligibilityInput`

This gives you autocomplete and type checking for tool arguments.

#### Part D: Errors (Lines 111-125)

```python
class ClinicalAgentError(Exception):        # Base error for the whole framework
    pass

class MaxAgentIterationsExceededError(ClinicalAgentError):  # Raised when loop limit hit
    pass
```

Simple inheritance: all framework errors extend `ClinicalAgentError`, so you can catch them all with one `except ClinicalAgentError`.

#### Part E: MessageFlag Enum (Lines 128-148)

```python
class MessageFlag(str, Enum):
    is_system_instruction = "is_system_instruction"  # The system prompt
    is_system_response = "is_system_response"        # "You didn't call a tool, try again"
    is_tool_result = "is_tool_result"                # "Tool returned: ..."
    is_summary = "is_summary"                        # "Summary of previous conversation: ..."
```

Flags are **invisible metadata** attached to messages. The LLM never sees them — they're used internally to identify message types (e.g., "which message is the system prompt? Let me replace it with an updated one").

#### Part F: Message and History (Lines 151-295)

```python
class Message(BaseModel):
    id: SkipJsonSchema[int | None] = None   # Position in history (auto-assigned)
    role: Literal["system", "user", "assistant"]  # Who said this
    content: str                              # What they said
    flags: SkipJsonSchema[list[str]] = Field(default_factory=list)  # Internal metadata
```

**What is `SkipJsonSchema`?** It tells Pydantic: "this field exists in Python, but DON'T include it when generating a JSON Schema." This matters because the JSON Schema is sent to the LLM — we don't want the LLM to see internal fields like `id` and `flags`.

```python
class History(RootModel[tuple[Message, ...]]):
    root: tuple[Message, ...]   # The actual list of messages (stored as a tuple)
```

**What is `RootModel`?** A normal `BaseModel` has named fields: `{"name": "...", "age": 42}`. A `RootModel` IS its data — `History` IS a tuple of messages. You create it like: `History.model_validate([{"role": "user", "content": "Hello"}])`.

**Key method — `add_message()` (Line 250):**
```python
def add_message(self, msg=None, *, role="user", content="", flags=None, index=None):
    # Two ways to call:
    # 1. history.add_message(some_message_object)
    # 2. history.add_message(role="user", content="Hello")

    if index is None:
        index = len(self.root)   # Default: append at the end

    # Build a Message if one wasn't provided
    if msg is None:
        msg = Message(id=index, role=role, content=content, flags=flags or [])

    # Insert into tuple: slice before + new message + slice after
    self.root = self.root[:index] + (msg,) + self.root[index:]

    # Re-number all message IDs (0, 1, 2, ...)
    self.ensure_valid_ids(raise_warning=False)
```

**Why a tuple and not a list?** Tuples are immutable per-element, which provides some safety. But note that `add_message` replaces the entire tuple reference — so History is effectively mutable, just not by accident.

#### Part G: Agent Responses (Lines 298-337)

```python
class AgentResponse[T: BaseModel](BaseModel):
    msg_to_user: str | None = None                  # "Let me look that up for you..."
    action: CallToolRequestParams[T] | None = None  # Tool call (if any)
    turn_completed: SkipJsonSchema[bool] = False     # Internal: set by agent loop

class AgentResponseThoughtful[T: BaseModel](AgentResponse[T]):
    thought: str | None = None   # "I should search for NSCLC trials first..."
```

This is what the LLM produces. The LLM outputs JSON like:
```json
{
  "msg_to_user": "Let me find that for you!",
  "action": {
    "tool_name": "search_trials",
    "arguments": {"condition": "NSCLC", "phase": "Phase 3"}
  }
}
```

The framework parses this JSON into an `AgentResponse` object.

#### Part H: TypedTool and BaseToolModel (Lines 372-536)

```python
class TypedTool(McpTool, Generic[T_co]):
    input_model: SkipJsonSchema[T_co] = Field(exclude=True)
```

`TypedTool` = `McpTool` + a Python class for the input. It bridges the gap between:
- JSON Schema (what the LLM sees): `{"type": "object", "properties": {"condition": {"type": "string"}}}`
- Python class (what your code uses): `class SearchTrials(BaseModel): condition: str`

```python
class BaseToolModel(BaseModel, metaclass=BaseToolMeta):
    @classmethod
    def convert_to_tool(cls) -> TypedTool[type[Self]]:
        tool = TypedTool(
            name=cls.__name__,           # Class name becomes tool name
            description=cls.__doc__,     # Docstring becomes tool description
            input_model=cls,             # The class itself is the input model
            _meta=meta,                  # TERMINATING flag if applicable
        )
        return tool
```

This is how you define tools in pure Python:
```python
class SearchTrials(BaseToolModel):
    """Search for clinical trials by condition."""  # ← This becomes the tool description
    condition: str
    phase: str = ""

tool = SearchTrials.convert_to_tool()
# tool.name = "SearchTrials"
# tool.description = "Search for clinical trials by condition."
# tool.inputSchema = {"type": "object", "properties": {"condition": {"type": "string"}, ...}}
```

#### Part I: Streaming Types (Lines 539-571)

```python
class TextStream(BaseModel):
    type: Literal["text"] = "text"
    delta: str   # The NEW text (not all text so far, just the new chunk)

class ToolCallStream[T: BaseModel](BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: CallToolRequestParams[T]
```

These are events yielded by `agent.stream()`. You check `event.type` to know what kind of event it is.

---

### Phase 3: LLM Communication — `llm.py`

**What this file does**: Sends conversation history to an LLM API and parses the streaming JSON response.

#### The Streaming Parser (Lines 46-80)

```python
class StructuredStreamParser[T: BaseModel]:
    def __init__(self, schema: type[T]):
        self.schema = schema  # e.g., AgentResponse[SearchTrials | CheckEligibility]
        self.buffer = ""      # Accumulates all JSON chunks received so far

    def feed(self, chunk: str) -> T | None:
        self.buffer += chunk  # Append new chunk to buffer
        try:
            # Try to parse the INCOMPLETE JSON
            # e.g., '{"msg_to_user": "Let me' → {"msg_to_user": "Let me"}
            parsed = parser.parse(self.buffer)
            result = self.schema.model_validate(parsed)
            return result     # Success! Return partial result
        except Exception:
            return None       # JSON is too incomplete to parse, wait for more
```

**Why partial JSON parsing?** The LLM sends JSON character by character. Without partial parsing, you'd have to wait for the ENTIRE response before showing anything to the user. With `partialjson`, you can show the `msg_to_user` text as it arrives, even before the `action` field is complete.

Example of what the buffer looks like during streaming:
```
Chunk 1: '{"msg_'                          → parse fails, return None
Chunk 2: '{"msg_to_user": "Let me'         → parse succeeds: {msg_to_user: "Let me"}
Chunk 3: '{"msg_to_user": "Let me look"'   → parse succeeds: {msg_to_user: "Let me look"}
...
Final:   '{"msg_to_user": "...", "action": {"tool_name": "search_trials", ...}}'
```

#### The Main Streaming Function (Lines 83-138)

```python
async def stream_agent_response(history, schema, client_config=None):
    # Step 1: If no client config provided, use global settings
    if client_config is None:
        client_config = OpenAiClientConfig(
            llm_model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            extra_kw=settings.llm_api_extra_kw,
        )

    # Step 2: Create the streaming parser for the given schema
    stream_parser = StructuredStreamParser(schema)

    # Step 3: Special case — file-based simulation for testing
    if client_config.base_url.startswith("file:"):
        # Read from a local file instead of calling an API
        # (Used for testing without spending money on API calls)
        ...

    # Step 4: Normal case — call the real API
    client = AsyncOpenAI(base_url=client_config.base_url, api_key=client_config.api_key)

    async with client.responses.stream(
        model=client_config.llm_model_name,
        input=history.compact(),     # Send only role + content (no IDs, flags)
        text_format=schema,          # Tell the API to output JSON matching this schema
        extra_body=client_config.extra_kw,
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                # Feed each text chunk to the parser
                parsed = stream_parser.feed(event.delta)
                if parsed is not None and parsed != last_yielded:
                    last_yielded = parsed
                    yield parsed   # Yield the latest parseable version
```

**The flow**: API sends text chunks → parser accumulates them → when enough JSON is parseable, yield a Pydantic object → repeat.

---

### Phase 4: History Management — `history_utils.py`

**What this file does**: When conversation history gets too long, it summarizes older messages using the LLM.

```python
def last_summary_index(history: History) -> int:
    """Find the index of the most recent summary message."""
    # Scan history in REVERSE to find the last message with the is_summary flag
    return next(
        (idx for idx, msg in reversed(list(enumerate(history.root)))
         if MessageFlag.is_summary in msg.flags),
        0,  # If no summary exists, return 0 (use entire history)
    )
```

```python
async def create_summary_entry(old_history, reduce_by, client_config=None):
    history = old_history.model_copy(deep=True)  # Don't modify the original

    # Take the first `reduce_by` messages and format them as text
    truncated_conv = "\n".join(
        f"{msg.role}: {msg.content}" for msg in history.root[:reduce_by]
    )

    # Ask the LLM to summarize them
    prompt = f"Please summarize the following conversation...\n\n{truncated_conv}"
    summary_response = await structured_agent_response(...)

    # Insert summary message right after the summarized messages
    history.add_message(
        role="system",
        content=f"Summary of previous conversation: {summary_response.summary}",
        flags=[MessageFlag.is_summary],
        index=reduce_by + last_summary_index(history),
    )
    return history
```

**Why summarize?** LLMs have a context window limit (e.g., 128K tokens). If the agent has a long conversation with many tool calls, the history can exceed this limit. Summarization compresses older messages into a single summary while keeping recent messages intact.

---

### Phase 5: The Environment — `environment.py`

**What this file does**: Defines what the agent's "world" looks like — what tools it has, what prompt it sees, how tools are executed.

#### The Abstract Interface (Lines 52-128)

```python
class Environment[T: BaseModel](ABC):
    history: History   # The conversation history (shared with the Agent)

    @abstractmethod
    async def get_context(self, remaining_iterations: int) -> History:
        """Build the prompt + history to send to the LLM."""
        ...

    @abstractmethod
    async def on_agent_message_completed(self, last_response) -> Message | TERMINATE:
        """After LLM responds: execute tools, decide to continue or stop."""
        ...

    @abstractmethod
    async def get_tools(self) -> Iterable[TypedTool]:
        """What tools does the agent have access to?"""
        ...
```

This is an **interface contract**. Any environment MUST implement these 3 methods. The Agent doesn't care HOW they're implemented — it just calls them.

#### The Default Environment (Lines 184-371)

`DefaultEnvironment` is a ready-to-use implementation. You only need to override ONE method: `call_tool()`.

**`get_context()` — How the prompt is built (Lines 226-289):**

```python
async def get_context(self, remaining_iterations):
    # Step 1: If history is too long, summarize it
    if len(self.history.root) - last_summary_index(self.history) > self.max_history_length:
        self.history = await create_summary_entry(...)

    # Step 2: Slice history from the last summary onward
    history = History.model_validate(
        self.history.root[last_summary_index(self.history):]
    )

    # Step 3: Remove the old system prompt (if any)
    if history.root and MessageFlag.is_system_instruction in history.root[0].flags:
        history.root = history.root[1:]

    # Step 4: Render a NEW system prompt using Jinja2 template
    # The template includes: tool descriptions, remaining iterations, extra instructions
    system_prompt = self.system_prompt_template.render(
        tools=tools,
        remaining_iterations=remaining_iterations,
        extra_instructions=...,
        terminating_tools=terminating_tools,
    )

    # Step 5: Prepend it at position 0
    history.add_message(role="system", content=system_prompt, index=0)
    return history
```

**Why rebuild the system prompt every iteration?** Because:
- The `remaining_iterations` count changes (creates urgency in the prompt)
- Tools might change dynamically (if using a callable tool list)
- The history might have been summarized (old system prompt was included in summarized messages)

**`on_agent_message_completed()` — The decision point (Lines 309-358):**

```python
async def on_agent_message_completed(self, last_response):
    # Case 1: LLM didn't call any tool
    if not last_response.action:
        return Message(content="No tool was called. Please continue or terminate.",
                       flags=[MessageFlag.is_system_response])

    # Case 2: LLM called a TERMINATING tool → Stop the loop
    if last_response.action.tool_name in terminating_tools:
        return TERMINATE

    # Case 3: LLM called a regular tool → Execute it and return the result
    tool_result = await self.call_tool(last_response.action)
    return Message(content=f"Tool result: {tool_result}",
                   flags=[MessageFlag.is_tool_result])
```

---

### Phase 6: Utilities — `utils.py`

#### `build_agent_response_schema()` (Lines 33-49)

This function dynamically creates the JSON schema that tells the LLM what to output.

```python
def build_agent_response_schema(disable_thought, tools):
    # Collect all tool input models: [SearchTrialsInput, CheckEligibilityInput, ...]
    # Create a Union type: SearchTrialsInput | CheckEligibilityInput | ...
    tool_type = Union[*tuple(tool.input_model for tool in tools)]

    # Return AgentResponse parameterized with this union
    # The LLM will output JSON where "action.arguments" matches ONE of these tool schemas
    if disable_thought:
        return AgentResponse[tool_type]
    else:
        return AgentResponseThoughtful[tool_type]
```

**Why is this needed?** The LLM needs to know ALL possible tool schemas so it can choose which tool to call and what arguments to provide. This function creates a single Pydantic model that represents "any of these tools."

#### `mcp_tools()` (Lines 52-87)

```python
def mcp_tools(client):
    """Returns a CALLABLE that fetches tools from an MCP server."""
    async def tools():
        async with client:
            return [TypedTool.from_tool(McpTool(**tool.model_dump()))
                    for tool in await client.list_tools()]
    return tools   # Note: returns the FUNCTION, not the result
```

**Why return a function instead of a list?** Because `DefaultEnvironment` accepts either a static list OR a callable. If you pass a callable, tools are re-fetched on every agent iteration. This enables dynamic tool discovery — tools can be added/removed from the MCP server while the agent is running.

---

### Phase 7: The Agent Loop — `agent.py`

This is where everything comes together.

#### `run()` — The Main Loop (Lines 132-208)

```python
async def run(self):
    iteration = 0
    while iteration < self.max_iterations:
        iteration += 1

        # ── STEP 1: Context Engineering ──
        # Environment builds: system prompt + relevant history
        context = await self.environment.get_context(
            self.max_iterations - iteration  # Pass remaining iterations
        )

        # ── STEP 2: Build Response Schema ──
        # Create a Pydantic model that combines all tool schemas
        schema = build_agent_response_schema(
            disable_thought=self.disable_thought,
            tools=await self.environment.get_tools(),
        )

        # ── STEP 3: Stream LLM Response ──
        # Send context to LLM, receive partial AgentResponse objects
        response = None
        async for response in stream_agent_response(context, schema, self.openai_client):
            yield response   # Yield each partial response to the caller

        # ── STEP 4: Mark Turn as Completed ──
        response.turn_completed = True

        # Save the LLM's response to history (as an "assistant" message)
        self.environment.history.add_message(
            role="assistant",
            content=response.model_dump_json(indent=2, ...),  # Serialize to JSON string
        )
        yield response   # Yield one final time with turn_completed=True

        # ── STEP 5: Handle the Response ──
        # If no tool call and we don't require one → agent is done
        if not self.require_terminating_tool_call and response.action is None:
            return

        # Ask environment: execute tool? terminate? continue?
        continuation_message = await self.environment.on_agent_message_completed(response)

        if continuation_message is TERMINATE:
            return   # Environment said to stop

        # ── STEP 6: Add Tool Result to History and Loop Back ──
        self.environment.history.add_message(continuation_message)

    # If we get here, the loop exceeded max_iterations
    raise MaxAgentIterationsExceededError(...)
```

#### `stream()` — User-Friendly Streaming (Lines 210-251)

`run()` yields full `AgentResponse` objects (useful programmatically). `stream()` wraps `run()` and yields just the new text deltas (useful for UIs).

```python
async def stream(self):
    prev_content = ""       # Track what we've already yielded
    has_completed_turn = False

    async for response in self.run():
        # When a full turn completes, emit the tool call (if any)
        if response.turn_completed:
            prev_content = ""
            has_completed_turn = True
            if response.action is not None:
                yield ToolCallStream(tool_call=response.action)
            continue

        # Skip if no text content
        if not response.msg_to_user:
            continue

        # Between turns, emit a separator ("\n\n")
        if has_completed_turn:
            yield TextStream(delta=self.message_separation_token)
            has_completed_turn = False

        # Yield only the NEW text (not the full cumulative text)
        if response.msg_to_user.startswith(prev_content):
            new_content = response.msg_to_user[len(prev_content):]
            if new_content:
                prev_content = response.msg_to_user
                yield TextStream(delta=new_content)
```

**Why the `prev_content` tracking?** The LLM parser yields *cumulative* text:
- Yield 1: `msg_to_user = "Let"`
- Yield 2: `msg_to_user = "Let me"`
- Yield 3: `msg_to_user = "Let me look"`

But the UI wants *incremental* text: `"Let"`, `" me"`, `" look"`. So `stream()` subtracts `prev_content` from the current `msg_to_user` to extract only the delta.

---

### Phase 8: The Working Example — `examples/simple.py`

This ties everything together. Read it as a recipe:

```python
# STEP 1: Define tools using FastMCP
mcp = FastMCP()

@mcp.tool
def day_of_date(date: str) -> str:
    """Get the day of the week for a given date."""
    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    return dt.strftime("%A")

# STEP 2: Create client (connects to the MCP server)
client = Client(mcp)

# STEP 3: Create environment (subclass DefaultEnvironment, implement call_tool)
class SimpleEnvironment[T: BaseToolModel](DefaultEnvironment[T]):
    async def call_tool(self, action):
        async with client:
            result = await client.call_tool(
                name=action.tool_name,
                arguments=action.arguments.model_dump()
            )
        return "\n".join(res.text for res in result.content if res.type == "text")

# STEP 4: Wire everything together
env = SimpleEnvironment(tools=mcp_tools(client))
agent = Agent(environment=env, disable_thought=True, require_terminating_tool_call=False)

# STEP 5: Run
async def main():
    agent.environment.history.add_message(role="user", content="What day is Sept 2, 2026?")
    async for event in agent.stream():
        print(event.model_dump_json(indent=2))

asyncio.run(main())
```

---

## 5. The Complete Flow — Trace a Request End-to-End

Let's trace what happens when a user asks: **"What day of the week is September 2, 2026?"**

```
1. User query added to history:
   history = [Message(role="user", content="What day of the week is September 2, 2026?")]

2. agent.run() starts iteration 1:

3. environment.get_context(6) called:
   → Renders system prompt with tool descriptions (day_of_date, days_between_dates)
   → Prepends system prompt to history
   → Returns: [system_prompt, user_query]

4. stream_agent_response() called:
   → Sends [system_prompt, user_query] to the LLM
   → LLM streams back JSON:
     {"msg_to_user": "Let me check that for you!", "action": {"tool_name": "day_of_date", "arguments": {"date": "2026-09-02"}}}
   → Parser yields partial AgentResponse objects as JSON arrives

5. Agent saves LLM response to history:
   history = [system_prompt, user_query, assistant_response]

6. environment.on_agent_message_completed() called:
   → "day_of_date" is NOT a terminating tool
   → Calls self.call_tool(action)
   → MCP server runs day_of_date("2026-09-02") → returns "Wednesday"
   → Returns Message(content="Tool result: Wednesday", flags=[is_tool_result])

7. Agent adds tool result to history:
   history = [system_prompt, user_query, assistant_response, tool_result]

8. Loop continues — iteration 2:

9. environment.get_context(5) called:
   → Renders NEW system prompt (remaining_iterations=5)
   → Returns: [new_system_prompt, user_query, assistant_response, tool_result]

10. stream_agent_response() called again:
    → LLM sees the tool result and generates final answer:
      {"msg_to_user": "September 2, 2026 falls on a Wednesday!"}
    → Note: no "action" field this time — the LLM is done

11. require_terminating_tool_call=False, and action is None:
    → Agent terminates naturally

12. stream() yields:
    → TextStream(delta="Let me check that for you!")   (from iteration 1)
    → ToolCallStream(tool_call=...)                    (the day_of_date call)
    → TextStream(delta="\n\n")                         (separator between turns)
    → TextStream(delta="September 2, 2026 falls on a Wednesday!")  (from iteration 2)
```

---

## 6. Exercises

After reading the code, try these exercises to deepen your understanding:

### Exercise 1: Trace the Types (pen and paper)
Draw a diagram showing the generic type parameter `T` as it flows from:
`BaseToolModel subclass` → `TypedTool[type[T]]` → `build_agent_response_schema()` → `AgentResponse[T]` → `CallToolRequestParams[T]`

### Exercise 2: Add a New Tool
In `examples/simple.py`, add a third tool `is_weekend(date: str) -> bool` that returns whether a date falls on a Saturday or Sunday. Run the agent with the query "Is December 25, 2026 a weekend?"

### Exercise 3: Create a Terminating Tool
Create a tool using `BaseToolModel` with `terminating=True`:
```python
class FinalAnswer(BaseToolModel, terminating=True):
    """Provide the final answer to the user."""
    answer: str
```
Then create an agent with `require_terminating_tool_call=True` and observe how the behavior changes.

### Exercise 4: Understand History Summarization
Add print statements inside `create_summary_entry()` and `last_summary_index()`. Then set `max_history_length=5` and `reduce_history_by=3`, and run a multi-turn conversation. Watch when summarization triggers and what the summary looks like.

### Exercise 5: Custom Environment
Create a new Environment subclass that doesn't use MCP at all. Instead, define tools as `BaseToolModel` subclasses and execute them directly in `call_tool()` using `isinstance()` checks. This will teach you the non-MCP path.

---

## Quick Reference: File → Purpose → Key Classes

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `settings.py` | Configuration loading | `Settings`, `settings` (singleton) |
| `types.py` | All data structures | `Message`, `History`, `AgentResponse`, `TypedTool`, `BaseToolModel`, `TERMINATE` |
| `llm.py` | LLM API communication | `StructuredStreamParser`, `stream_agent_response()`, `structured_agent_response()` |
| `history_utils.py` | History compression | `last_summary_index()`, `create_summary_entry()` |
| `utils.py` | Helpers | `build_agent_response_schema()`, `mcp_tools()` |
| `environment.py` | Agent's world definition | `Environment` (abstract), `DefaultEnvironment` |
| `agent.py` | The main loop | `Agent.run()`, `Agent.stream()` |
| `__init__.py` | Public API | Re-exports everything users need |
