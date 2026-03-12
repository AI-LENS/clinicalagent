"""
Environment definitions for the clinicalagent framework.

An Environment is the agent's "world" — it defines:
    1. **What tools are available** (``get_tools()``)
    2. **What context the LLM sees** (``get_context()``) — system prompt, history, instructions
    3. **What happens after the LLM responds** (``on_agent_message_completed()``) — tool execution,
       termination logic, continuation prompts

This module provides:
    - ``Environment`` — Abstract base class defining the interface.
    - ``DefaultEnvironment`` — Batteries-included implementation with Jinja2-rendered system prompts,
      automatic history summarization, tool execution hooks, and termination detection.

Architecture:
    The ``Agent`` class is intentionally simple — it just runs the loop. All domain-specific
    behavior lives in the Environment. This separation means you can swap environments without
    changing the agent, enabling different clinical workflows (trial matching, adverse event
    monitoring, etc.) to share the same agent loop.

Customization:
    To build a custom environment, you typically:
    1. Subclass ``DefaultEnvironment``
    2. Override ``call_tool()`` to execute tools against your backend (MCP server, REST API, etc.)
    3. Optionally override ``get_context()`` for custom prompt engineering
    4. Optionally override ``on_agent_message_completed()`` for custom termination logic
"""

import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Iterable

from jinja2 import Template
from pydantic import BaseModel

from .history_utils import create_summary_entry, last_summary_index
from .settings import settings
from .types import (
    TERMINATE,
    AgentResponse,
    CallToolRequestParams,
    History,
    Message,
    MessageFlag,
    OpenAiClientConfig,
    TypedTool,
)

logger = logging.getLogger(__name__)


class Environment[T: BaseModel](ABC):
    """
    Abstract base class defining the interface for an agent environment.

    An Environment controls three aspects of agent behavior:
        1. **Context engineering** — What the LLM sees each iteration (``get_context``).
        2. **Post-response handling** — What happens after the LLM produces a response,
           including tool execution and termination decisions (``on_agent_message_completed``).
        3. **Tool availability** — Which tools the agent can call (``get_tools``).

    Type Parameter:
        T: The base type for tool input models. All tools in this environment must have
           input schemas that are subtypes of ``T``.

    Subclassing:
        Implement all three abstract methods. For most use cases, subclass
        ``DefaultEnvironment`` instead — it provides standard implementations
        and you only need to override ``call_tool()``.

    Attributes:
        history: The mutable conversation history. The agent loop reads from and writes to this.
    """

    history: History

    @abstractmethod
    async def get_context(self, remaining_iterations: int) -> History:
        """
        Build and return the conversation context for the current agent iteration.

        This method is called at the start of each agent loop iteration. It should
        return a History object containing everything the LLM needs to see — typically
        a system prompt followed by relevant conversation history.

        Args:
            remaining_iterations: How many iterations the agent has left before hitting
                ``max_iterations``. Useful for injecting urgency into the system prompt.

        Returns:
            A History object to send to the LLM as context.
        """
        raise NotImplementedError()

    @abstractmethod
    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | TERMINATE:
        """
        Handle the agent's response after each completed LLM turn.

        This is the main decision point in the agent loop. It should:
            1. Inspect the response (did the agent make a tool call? which tool?)
            2. Execute any tool calls and collect results
            3. Decide whether to continue (return a Message) or stop (return TERMINATE)

        Args:
            last_response: The fully parsed agent response from the most recent LLM call.

        Returns:
            ``TERMINATE`` to stop the agent loop.
            A ``Message`` to add to history and continue the loop (typically containing
            the tool result or a continuation prompt).
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_tools(self) -> Iterable[TypedTool[type[T]]]:
        """
        Return the tools available to the agent in this environment.

        Called at the start of each iteration to build the system prompt and response schema.
        The tool list can be dynamic (e.g., fetched from an MCP server each time).

        Returns:
            An iterable of ``TypedTool`` instances.
        """
        raise NotImplementedError()


# ── Jinja2 prompt templates ──────────────────────────────────────────────────
# These templates are rendered by DefaultEnvironment.get_context() on each iteration.
# Template variables: tools, remaining_iterations, extra_instructions, terminating_tools.

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI agent with access to the following tools:

{% for tool in tools %}
- {{ tool.name }}: {{ tool.description | replace('\n', '\n    ') }}
    Input Schema: {{ tool.inputSchema | safe }}
{% endfor %}

When you decide to use a tool, provide the tool name and arguments in your response. After the tool call, you will receive the result which you should use to continue the conversation.

You must return a valid json object as per the schema provided.

Example response:
{
    "msg_to_user": "<some message to keep user engaged>", (message to user should not contain any error, tool call or technical information)
    "action": {
        "name": "tool_name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    }
}

{% if remaining_iterations <= 7 %}
CRITICAL: You have {{ remaining_iterations }} iterations remaining. Minimize tool calls.
{% else %}
CRITICAL: Try to complete your task in as few iterations as possible.
{% endif %}

{% if terminating_tools|length != 0 %}
TERMINATION REQUIREMENT: As soon as you have sufficient information to answer the user's query, you MUST immediately call one of these terminating tools: {{ terminating_tools | map(attribute='name') | join(', ') }}. Do NOT make additional tool calls after you can answer. DO NOT continue exploring unnecessarily.
{% endif %}

{% if extra_instructions %}
{{ extra_instructions }}

Complete the task and IMMEDIATELY end conversation with the appropriate terminating tool once done. Do NOT wait or ask for confirmation.
{% endif %}
"""

CONTINUATION_TEMPLATE = """No tool was called in the last response. If you have not yet reached a conclusion, please feel free to explore.
{% if terminating_tools|length != 0 %}
If you have, you must conclude by calling any one of the terminating tool(s): {{ terminating_tools | join(', ') }}.
{% endif %}"""
"""Prompt sent to the LLM when it produces a response without any tool call and
``require_terminating_tool_call`` is effectively active via the DefaultEnvironment.
Nudges the LLM to either explore further or terminate properly."""


class DefaultEnvironment[T: BaseModel](Environment[T]):
    """
    Default environment implementation providing standard agent behavior.

    This class manages the agent's context, tool execution, and conversation flow, including:
    - Maintaining and summarizing conversation history
    - Rendering system prompts with available tools and instructions
    - Handling tool execution and termination logic
    - Supporting both synchronous and asynchronous tool lists
    - Integrating with OpenAI client configuration if provided
    """

    def __init__(
        self,
        tools: list[TypedTool[type[T]]]
        | Callable[[], Awaitable[list[TypedTool[type[T]]]]],
        extra_instructions: Callable[[], str] | str | None = None,
        history: History | None = None,
        max_history_length: int | None = settings.max_history_length,
        reduce_history_by: int = settings.reduce_history_by,
        openai_client: OpenAiClientConfig | None = None,
    ):
        """
        Initialize a DefaultEnvironment instance.

        Args:
            tools: List of available tools, or a callable returning a list of tools (can be async).
            extra_instructions: Optional string or callable providing additional instructions for the system prompt.
            history: Optional conversation history. If not provided, a new empty history is created.
            max_history_length: Maximum number of history entries to keep before summarizing.
            reduce_history_by: Number of entries to reduce when summarizing history.
            openai_client: Optional OpenAI client configuration for summarization.
        """
        self.extra_instructions = extra_instructions
        self.system_prompt_template: Template = Template(DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self.continuation_template: Template = Template(CONTINUATION_TEMPLATE)
        self.history = history or History.model_validate([])
        self.max_history_length = max_history_length
        self.reduce_history_by = reduce_history_by
        self.openai_client = openai_client
        self.provided_tools = tools

    async def get_context(self, remaining_iterations: int) -> History:
        """
        Build and return the conversation context for the agent.

        This method:
            - Summarizes history if it exceeds the maximum length
            - Removes any previous system instruction from the history
            - Renders and prepends a new system prompt with the current tools, remaining iterations, and extra instructions

        Args:
            remaining_iterations: Number of iterations left for the agent to complete its task

        Returns:
            Updated History object with the new system prompt prepended
        """

        if (
            self.max_history_length is not None
            and len(self.history.root) - last_summary_index(self.history)
            > self.max_history_length
        ):
            self.history = await create_summary_entry(
                old_history=self.history,
                reduce_by=self.reduce_history_by,
                client_config=self.openai_client,
            )

        # Extract relevant history after last summary
        history = History.model_validate(
            self.history.root[last_summary_index(self.history) :]
        )
        if history.root and MessageFlag.is_system_instruction in history.root[0].flags:
            history.root = history.root[1:]

        # Render system prompt
        tools = await self.get_tools()
        terminating_tools = [
            tool
            for tool in tools
            if (
                tool.meta
                and "clinicalagent" in tool.meta
                and "TERMINATING" in tool.meta["clinicalagent"]
            )
        ]
        system_prompt = self.system_prompt_template.render(
            tools=tools,
            remaining_iterations=remaining_iterations,
            extra_instructions=self.extra_instructions()
            if callable(self.extra_instructions)
            else self.extra_instructions,
            terminating_tools=terminating_tools,
        )

        # Prepend system prompt to history
        history.add_message(
            role="system",
            content=system_prompt,
            flags=[MessageFlag.is_system_instruction],
            index=0,
        )
        logger.debug("History:\n" + str(history.compact()))

        return history

    async def call_tool(self, action: CallToolRequestParams) -> str | None:
        """
        Execute a tool action. (To be implemented by subclasses.)

        Args:
            action: The tool action to execute (parameters and tool name).

        Returns:
            The result of the tool execution as a string, or None to indicate termination.

        Raises:
            NotImplementedError: This base implementation must be overridden in a subclass.
        """
        raise NotImplementedError(
            "DefaultEnvironment does not implement tool calling. "
            "Subclass and override call_tool() to provide tool execution logic."
        )

    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | TERMINATE:
        """
        Handle the agent's response after each message is completed.

        This method determines whether to terminate the conversation, execute a tool, or prompt the agent to continue:
            - If no tool action was taken, prompts the agent to continue or terminate
            - If a terminating tool was called, returns TERMINATE
            - Otherwise, executes the requested tool and returns the result as a Message

        Args:
            last_response: The most recent agent response to evaluate

        Returns:
            TERMINATE if a terminating tool was called
            Message with tool result if a tool was executed
            Message prompting continuation if no action was taken
        """
        terminating_tools = [
            tool.name
            for tool in await self.get_tools()
            if (
                tool.meta
                and "clinicalagent" in tool.meta
                and "TERMINATING" in tool.meta["clinicalagent"]
            )
        ]
        # Error if no action was taken
        if not last_response.action:
            return Message(
                role="user",
                content=self.continuation_template.render(
                    terminating_tools=terminating_tools
                ),
                flags=[MessageFlag.is_system_response],
            )

        if len(terminating_tools) == 0:
            logger.warning("No terminating tools found in environment.")
        if last_response.action.tool_name in terminating_tools:
            return TERMINATE

        # Execute the tool and return the result as a Message
        tool_result = await self.call_tool(last_response.action)
        return Message(
            role="user",
            content=f"Tool result: {tool_result}",
            flags=[MessageFlag.is_tool_result],
        )

    async def get_tools(self) -> list[TypedTool[type[T]]]:
        """
        Return the list of tools available to the agent in this environment.

        Returns:
            List of BaseTool objects (may be resolved from a callable or returned directly)
        """
        return (
            await self.provided_tools()
            if callable(self.provided_tools)
            else self.provided_tools
        )
