"""
Utility functions for the clinicalagent framework.

This module provides:
    - ``build_agent_response_schema()``: Dynamically constructs a Pydantic response schema
      based on the available tools, used to tell the LLM what JSON structure to produce.
    - ``mcp_tools()``: Creates a callable that fetches tools from a FastMCP client,
      suitable for passing to ``DefaultEnvironment(tools=...)``.
"""

from typing import Iterable, Literal, Union, cast, overload

from pydantic import BaseModel

from .types import (
    AgentResponse,
    AgentResponseThoughtful,
    BaseToolModel,
    McpTool,
    TypedTool,
)


@overload
def build_agent_response_schema[T: BaseModel](
    disable_thought: Literal[True], tools: Iterable[TypedTool[type[T]]]
) -> type[AgentResponse[T]]: ...
@overload
def build_agent_response_schema[T: BaseModel](
    disable_thought: Literal[False], tools: Iterable[TypedTool[type[T]]]
) -> type[AgentResponseThoughtful[T]]: ...
def build_agent_response_schema[T: BaseModel](
    disable_thought: bool, tools: Iterable[TypedTool[type[T]]]
) -> type[AgentResponse[T] | AgentResponseThoughtful[T]]:
    """
    Dynamically build a Pydantic response schema that includes all available tool input types.

    The LLM must output JSON conforming to either ``AgentResponse`` or ``AgentResponseThoughtful``.
    The ``action`` field's type is a union of all tool input models, so the LLM can call any
    registered tool. This function constructs that union type at runtime.

    How it works:
        1. Collects all ``input_model`` classes from the provided tools.
        2. Creates a ``Union[ToolA, ToolB, ...]`` type from them.
        3. Parameterizes ``AgentResponse[Union[...]]`` or ``AgentResponseThoughtful[Union[...]]``
           depending on ``disable_thought``.

    Args:
        disable_thought: If True, returns ``AgentResponse[T]`` (no thought field).
            If False, returns ``AgentResponseThoughtful[T]`` (includes ``thought`` field).
        tools: The tools available in the current environment.

    Returns:
        A parameterized Pydantic model class suitable for structured output parsing.
    """
    if len(list(tools)) == 0:
        tool_type = None
    else:
        tool_type = Union[*tuple(tool.input_model for tool in tools)]
    if disable_thought:
        return AgentResponse[tool_type]  # type: ignore
    else:
        return AgentResponseThoughtful[tool_type]  # type: ignore


def mcp_tools(client):
    """
    Create a callable that fetches tools from a FastMCP client.

    Returns an async function that, when awaited, connects to the MCP server,
    lists available tools, and converts them to ``TypedTool`` instances. This callable
    is designed to be passed directly to ``DefaultEnvironment(tools=mcp_tools(client))``.

    Because it returns a callable (not a static list), the tool list is re-fetched on
    each agent iteration, enabling dynamic tool discovery.

    Args:
        client: A ``fastmcp.Client`` instance connected to an MCP server.

    Returns:
        An async callable that returns ``list[TypedTool[type[BaseToolModel]]]``.

    Raises:
        ImportError: If the ``fastmcp`` package is not installed.
            Install via ``pip install clinicalagent[mcp]``.

    Example:
        >>> from fastmcp import Client, FastMCP
        >>> mcp_server = FastMCP()
        >>> client = Client(mcp_server)
        >>> env = DefaultEnvironment(tools=mcp_tools(client))
    """
    try:
        from fastmcp import Client

        client = cast(Client, client)
    except ImportError:
        raise ImportError("fastmcp is not installed.")

    async def tools() -> list[TypedTool[type[BaseToolModel]]]:
        async with client:
            return [
                TypedTool.from_tool(McpTool(**tool.model_dump()))
                for tool in await client.list_tools()
            ]

    return tools
