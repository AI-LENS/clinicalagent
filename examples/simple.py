"""
Simple example demonstrating the clinicalagent framework with date calculation tools.

This example shows the minimal steps to create and run an agent:
    1. Define tools using FastMCP (any MCP-compatible server works).
    2. Create an MCP client connected to those tools.
    3. Subclass ``DefaultEnvironment`` and implement ``call_tool()`` to execute MCP tool calls.
    4. Create an ``Agent`` with the environment and stream responses.

Prerequisites:
    - Install with MCP support: ``pip install clinicalagent[mcp]``
    - Configure LLM in ``clinicalagent-config.yaml`` (see examples/clinicalagent-config.yaml)

Run:
    python examples/simple.py
"""

import asyncio

from fastmcp import Client, FastMCP

from clinicalagent import Agent, DefaultEnvironment
from clinicalagent.types import BaseToolModel, CallToolRequestParams
from clinicalagent.utils import mcp_tools

# ── Step 1: Define tools using FastMCP ───────────────────────────────────────
# ClinicalAgent supports any MCP-compatible server. Here we define two simple
# date calculation tools for demonstration purposes.
mcp = FastMCP()


@mcp.tool
def day_of_date(date: str) -> str:
    """Get the day of the week for a given date string in YYYY-MM-DD format."""
    import datetime

    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    return dt.strftime("%A")


@mcp.tool
def days_between_dates(start_date: str, end_date: str) -> int:
    """Calculate the number of days between two dates in YYYY-MM-DD format."""
    import datetime

    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    delta = end_dt - start_dt
    return abs(delta.days)


# ── Step 2: Create the MCP client ────────────────────────────────────────────
# The Client wraps the FastMCP server, enabling tool discovery and execution.
client = Client(mcp)


# ── Step 3: Define the environment ───────────────────────────────────────────
# Subclass DefaultEnvironment and implement call_tool() to define how tools are executed.
# DefaultEnvironment handles everything else: system prompt rendering, history management,
# termination detection, and continuation prompts.
class SimpleEnvironment[T: BaseToolModel](DefaultEnvironment[T]):
    async def call_tool(self, action: CallToolRequestParams[T]) -> str | None:
        """Execute a tool call by forwarding it to the MCP server."""
        async with client:
            result = await client.call_tool(
                name=action.tool_name, arguments=action.arguments.model_dump()
            )
        return "\n".join(res.text for res in result.content if res.type == "text")


# ── Step 4: Create the agent and run ─────────────────────────────────────────
# - tools=mcp_tools(client): Dynamically fetches tools from the MCP server each iteration.
# - disable_thought=True: Don't include the LLM's internal reasoning in responses.
# - require_terminating_tool_call=False: Allow the agent to stop naturally when it has
#   no more tool calls to make (instead of requiring an explicit termination tool).
env = SimpleEnvironment(tools=mcp_tools(client))
agent = Agent(
    environment=env, disable_thought=True, require_terminating_tool_call=False
)


async def main(query: str | None = None):
    """Run the agent with a query, streaming events to stdout."""
    if not query:
        query = input("Query: ")
    # Add the user's query to the conversation history
    agent.environment.history.add_message(role="user", content=query)
    # Stream events: each event is either a TextStream (incremental text) or
    # ToolCallStream (completed tool invocation).
    async for event in agent.stream():
        print(event.model_dump_json(indent=2))
        print("------------")


asyncio.run(main(query="What day of the week is September 2, 2026?"))
