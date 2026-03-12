"""
clinicalagent — A lightweight Python framework for building AI agents tailored to clinical trial workflows.

This package provides the core building blocks for creating agentic AI systems that can:
- Interact with clinical data sources via tool calling (MCP-compatible)
- Maintain conversation history with automatic summarization
- Stream structured responses from LLMs in real time
- Support auditable "thought process" traces for regulatory transparency

Architecture Overview:
    Agent          — The main orchestrator that runs the agentic loop (context → LLM → tool call → repeat).
    Environment    — Abstract interface defining tools, context engineering, and termination logic.
    DefaultEnvironment — A batteries-included Environment with system prompt rendering, history
                         summarization, and tool execution hooks.
    History        — An ordered, ID-validated sequence of Messages forming the conversation context.
    TypedTool      — A type-safe wrapper around MCP tool definitions, binding a Pydantic model to each tool's input schema.
    BaseToolModel  — Base class for defining tool input schemas with optional termination semantics.

Typical usage:
    1. Define tools (via MCP server or BaseToolModel subclasses)
    2. Create an Environment (or subclass DefaultEnvironment)
    3. Instantiate an Agent with the environment
    4. Add a user query to history, then iterate over agent.run() or agent.stream()

Example:
    >>> from clinicalagent import Agent, DefaultEnvironment
    >>> env = MyEnvironment(tools=[...])
    >>> env.history.add_message(role="user", content="Find trials for NSCLC")
    >>> agent = Agent(environment=env, disable_thought=True)
    >>> async for event in agent.stream():
    ...     print(event.delta)
"""

from .agent import Agent
from .environment import DefaultEnvironment, Environment
from .types import (
    AgentResponse,
    BaseToolModel,
    CallToolRequestParams,
    History,
    MaxAgentIterationsExceededError,
    Message,
    MessageFlag,
    Token,
    TypedTool,
    ClinicalAgentError,
)

__all__ = [
    "AgentResponse",
    "CallToolRequestParams",
    "History",
    "ClinicalAgentError",
    "MaxAgentIterationsExceededError",
    "Message",
    "MessageFlag",
    "Token",
    # agent
    "Agent",
    # environment
    "Environment",
    "DefaultEnvironment",
    "TypedTool",
    "BaseToolModel",
]


def main() -> None:
    print("Hello from clinicalagent!")
