"""Per-iteration client config (issue #3).

The Environment owns model choice per turn: ``Agent.run`` consults
``environment.get_llm_config()`` each iteration and falls back to the Agent's
construction-time config when it returns None. This is what lets one agent
run a script-authoring turn on a different model than its driver turns.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

import clinicalagent.agent as agent_mod
from clinicalagent.agent import Agent
from clinicalagent.environment import DefaultEnvironment, Environment
from clinicalagent.types import (
    AgentResponse,
    CallToolRequestParams,
    History,
    Message,
    MessageFlag,
    OpenAiClientConfig,
)

DEFAULT_CFG = OpenAiClientConfig(
    llm_model_name="driver", base_url="http://d", api_key="k"
)
TURN_CFG = OpenAiClientConfig(llm_model_name="writer", base_url="http://w", api_key="k")


class _Args(BaseModel):
    model_config = ConfigDict(extra="allow")


def _turn_response(with_action: bool) -> AgentResponse:
    if with_action:
        return AgentResponse(
            msg_to_user="working",
            action=CallToolRequestParams(tool_name="noop", arguments=_Args()),
        )
    return AgentResponse(msg_to_user="done")


class _ScriptedEnv(Environment):
    """Two-turn environment: continues once, then the agent ends naturally."""

    def __init__(self, configs: list[OpenAiClientConfig | None]):
        self.history = History.model_validate([])
        self._configs = list(configs)

    async def get_context(self, remaining_iterations: int) -> History:
        return History.model_validate([])

    async def get_tools(self):
        return []

    async def on_agent_message_completed(self, last_response):
        return Message(
            role="user", content="continue", flags=[MessageFlag.is_tool_result]
        )

    async def get_llm_config(self):
        return self._configs.pop(0) if self._configs else None


def _scripted_stream(recorded: list, responses: list[AgentResponse]):
    calls = {"n": 0}

    async def fake_stream(context, schema, client_config=None):
        recorded.append(client_config)
        response = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        yield response

    return fake_stream


@pytest.mark.asyncio
async def test_agent_uses_environment_config_per_iteration(monkeypatch):
    recorded: list = []
    monkeypatch.setattr(
        agent_mod,
        "stream_agent_response",
        _scripted_stream(recorded, [_turn_response(True), _turn_response(False)]),
    )
    env = _ScriptedEnv(configs=[TURN_CFG, None])
    agent = Agent(environment=env, openai_client=DEFAULT_CFG)

    async for _ in agent.run():
        pass

    # Turn 1 ran on the environment's override; turn 2 fell back to the
    # Agent's construction-time config.
    assert recorded == [TURN_CFG, DEFAULT_CFG]


@pytest.mark.asyncio
async def test_agent_falls_back_to_default_when_env_returns_none(monkeypatch):
    recorded: list = []
    monkeypatch.setattr(
        agent_mod,
        "stream_agent_response",
        _scripted_stream(recorded, [_turn_response(False)]),
    )
    env = _ScriptedEnv(configs=[None])
    agent = Agent(environment=env, openai_client=DEFAULT_CFG)

    async for _ in agent.run():
        pass

    assert recorded == [DEFAULT_CFG]


@pytest.mark.asyncio
async def test_default_environment_defaults_to_no_override():
    env = DefaultEnvironment(tools=[])
    assert await env.get_llm_config() is None
