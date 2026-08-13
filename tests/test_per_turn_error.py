"""Per-turn error hook (issue #6).

A turn whose LLM stream raises is the Environment's to judge:
``on_turn_failed(exc)`` returning a Message recovers the loop (the message
joins history, the next iteration proceeds); the default ``None`` propagates
exactly as before.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

import clinicalagent.agent as agent_mod
from clinicalagent.agent import Agent
from clinicalagent.environment import DefaultEnvironment, Environment
from clinicalagent.types import (
    AgentResponse,
    History,
    Message,
    MessageFlag,
    OpenAiClientConfig,
)

CFG = OpenAiClientConfig(llm_model_name="m", base_url="http://x", api_key="k")


class _Args(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Env(Environment):
    def __init__(self, recover: bool):
        self.history = History.model_validate([])
        self.recover = recover
        self.seen_exc: Exception | None = None

    async def get_context(self, remaining_iterations: int) -> History:
        return History.model_validate([])

    async def get_tools(self):
        return []

    async def on_agent_message_completed(self, last_response):
        return Message(
            role="user", content="continue", flags=[MessageFlag.is_tool_result]
        )

    async def on_turn_failed(self, exc):
        self.seen_exc = exc
        if not self.recover:
            return None
        return Message(
            role="user",
            content="Tool result: the model call failed; continue without it.",
            flags=[MessageFlag.is_tool_result],
        )


def _stream_that_fails_once(responses):
    calls = {"n": 0}

    async def fake_stream(context, schema, client_config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("provider down")
        yield responses[min(calls["n"] - 2, len(responses) - 1)]

    return fake_stream


@pytest.mark.asyncio
async def test_recovering_hook_continues_the_loop(monkeypatch):
    monkeypatch.setattr(
        agent_mod,
        "stream_agent_response",
        _stream_that_fails_once([AgentResponse(msg_to_user="done")]),
    )
    env = _Env(recover=True)
    agent = Agent(environment=env, openai_client=CFG)

    seen = [r async for r in agent.run()]

    assert isinstance(env.seen_exc, ConnectionError)
    # The recovery message joined history and the loop went on to finish.
    assert any("failed" in (m.content or "") for m in env.history.root)
    assert seen[-1].msg_to_user == "done"


@pytest.mark.asyncio
async def test_default_hook_propagates(monkeypatch):
    monkeypatch.setattr(
        agent_mod,
        "stream_agent_response",
        _stream_that_fails_once([AgentResponse(msg_to_user="done")]),
    )
    env = _Env(recover=False)
    agent = Agent(environment=env, openai_client=CFG)

    with pytest.raises(ConnectionError):
        async for _ in agent.run():
            pass
    assert isinstance(env.seen_exc, ConnectionError)


@pytest.mark.asyncio
async def test_base_environment_default_is_propagate():
    env = DefaultEnvironment(tools=[])
    assert await env.on_turn_failed(RuntimeError("boom")) is None
