"""Token-usage surfacing (issue #3).

Both LLM helpers report the provider's token usage on the parsed result when
the result's schema declares a ``usage`` field (AgentResponse does). Usage is
``None`` when the provider omits it — never an error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import clinicalagent.llm as llm_mod
from clinicalagent.llm import stream_agent_response, structured_agent_response
from clinicalagent.types import AgentResponse, History, OpenAiClientConfig

CFG = OpenAiClientConfig(llm_model_name="m", base_url="http://fake", api_key="k")


class _DeltaEvent:
    type = "response.output_text.delta"

    def __init__(self, delta: str):
        self.delta = delta


class _CompletedEvent:
    type = "response.completed"

    def __init__(self, usage):
        self.response = SimpleNamespace(usage=usage)


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _fake_openai(events=None, parse_response=None):
    class _FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            async def create(**kw):
                return _FakeStream(events or [])

            async def parse(**kw):
                return parse_response

            self.responses = SimpleNamespace(create=create, parse=parse)

    return _FakeAsyncOpenAI


async def _last(agen):
    result = None
    async for item in agen:
        result = item
    return result


@pytest.mark.asyncio
async def test_streamed_usage_lands_on_final_response(monkeypatch):
    usage = SimpleNamespace(input_tokens=120, output_tokens=40)
    events = [
        _DeltaEvent('{"msg_to_'),
        _DeltaEvent('user": "hi"}'),
        _CompletedEvent(usage),
    ]
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", _fake_openai(events=events))

    final = await _last(
        stream_agent_response(History.model_validate([]), AgentResponse, CFG)
    )

    assert final is not None and final.msg_to_user == "hi"
    assert final.usage is not None
    assert final.usage.input_tokens == 120
    assert final.usage.output_tokens == 40


@pytest.mark.asyncio
async def test_streamed_usage_is_none_when_provider_omits_it(monkeypatch):
    events = [_DeltaEvent('{"msg_to_user": "hi"}'), _CompletedEvent(None)]
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", _fake_openai(events=events))

    final = await _last(
        stream_agent_response(History.model_validate([]), AgentResponse, CFG)
    )

    assert final is not None and final.usage is None


@pytest.mark.asyncio
async def test_simulated_file_stream_reports_no_usage(tmp_path):
    stream_file = tmp_path / "sim.txt"
    stream_file.write_text('{"msg_to_user": "simulated"}\n')
    cfg = OpenAiClientConfig(
        llm_model_name="newline", base_url=f"file:{stream_file}", api_key=None
    )

    final = await _last(
        stream_agent_response(History.model_validate([]), AgentResponse, cfg)
    )

    assert final is not None and final.msg_to_user == "simulated"
    assert final.usage is None


@pytest.mark.asyncio
async def test_parsed_usage_lands_when_schema_declares_the_field(monkeypatch):
    usage = SimpleNamespace(input_tokens=7, output_tokens=3)
    parsed = AgentResponse(msg_to_user="done")
    monkeypatch.setattr(
        llm_mod,
        "AsyncOpenAI",
        _fake_openai(parse_response=SimpleNamespace(output_parsed=parsed, usage=usage)),
    )

    out = await structured_agent_response(
        History.model_validate([]), AgentResponse, CFG
    )

    assert out.usage is not None
    assert out.usage.input_tokens == 7 and out.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_parsed_plain_schema_without_usage_field_is_untouched(monkeypatch):
    class Plain(BaseModel):
        answer: str

    usage = SimpleNamespace(input_tokens=7, output_tokens=3)
    parsed = Plain(answer="42")
    monkeypatch.setattr(
        llm_mod,
        "AsyncOpenAI",
        _fake_openai(parse_response=SimpleNamespace(output_parsed=parsed, usage=usage)),
    )

    out = await structured_agent_response(History.model_validate([]), Plain, CFG)

    assert out.answer == "42"
    assert not hasattr(out, "usage")
