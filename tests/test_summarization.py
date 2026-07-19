import pytest

import clinicalagent.history_utils as history_utils
from clinicalagent.agent import Agent
from clinicalagent.environment import DefaultEnvironment
from clinicalagent.history_utils import ConvSummary, create_summary_entry
from clinicalagent.types import AgentResponse, History, Message, MessageFlag


def make_history(*msgs: Message) -> History:
    return History.model_validate(list(msgs))


def capture_summarizer(monkeypatch, summary: str = "a sufficiently long summary of the prior conversation"):
    """Replace the LLM call; capture the prompt it was given."""
    prompts: list[str] = []

    async def fake(history, schema, client_config=None):
        prompts.append(history.root[0].content)
        return ConvSummary(summary=summary)

    monkeypatch.setattr(history_utils, "structured_agent_response", fake)
    return prompts


@pytest.mark.asyncio
async def test_second_summary_covers_messages_after_previous_summary(monkeypatch):
    prompts = capture_summarizer(monkeypatch)
    history = make_history(
        Message(role="user", content="original request"),
        Message(role="assistant", content="ALPHA"),
        Message(role="user", content="BETA"),
        Message(role="system", content="Summary: old summary text", flags=[MessageFlag.is_summary]),
        Message(role="user", content="GAMMA"),
        Message(role="assistant", content="DELTA"),
        Message(role="user", content="EPSILON"),
        Message(role="assistant", content="ZETA"),
    )

    new = await create_summary_entry(history, reduce_by=3)

    prompt = prompts[0]
    # The slice being retired is [old summary, GAMMA, DELTA] — not messages 0-2.
    assert "GAMMA" in prompt and "DELTA" in prompt
    assert "old summary text" in prompt  # previous summary chains forward
    assert "ALPHA" not in prompt  # already covered by the previous summary
    # New marker sits right after the retired slice; nothing is dropped unsummarized.
    summary_indices = [
        i for i, m in enumerate(new.root) if MessageFlag.is_summary in m.flags
    ]
    assert summary_indices[-1] == 6
    assert [m.content for m in new.root[7:]] == ["EPSILON", "ZETA"]


@pytest.mark.asyncio
async def test_summary_entry_preserves_original_user_request(monkeypatch):
    capture_summarizer(monkeypatch)
    history = make_history(
        Message(role="user", content="find phase 2 oncology trials"),
        Message(role="assistant", content="working"),
        Message(role="user", content="ok"),
        Message(role="assistant", content="done"),
    )

    new = await create_summary_entry(history, reduce_by=2)

    summary_msg = next(m for m in new.root if MessageFlag.is_summary in m.flags)
    assert "find phase 2 oncology trials" in summary_msg.content


@pytest.mark.asyncio
async def test_degenerate_summary_keeps_history_unchanged(monkeypatch):
    capture_summarizer(monkeypatch, summary="   ")
    history = make_history(
        Message(role="user", content="request"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="more"),
    )

    new = await create_summary_entry(history, reduce_by=2)

    assert not any(MessageFlag.is_summary in m.flags for m in new.root)
    assert [m.content for m in new.root] == [m.content for m in history.root]


def test_estimate_tokens_is_chars_over_four():
    history = make_history(Message(role="user", content="x" * 400))
    assert history_utils.estimate_tokens(history.root) == 100


@pytest.mark.asyncio
async def test_token_budget_triggers_summarization(monkeypatch):
    import clinicalagent.environment as environment_mod

    called = []

    async def fake_summary(old_history, reduce_by, client_config=None):
        called.append(reduce_by)
        new = old_history.model_copy(deep=True)
        new.add_message(
            role="system", content="Summary: s", flags=[MessageFlag.is_summary], index=reduce_by
        )
        return new

    monkeypatch.setattr(environment_mod, "create_summary_entry", fake_summary)

    env = DefaultEnvironment(
        tools=[],
        max_history_length=None,
        reduce_history_by=1,
        max_context_tokens=10,
    )
    env.history = make_history(
        Message(role="user", content="x" * 400),
        Message(role="assistant", content="y" * 400),
    )
    await env.get_context(remaining_iterations=5)
    assert called


@pytest.mark.asyncio
async def test_stream_raises_on_non_continuation_content():
    env = DefaultEnvironment(tools=[])
    agent = Agent(environment=env)

    async def fake_run():
        yield AgentResponse(msg_to_user="hello")
        yield AgentResponse(msg_to_user="different text")

    agent.run = fake_run  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        async for _ in agent.stream():
            pass
