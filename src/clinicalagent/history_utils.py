import logging
from typing import Iterable

from pydantic import BaseModel

from .llm import structured_agent_response
from .types import History, Message, MessageFlag, OpenAiClientConfig

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = (
    "Summarize the conversation below. The summary will be handed to another "
    "AI assistant that must continue the task; it will see only this summary "
    "and the messages that come after it, not the original conversation. "
    "Preserve everything needed to continue seamlessly: the user's requests, "
    "decisions already made, key facts and identifiers (trial IDs, eligibility "
    "criteria, patient details), tool results relied upon, errors encountered "
    "and how they were resolved, and what remains to be done.\n\n"
    "Conversation:\n{conversation}"
)

# Below this the summary is considered degenerate (empty/echo) and rejected.
MIN_SUMMARY_CHARS = 20


class ConvSummary(BaseModel):
    summary: str


def last_summary_index(history: History) -> int:
    return next(
        (
            idx
            for idx, message in reversed(list(enumerate(history.root)))
            if MessageFlag.is_summary in message.flags
        ),
        0,
    )


def estimate_tokens(messages: Iterable[Message]) -> int:
    """Rough token estimate: total characters / 4."""
    return sum(len(msg.content) for msg in messages) // 4


async def create_summary_entry(
    old_history: History,
    reduce_by: int,
    client_config: OpenAiClientConfig | None = None,
) -> History:
    """Summarize the `reduce_by` messages after the last summary and insert a
    new summary marker after them. On a degenerate summary (after one retry)
    the history is returned unchanged rather than losing information."""
    history = old_history.model_copy(deep=True)
    start = last_summary_index(history)

    # The slice being retired — starts at the previous summary (if any) so
    # earlier information chains forward into the new summary.
    retired = history.root[start : start + reduce_by]
    truncated_conv = "\n".join(f"{msg.role}: {msg.content}" for msg in retired)

    truncation_hist = History.model_validate(
        [Message(role="user", content=SUMMARIZATION_PROMPT.format(conversation=truncated_conv))]
    )
    for _ in range(2):
        summary_response = await structured_agent_response(
            history=truncation_hist,
            schema=ConvSummary,
            client_config=client_config,
        )
        if len(summary_response.summary.strip()) >= MIN_SUMMARY_CHARS:
            break
    else:
        logger.warning(
            "Summarizer returned a degenerate summary twice; keeping history unchanged."
        )
        return history

    content = f"Summary of previous conversation: {summary_response.summary}"
    # Keep the original user request verbatim — it must never be lost to compaction.
    original_request = next(
        (msg.content for msg in history.root if msg.role == "user" and not msg.flags),
        None,
    )
    if original_request:
        content = f"Original user request: {original_request}\n\n{content}"

    history.add_message(
        role="system",
        content=content,
        flags=[MessageFlag.is_summary],
        index=start + reduce_by,
    )
    return history
