"""
History summarization utilities for managing conversation length.

When the conversation history exceeds ``max_history_length``, these utilities compress
older messages into a single LLM-generated summary. This keeps the context window
manageable while preserving important information from earlier turns.

The summarization strategy:
    1. ``last_summary_index()`` finds the position of the most recent summary message.
    2. ``create_summary_entry()`` takes the first ``reduce_by`` messages from the history,
       asks the LLM to summarize them, and inserts the summary as a new message with
       the ``is_summary`` flag.
    3. The ``DefaultEnvironment.get_context()`` method then slices the history from the
       last summary onward, effectively "forgetting" the raw messages that were summarized.

Note: The original messages are NOT deleted — the summary is inserted alongside them.
    The slicing happens at context-building time, not at summarization time.
"""

from pydantic import BaseModel

from .llm import structured_agent_response
from .types import History, Message, MessageFlag, OpenAiClientConfig


class ConvSummary(BaseModel):
    """Schema for the LLM-generated conversation summary. Used as the structured output format."""

    summary: str


def last_summary_index(history: History) -> int:
    """
    Find the index of the most recent summary message in the history.

    Scans the history in reverse to find the last message flagged with ``is_summary``.
    This index is used by ``DefaultEnvironment.get_context()`` to slice the history —
    only messages from this index onward are included in the LLM context.

    Args:
        history: The full conversation history.

    Returns:
        The index of the last summary message, or 0 if no summary exists
        (meaning the entire history should be used).
    """
    return next(
        (
            idx
            for idx, message in reversed(list(enumerate(history.root)))
            if MessageFlag.is_summary in message.flags
        ),
        0,
    )


async def create_summary_entry(
    old_history: History,
    reduce_by: int,
    client_config: OpenAiClientConfig | None = None,
) -> History:
    """
    Summarize the first ``reduce_by`` messages of the history using the LLM.

    Creates a deep copy of the history, extracts the first ``reduce_by`` messages,
    sends them to the LLM for summarization, and inserts the resulting summary
    as a new message at position ``reduce_by + last_summary_index``.

    Args:
        old_history: The current conversation history (not modified — a deep copy is made).
        reduce_by: Number of messages from the start to include in the summary.
        client_config: Optional LLM client config for the summarization call.
            Falls back to global settings if None.

    Returns:
        A new History object with the summary message inserted.
    """
    history = old_history.model_copy(deep=True)

    truncated_conv = "\n".join(
        f"{msg.role}: {msg.content}" for msg in history.root[:reduce_by]
    )

    prompt = f"Please summarize the following conversation between the user and the assistant in a concise manner, retaining all important details:\n\n{truncated_conv}"

    truncation_hist = History.model_validate([Message(role="user", content=prompt)])
    summary_response = await structured_agent_response(
        history=truncation_hist,
        schema=ConvSummary,
        client_config=client_config,
    )
    history.add_message(
        role="system",
        content=f"Summary of previous conversation: {summary_response.summary}",
        flags=[MessageFlag.is_summary],
        index=reduce_by + last_summary_index(history),
    )
    return history
