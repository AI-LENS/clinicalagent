import pytest
from unittest.mock import AsyncMock, patch

from clinicalagent.types import History, Message, MessageFlag
from clinicalagent.history_utils import last_summary_index, create_summary_entry, ConvSummary

def test_last_summary_index_empty():
    history = History(())
    assert last_summary_index(history) == 0

def test_last_summary_index_with_summary():
    history = History((
        Message(role="user", content="Hi"),
        Message(role="system", content="A summary", flags=[MessageFlag.is_summary]),
        Message(role="user", content="Hello again")
    ))
    assert last_summary_index(history) == 1

def test_last_summary_index_multiple_summaries():
    history = History((
        Message(role="system", content="Sum1", flags=[MessageFlag.is_summary]),
        Message(role="user", content="msg"),
        Message(role="system", content="Sum2", flags=[MessageFlag.is_summary]),
        Message(role="user", content="Hello again")
    ))
    # Should return the index of the latest summary, which is 2
    assert last_summary_index(history) == 2

@pytest.mark.asyncio
async def test_create_summary_entry():
    history = History((
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
        Message(role="user", content="How are you?")
    ))
    
    mock_summary = ConvSummary(summary="User greeted assistant")
    
    with patch("clinicalagent.history_utils.structured_agent_response", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = mock_summary
        
        # We reduce by 2 messages
        new_history = await create_summary_entry(history, reduce_by=2)
        
        # Original should not be modified
        assert len(history.root) == 3
        
        # New history should have the summary inserted at index 2 (reduce_by 2 + last_summary_index 0)
        assert len(new_history.root) == 4
        assert new_history.root[2].role == "system"
        assert MessageFlag.is_summary in new_history.root[2].flags
        assert "User greeted assistant" in new_history.root[2].content
        
        # Verify the prompt sent to the LLM
        mock_agent.assert_called_once()
        args, kwargs = mock_agent.call_args
        truncation_hist = kwargs['history']
        
        # The prompt should contain truncated text of the first 2 msgs
        prompt_content = truncation_hist.root[0].content
        assert "user: Hi" in prompt_content
        assert "assistant: Hello" in prompt_content
        assert "How are you?" not in prompt_content
