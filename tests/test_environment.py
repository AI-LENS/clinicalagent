import pytest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from clinicalagent.types import History, Message, TypedTool, AgentResponse, CallToolRequestParams, TERMINATE
from clinicalagent.environment import DefaultEnvironment

class MyInput(BaseModel):
    arg: str

def test_default_env_init():
    tools = []
    env = DefaultEnvironment(tools=tools)
    assert env.provided_tools == tools
    assert len(env.history.root) == 0

@pytest.mark.asyncio
async def test_get_context_no_summary():
    tool = TypedTool(
        name="test_tool",
        input_model=MyInput,
        description="A test tool"
    )
    env = DefaultEnvironment(tools=[tool], max_history_length=10)
    # Add some history
    env.history.add_message(Message(role="user", content="hello"))
    
    with patch("clinicalagent.environment.create_summary_entry") as mock_summary:
        ctx = await env.get_context(remaining_iterations=5)
        # Should not have called summary because history length 1 < 10
        mock_summary.assert_not_called()
        
        # Context should have system prompt prepended
        assert len(ctx.root) == 2
        assert ctx.root[0].role == "system"
        assert "test_tool" in ctx.root[0].content
        assert "A test tool" in ctx.root[0].content

@pytest.mark.asyncio
async def test_get_context_with_summary():
    tool = TypedTool(name="test_tool", input_model=MyInput)
    env = DefaultEnvironment(tools=[tool], max_history_length=2, reduce_history_by=1)
    
    env.history.add_message(Message(role="user", content="m1"))
    env.history.add_message(Message(role="assistant", content="m2"))
    env.history.add_message(Message(role="user", content="m3"))
    env.history.add_message(Message(role="assistant", content="m4"))
    
    # We should trigger a summary here.
    with patch("clinicalagent.environment.create_summary_entry", new_callable=AsyncMock) as mock_summary:
        # Mock the resulted history from summary
        summary_hist = History((
            Message(role="system", content="A summary", flags=["is_summary"]),
            Message(role="user", content="m3"),
            Message(role="assistant", content="m4"),
        ))
        mock_summary.return_value = summary_hist
        
        ctx = await env.get_context(remaining_iterations=5)
        mock_summary.assert_called_once()
        
        assert len(ctx.root) == 4 # system prompt + summary result
        assert ctx.root[0].role == "system" # the dynamically created system prompt
        assert ctx.root[1].content == "A summary"
        assert ctx.root[2].content == "m3"

@pytest.mark.asyncio
async def test_on_agent_message_completed_no_action():
    env = DefaultEnvironment(tools=[])
    resp = AgentResponse[BaseModel](msg_to_user="hello")
    res = await env.on_agent_message_completed(resp)
    
    assert isinstance(res, Message)
    assert res.role == "user"
    assert "No tool was called" in res.content

@pytest.mark.asyncio
async def test_on_agent_message_completed_terminating_tool():
    terminating_tool = TypedTool(
        name="term_tool",
        input_model=MyInput,
        _meta={"clinicalagent": ["TERMINATING"]}
    )
    env = DefaultEnvironment(tools=[terminating_tool])
    
    action = CallToolRequestParams(tool_name="term_tool", arguments=MyInput(arg="done"))
    resp = AgentResponse[BaseModel](action=action)
    
    res = await env.on_agent_message_completed(resp)
    assert res is TERMINATE

@pytest.mark.asyncio
async def test_on_agent_message_completed_normal_tool():
    normal_tool = TypedTool(name="normal_tool", input_model=MyInput)
    env = DefaultEnvironment(tools=[normal_tool])
    
    # mock call_tool
    env.call_tool = AsyncMock(return_value="tool output")
    
    action = CallToolRequestParams(tool_name="normal_tool", arguments=MyInput(arg="test"))
    resp = AgentResponse[BaseModel](action=action)
    
    res = await env.on_agent_message_completed(resp)
    env.call_tool.assert_called_once_with(action)
    
    assert isinstance(res, Message)
    assert res.role == "user"
    assert "Tool result: tool output" in res.content

@pytest.mark.asyncio
async def test_call_tool_not_implemented():
    env = DefaultEnvironment(tools=[])
    action = CallToolRequestParams(tool_name="tool", arguments=MyInput(arg="test"))
    with pytest.raises(NotImplementedError):
        await env.call_tool(action)
