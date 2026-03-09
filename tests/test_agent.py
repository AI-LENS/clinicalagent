import pytest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from clinicalagent.types import History, AgentResponse, Message, TERMINATE, CallToolRequestParams, ToolCallStream, TextStream, BaseToolModel
from clinicalagent.environment import DefaultEnvironment
from clinicalagent.agent import Agent, MaxAgentIterationsExceededError

class MyInput(BaseToolModel):
    """A test tool input"""
    arg: str

class TermInput(BaseToolModel, terminating=True):
    """A terminating tool input"""
    arg: str

@pytest.fixture
def mock_env():
    env = DefaultEnvironment([])
    env.get_context = AsyncMock(return_value=History(()))
    env.get_tools = AsyncMock(return_value=[])
    env.on_agent_message_completed = AsyncMock(return_value=TERMINATE)
    return env

@pytest.mark.asyncio
async def test_agent_run_terminates_no_tool(mock_env):
    agent = Agent(environment=mock_env, max_iterations=2, require_terminating_tool_call=False)
    
    mock_resp_1 = AgentResponse[BaseModel](msg_to_user="Hi", turn_completed=False)
    mock_resp_2 = AgentResponse[BaseModel](msg_to_user="Hi guys", turn_completed=False)
    
    async def mock_stream(*args, **kwargs):
        yield mock_resp_1
        yield mock_resp_2
        
    with patch("clinicalagent.agent.stream_agent_response", new=mock_stream):
        # We start the loop
        responses = [r async for r in agent.run()]
        
        # We expect 3 responses yielded: 2 from stream + 1 when it marks it completed
        assert len(responses) == 3
        
        # It should terminate because require_terminating_tool_call is False 
        # and response.action is None
        assert mock_env.on_agent_message_completed.call_count == 0

@pytest.mark.asyncio
async def test_agent_run_terminating_tool(mock_env):
    tool = TermInput.convert_to_tool()
    mock_env.get_tools = AsyncMock(return_value=[tool])
    agent = Agent(environment=mock_env, max_iterations=3, require_terminating_tool_call=True)
    
    mock_resp = AgentResponse[TermInput](action=CallToolRequestParams(tool_name="TermInput", arguments=TermInput(arg="val")))
    
    async def mock_stream(*args, **kwargs):
        yield mock_resp
        
    with patch("clinicalagent.agent.stream_agent_response", new=mock_stream):
        responses = [r async for r in agent.run()]
        # 2 yields: one from stream, one after it's completed
        assert len(responses) == 2
        mock_env.on_agent_message_completed.assert_called_once()
        # Because we return TERMINATE from the mock, it doesn't loop

@pytest.mark.asyncio
async def test_agent_run_max_iterations(mock_env):
    tool = TermInput.convert_to_tool()
    mock_env.get_tools = AsyncMock(return_value=[tool])
    # The environment returns a normal message to continue the loop
    mock_env.on_agent_message_completed = AsyncMock(return_value=Message(role="user", content="Continue"))
    
    agent = Agent(environment=mock_env, max_iterations=2, require_terminating_tool_call=True)
    
    mock_resp = AgentResponse[TermInput](action=CallToolRequestParams(tool_name="TermInput", arguments=TermInput(arg="val")))
    
    async def mock_stream(*args, **kwargs):
        yield mock_resp
        
    with patch("clinicalagent.agent.stream_agent_response", new=mock_stream):
        with pytest.raises(MaxAgentIterationsExceededError):
            async for r in agent.run():
                pass

@pytest.mark.asyncio
async def test_agent_stream(mock_env):
    agent = Agent(environment=mock_env, max_iterations=2, require_terminating_tool_call=False)
    
    mock_resp_1 = AgentResponse[BaseModel](msg_to_user="Hi", turn_completed=False)
    mock_resp_2 = AgentResponse[BaseModel](msg_to_user="Hi guys", turn_completed=False)
    
    async def mock_run():
        yield mock_resp_1
        yield mock_resp_2
        # Final response from agent run loop where turn_completed=True
        mock_resp_final = AgentResponse[BaseModel](msg_to_user="Hi guys", action=CallToolRequestParams(tool_name="x", arguments=MyInput(arg="y")), turn_completed=True)
        yield mock_resp_final
        
    with patch.object(agent, "run", side_effect=mock_run):
        streams = [s async for s in agent.stream()]
        
        # We expect TextStream("Hi"), TextStream(" guys"), then ToolCallStream
        assert len(streams) == 3
        assert isinstance(streams[0], TextStream) and streams[0].delta == "Hi"
        assert isinstance(streams[1], TextStream) and streams[1].delta == " guys"
        assert isinstance(streams[2], ToolCallStream) and streams[2].tool_call.tool_name == "x"
