import pytest
from unittest.mock import Mock, patch, AsyncMock

from pydantic import BaseModel

from clinicalagent.types import (
    TypedTool,
    AgentResponse,
    AgentResponseThoughtful,
)
from clinicalagent.utils import build_agent_response_schema, mcp_tools

class ExampleInput(BaseModel):
    arg: str

def test_build_agent_response_schema_no_tools_no_thought():
    cls = build_agent_response_schema(disable_thought=True, tools=[])
    # The returned class should be derived from AgentResponse
    assert issubclass(cls, AgentResponse)
    assert not issubclass(cls, AgentResponseThoughtful)

def test_build_agent_response_schema_no_tools_with_thought():
    cls = build_agent_response_schema(disable_thought=False, tools=[])
    assert issubclass(cls, AgentResponseThoughtful)

def test_build_agent_response_schema_with_tools():
    tool = TypedTool(name="ExampleTool", input_model=ExampleInput)
    cls = build_agent_response_schema(disable_thought=False, tools=[tool])
    assert issubclass(cls, AgentResponseThoughtful)

@pytest.mark.asyncio
async def test_mcp_tools():
    # Mock fastmcp and fastmcp.Client
    mock_fastmcp = Mock()
    mock_client = AsyncMock()
    mock_fastmcp.Client = mock_client.__class__
    
    mock_client.__aenter__.return_value = mock_client
    
    # Mock return of list_tools
    mock_tool = Mock()
    mock_tool.model_dump.return_value = {
        "name": "mock_mcp_tool",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}
    }
    mock_client.list_tools.return_value = [mock_tool]

    # Patch import since it might fail if fastmcp is not installed
    with patch.dict('sys.modules', {'fastmcp': mock_fastmcp}):
        tools_factory = mcp_tools(mock_client)
        assert callable(tools_factory)
        
        tools = await tools_factory()
        assert len(tools) == 1
        assert tools[0].name == "mock_mcp_tool"

def test_mcp_tools_import_error():
    with patch.dict('sys.modules', {'fastmcp': None}):
        # Mocking import correctly to raise ImportError
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'fastmcp':
                raise ImportError("No module named 'fastmcp'")
            return real_import(name, *args, **kwargs)
            
        with patch('builtins.__import__', mock_import):
            with pytest.raises(ImportError, match="fastmcp is not installed."):
                mcp_tools(Mock())
