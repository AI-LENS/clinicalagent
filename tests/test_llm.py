import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch
from io import StringIO
import json

from clinicalagent.llm import StructuredStreamParser, stream_agent_response, simulated_agent_stream, structured_agent_response
from clinicalagent.types import History, Message, OpenAiClientConfig

class MockSchema(BaseModel):
    name: str

def test_structured_stream_parser():
    parser = StructuredStreamParser(MockSchema)
    
    # Incomplete json shouldn't generate output
    result = parser.feed('{"name"')
    assert result is None
    
    # Complete json should generate parsed model
    result2 = parser.feed(':"test"}')
    assert isinstance(result2, MockSchema)
    assert result2.name == "test"

@pytest.mark.asyncio
async def test_simulated_agent_stream_newline(tmp_path):
    history = History((Message(role="user", content="Hi"),))
    
    mock_file = tmp_path / "stream.txt"
    mock_file.write_text('{"name": "a"}\n{"name": "b"}\n<---xxx--->\n{"name": "c"}')
    
    stream_gen = simulated_agent_stream(mock_file, history, newline_delimited=True)
    
    lines = [chunk async for chunk in stream_gen]
    
    # No assistant messages in history, so we read chunk index 0 -> {"name": "a"}\n{"name": "b"}\n
    assert len(lines) == 2
    assert "a" in lines[0]
    assert "b" in lines[1]

@pytest.mark.asyncio
async def test_structured_agent_response():
    client_config = OpenAiClientConfig(
        llm_model_name="test_model",
        base_url="http://test",
        api_key="test"
    )
    history = History(())
    
    # Mock AsyncOpenAI client
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.output_parsed = MockSchema(name="parsed_test")
    mock_client.responses.parse.return_value = mock_response
    
    with patch("clinicalagent.llm.AsyncOpenAI", return_value=mock_client):
        res = await structured_agent_response(history, MockSchema, client_config)
        assert res.name == "parsed_test"
        mock_client.responses.parse.assert_called_once()
