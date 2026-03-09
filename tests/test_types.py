import pytest
from pydantic import ValidationError
from pydantic import BaseModel

from clinicalagent.types import (
    McpTool,
    Message,
    History,
    AgentResponse,
    AgentResponseThoughtful,
    TypedTool,
    BaseToolModel,
)

def test_mcp_tool_basic():
    tool = McpTool(
        name="test_tool",
        inputSchema={"type": "object"}
    )
    assert tool.name == "test_tool"
    assert tool.inputSchema == {"type": "object"}
    
def test_message_creation():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.flags == []

def test_history_initialization_assigns_ids():
    msg1 = Message(role="user", content="Hello")
    msg2 = Message(role="assistant", content="Hi there")
    
    history = History((msg1, msg2))
    assert history.root[0].id == 0
    assert history.root[1].id == 1

def test_history_add_message():
    history = History(())
    history.add_message(role="system", content="System instruction")
    assert len(history.root) == 1
    assert history.root[0].id == 0
    assert history.root[0].role == "system"

    msg = Message(role="user", content="test")
    history.add_message(msg)
    assert len(history.root) == 2
    assert history.root[1].id == 1
    assert history.root[1].content == "test"

    # Insert at specific index
    history.add_message(role="assistant", content="insert", index=1)
    assert len(history.root) == 3
    assert history.root[1].id == 1
    assert history.root[1].role == "assistant"
    # IDs should be re-assigned
    assert history.root[2].id == 2
    assert history.root[2].content == "test"

def test_history_add_message_out_of_bounds():
    history = History(())
    with pytest.raises(IndexError):
        history.add_message(role="user", content="test", index=5)

def test_history_compact():
    history = History((Message(role="user", content="Q1"), Message(role="assistant", content="A1")))
    compacted = history.compact()
    assert compacted == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"}
    ]

def test_agent_response():
    resp = AgentResponse[BaseModel](msg_to_user="Hello")
    assert resp.msg_to_user == "Hello"
    assert not resp.turn_completed

def test_agent_response_thoughtful():
    resp = AgentResponseThoughtful[BaseModel](thought="I should say hi", msg_to_user="Hi")
    assert resp.thought == "I should say hi"
    assert resp.msg_to_user == "Hi"

class MockInput(BaseModel):
    param1: str
    param2: int

def test_typed_tool_validate_schema_from_input_model():
    tool = TypedTool(
        name="test_tool",
        input_model=MockInput,
    )
    assert tool.name == "test_tool"
    assert tool.inputSchema is not None
    assert "param1" in tool.inputSchema["properties"]

def test_typed_tool_missing_schema_and_model():
    with pytest.raises(ValidationError):
        TypedTool(name="test_tool")

class CustomToolModel(BaseToolModel):
    """My custom tool"""
    my_field: str

class TerminateToolModel(BaseToolModel, terminating=True):
    """My terminate tool"""
    reason: str

def test_base_tool_model_convert_to_tool():
    tool = CustomToolModel.convert_to_tool()
    assert tool.name == "CustomToolModel"
    assert tool.description == "My custom tool"
    assert not tool.input_model.__is_terminating__
    assert tool.meta == {}

def test_base_tool_model_terminating():
    tool = TerminateToolModel.convert_to_tool()
    assert tool.name == "TerminateToolModel"
    assert tool.input_model.__is_terminating__
    assert tool.meta is not None
    assert "TERMINATING" in tool.meta.get("clinicalagent", [])
