import pytest
import os
from pydantic import ValidationError

def test_settings_load_env(monkeypatch):
    monkeypatch.setenv("llm_model_name", "env_test_model")
    monkeypatch.setenv("llm_base_url", "http://env-test")
    monkeypatch.setenv("max_agent_iterations", "10")
    monkeypatch.setenv("reduce_history_by", "5")
    
    from clinicalagent.settings import Settings
    
    # Instantiate to read env vars
    test_settings = Settings()
    
    assert test_settings.llm_model_name == "env_test_model"
    assert test_settings.llm_base_url == "http://env-test"
    assert test_settings.max_agent_iterations == 10
    assert test_settings.reduce_history_by == 5

def test_settings_default_values(monkeypatch):
    # Ensure no environment variables interfere
    monkeypatch.delenv("llm_model_name", raising=False)
    monkeypatch.delenv("llm_base_url", raising=False)
    
    from clinicalagent.settings import Settings
    
    # Create with explicit None to override any file configs during test or test defaults
    # Since config files might exist on the system, we test the class behavior
    test_settings = Settings(llm_model_name=None, llm_base_url=None)
    
    assert test_settings.llm_model_name is None
    assert test_settings.llm_base_url is None
    assert test_settings.max_agent_iterations == 7
    assert test_settings.max_history_length == 11
    assert test_settings.reduce_history_by == 7
    assert test_settings.llm_api_extra_kw == {}
