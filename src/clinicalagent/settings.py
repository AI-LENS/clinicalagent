"""
Global configuration for the clinicalagent framework.

Settings are loaded from multiple sources in the following priority order (highest first):

1. **Constructor arguments** (``init_settings``) — programmatic overrides.
2. **YAML config file** (``clinicalagent-config.yaml``) — project-level configuration.
3. **``.env`` file** (``dotenv_settings``) — environment variable file in the project root.
4. **Environment variables** (``env_settings``) — system-level environment variables.
5. **File secrets** (``file_secret_settings``) — Docker/Kubernetes-style secret files.

To configure the LLM provider, create a ``clinicalagent-config.yaml`` in your project root:

.. code-block:: yaml

    llm_model_name: gpt-4o
    llm_base_url: https://api.openai.com/v1
    llm_api_key: sk-...

Or use environment variables with matching names (e.g., ``LLM_MODEL_NAME``).
"""

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    """
    Global settings for the clinicalagent framework.

    All fields have sensible defaults and can be overridden via YAML config,
    environment variables, or ``.env`` files (see module docstring for priority order).

    Attributes:
        llm_model_name: The LLM model identifier (e.g., ``"gpt-4o"``, ``"google/gemini-2.5-flash"``).
            Required for API calls; must be set in config or environment.
        llm_base_url: Base URL for the OpenAI-compatible API endpoint.
            Examples: ``"https://api.openai.com/v1"``, ``"https://openrouter.ai/api/v1"``.
            Required for API calls; must be set in config or environment.
        llm_api_key: API key for authentication. May be None for local/unauthenticated endpoints.
        llm_api_extra_kw: Extra keyword arguments passed as ``extra_body`` to OpenAI API calls.
            Useful for provider-specific parameters (e.g., ``{"temperature": 0.7}``).
        max_agent_iterations: Maximum number of agent loop iterations before raising
            ``MaxAgentIterationsExceededError``. Acts as a safety limit to prevent infinite loops.
        max_history_length: Maximum number of messages in the conversation history before
            automatic summarization is triggered. Set to None to disable summarization.
        reduce_history_by: When summarization is triggered, this many messages from the start
            of the history are summarized into a single summary message. Should be less than
            ``max_history_length``.
    """

    llm_model_name: str | None = None
    llm_base_url: str | None = None

    llm_api_key: str | None = None
    llm_api_extra_kw: dict = {}

    max_agent_iterations: int = 7
    max_history_length: int | None = 11
    reduce_history_by: int = 7

    model_config = SettingsConfigDict(
        yaml_file="clinicalagent-config.yaml",
        yaml_file_encoding="utf-8",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the priority order of configuration sources.

        The order returned here determines which source wins when the same setting
        is defined in multiple places. YAML config is prioritized over environment
        variables to allow project-level overrides.
        """
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )


settings = Settings()  # type: ignore
"""
Module-level singleton instance of ``Settings``.

This is loaded once at import time and used throughout the framework as the default
configuration. To override settings programmatically, pass ``OpenAiClientConfig``
directly to the ``Agent`` or ``DefaultEnvironment`` constructors.
"""
