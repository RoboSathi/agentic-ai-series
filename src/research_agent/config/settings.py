"""Configuration for the research agent.

These settings live with the agent, not with the platform: the section is
declared in this package's overlay and read here, so `agentic_commons`
never learns what a research run is.
"""

from pydantic_settings import SettingsConfigDict

from agentic_commons.config.settings import YamlBackedSettings


class ResearchAgentSettings(YamlBackedSettings):
    """The research agent's own wiring values."""

    _yaml_section = "research_agent"

    title: str
    placeholder: str
    greeting: str

    model_config = SettingsConfigDict(env_prefix="RESEARCH_AGENT__")


_SETTINGS: ResearchAgentSettings | None = None


def get_agent_settings() -> ResearchAgentSettings:
    """The agent's settings, built once per process."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = ResearchAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS


def reload_agent_settings() -> ResearchAgentSettings:
    """Rebuild from disk — used by tests that change the overlay."""
    global _SETTINGS
    _SETTINGS = ResearchAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS
