"""Start-up checks for the live posture: fail with a sentence, not a traceback.

A missing model key would otherwise surface minutes later as a failed
interview turn; a missing search key only narrows search to DuckDuckGo,
so it is a warning the greeting repeats, not a stop.
"""

from agent_platform.config.settings import AppConfig
from research_agent.config.settings import ResearchAgentSettings

_TAVILY = "tavily"


def missing_model_key(settings: AppConfig, agent: ResearchAgentSettings) -> str | None:
    """The fatal message when routed roles use a provider whose key is not set."""
    llm = settings.llm
    keyless = {
        role: llm.provider_settings(spec.provider).key_env_vars
        for role, spec in llm.roles.items()
        if llm.provider_settings(spec.provider).key_env_vars and not llm.api_key_for(spec.provider)
    }
    if not keyless:
        return None
    envs = sorted({env for names in keyless.values() for env in names})
    return agent.setup.missing_model_key.format(
        roles=", ".join(sorted(keyless)), env=" or ".join(envs)
    )


def search_warning(settings: AppConfig, agent: ResearchAgentSettings) -> str | None:
    """The warning when live search has no Tavily key (DuckDuckGo only)."""
    web = settings.web
    uses_tavily = _TAVILY in web.search.providers + web.search.time_bounded_providers
    if web.mode != "live" or not uses_tavily or web.tavily.api_key:
        return None
    return agent.setup.no_search_key
