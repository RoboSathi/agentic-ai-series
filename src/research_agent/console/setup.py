"""Start-up checks for the live posture: fail with a sentence, not a traceback.

A missing model key would otherwise surface minutes later as a failed
interview turn; a missing search key only narrows search to DuckDuckGo,
so it is a warning the greeting repeats, not a stop.
"""

from urllib.parse import urlparse

from agentic_commons.config.settings import AppConfig, ProviderName
from agentic_commons.resilience.rate_limiter import get_rate_limiter
from research_agent.config.settings import ResearchAgentSettings

_NIM: ProviderName = "nim"
_TAVILY = "tavily"


def missing_model_key(settings: AppConfig, agent: ResearchAgentSettings) -> str | None:
    """The fatal message when routed roles need an NVIDIA key that is not set."""
    if agent.model_backend != "routed" or settings.llm.api_key_for(_NIM):
        return None
    roles = sorted(role for role, spec in settings.llm.roles.items() if spec.provider == _NIM)
    return agent.setup.missing_model_key.format(roles=", ".join(roles)) if roles else None


def search_warning(settings: AppConfig, agent: ResearchAgentSettings) -> str | None:
    """The warning when live search has no Tavily key (DuckDuckGo only)."""
    web = settings.web
    uses_tavily = _TAVILY in web.search.providers + web.search.time_bounded_providers
    if web.mode != "live" or not uses_tavily or web.tavily.api_key:
        return None
    return agent.setup.no_search_key


def search_usage(settings: AppConfig, agent: ResearchAgentSettings) -> str | None:
    """ "Tavily this month: N / quota" while searching live; None otherwise."""
    limiter = get_rate_limiter()
    host = urlparse(settings.web.tavily.base_url).hostname or ""
    bucket = limiter.bucket_for_host(host) if limiter is not None else None
    if settings.web.mode != "live" or limiter is None or bucket is None:
        return None
    quota = limiter.quota_for(bucket)
    if quota is None:
        return None
    return agent.setup.search_usage.format(used=limiter.quota_used(bucket), quota=quota)
