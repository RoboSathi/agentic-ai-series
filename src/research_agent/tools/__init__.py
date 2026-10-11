"""The research agent's tools: read-only, each returning data for the model.

Registered for the direct-answer loop (`graph/answer_nodes.py`). Every tool
returns the URLs its output was read from under `URLS_KEY`, so the reply can
cite them and the grounding check knows where each fact came from.
"""

# `ToolResult.data[URLS_KEY]`: the URLs the output was read from, for the reply's Sources line.
URLS_KEY = "urls"
# A search's own query and the dates its results give themselves, for the recency check.
QUERY_KEY = "query"
PUBLISHED_KEY = "published"
# A search's text per hit, in `URLS_KEY` order: each hit is evidence from its own page.
HITS_KEY = "hits"
# Set on a fetched page's result: that URL was read in full, not only as a search snippet.
PAGE_KEY = "page"
# A search's route, as the OBSERVATION heading names it ("ddgs", "tavily (ddgs failed)").
PROVIDER_KEY = "provider"

from research_agent.tools.registry import NAMESPACE, build_answer_registry  # noqa: E402

__all__ = [
    "HITS_KEY",
    "NAMESPACE",
    "PAGE_KEY",
    "PROVIDER_KEY",
    "PUBLISHED_KEY",
    "QUERY_KEY",
    "URLS_KEY",
    "build_answer_registry",
]
