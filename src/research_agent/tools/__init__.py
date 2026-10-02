"""The research agent's tools: read-only, each returning data for the model.

Registered for the direct-answer loop (`graph/answer_nodes.py`). Every tool
returns the URLs its output was read from under `URLS_KEY`, so the reply can
cite them and the grounding check knows where each fact came from.
"""

# `ToolResult.data[URLS_KEY]`: the URLs the output was read from, for the reply's Sources line.
URLS_KEY = "urls"

from research_agent.tools.registry import NAMESPACE, build_answer_registry  # noqa: E402

__all__ = ["NAMESPACE", "URLS_KEY", "build_answer_registry"]
