"""Requests this agent will not help with, refused by pattern before any model reads them.

The same `scoping.refuse_patterns` the research-kind step applies to a `/research` question, here
for a plain question: no model is asked whether to help build a weapon, on either path.
"""

import re

from agent_platform.logging import instrument


@instrument("research_agent")
class RefusalRule:
    """Configured patterns in, a yes/no out."""

    def __init__(self, patterns: list[str]) -> None:
        self._patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def refused(self, message: str) -> bool:
        return any(pattern.search(message) for pattern in self._patterns)


__all__ = ["RefusalRule"]
