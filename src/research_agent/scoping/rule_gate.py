"""The scope gate's deterministic layer: a message that plainly names a kind of research.

Cheaper and more predictable than any model: when a message carries several cues of one
research kind ("compare", "options", "which should we choose" → a decision), the answer is a
lookup in the configured cue lists. It decides only when that is clear — one kind leads
and enough of its cues appear — and otherwise says nothing, so the encoder and then the LLM
get the cases that need judgement. It never decides Q&A: a quick question is exactly the case
where cue words are least informative.
"""

import re

from agent_platform.logging import instrument
from research_agent.config.settings import ScopingSettings
from research_agent.domain.scope import RESEARCH_ARCHETYPES, Archetype, GateVerdict


@instrument("research_agent")
class RuleGate:
    """Cue lists in, a research verdict out — or None when the cues are not decisive."""

    def __init__(self, settings: ScopingSettings) -> None:
        self._min_hits = settings.rule_gate_min_cue_hits
        self._cues: dict[Archetype, list[tuple[str, re.Pattern[str]]]] = {
            name: [(text.strip(), _cue(text)) for text in settings.archetype_keywords.get(name, [])]
            for name in RESEARCH_ARCHETYPES
        }

    def verdict(self, message: str) -> GateVerdict | None:
        hits = {name: self._hits(name, message) for name in self._cues}
        ranked = sorted(hits.items(), key=lambda item: len(item[1]), reverse=True)
        (best, found), runner_up = ranked[0], len(ranked[1][1])
        if len(found) < self._min_hits or len(found) == runner_up:
            return None
        return GateVerdict(
            decision="scopable",
            mode="research",
            archetype=best,
            topic=message.strip(),
            reason=", ".join(found),
            source="rule",
        )

    def _hits(self, name: Archetype, message: str) -> list[str]:
        return [text for text, pattern in self._cues[name] if pattern.search(message)]


def _cue(text: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(text.strip())}\b", re.IGNORECASE)
