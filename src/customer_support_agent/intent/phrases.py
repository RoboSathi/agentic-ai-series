"""Whether a message asks for something, or says it does not want it.

A phrase rule used to match a phrase anywhere: "refund" made "I don't want a refund"
a refund request, and "I didn't get my refund" one too. A phrase now counts only when
it is *asserted* — no negation in the few words before it. One that appears only
negated is *refused*: the customer said they do not want it, and no classifier may
grant it back. The intent router and the deterministic checks read phrases the same
way, through this one reader, so the two never disagree about what was asked.

The negation words and how far back to look are configuration
(`intent_router.negation`).
"""

import re

from customer_support_agent.config import get_agent_settings
from customer_support_agent.config.settings import PhraseNegationSettings

# Typographic apostrophes a customer's keyboard may send ("didn’t").
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})
_WORD = re.compile(r"[\w']+")
# Where one clause ends and the next begins.
_CLAUSE_END = re.compile(r"[,;.!?]")


def normalised(text: str) -> str:
    """Case-folded, with typographic apostrophes made plain, for phrase matching."""
    return text.translate(_APOSTROPHES).casefold()


class PhraseReader:
    """Reads whether each phrase in a message is asserted or refused."""

    def __init__(self, settings: PhraseNegationSettings | None = None) -> None:
        negation = settings or get_agent_settings().intent_router.negation
        self._window = negation.window_words
        self._breaks = [
            re.compile(rf"\b{re.escape(normalised(word))}\b") for word in negation.clause_breaks
        ]
        self._markers = [
            re.compile(rf"\b{re.escape(normalised(marker))}\b") for marker in negation.markers
        ]

    def asserted(self, text: str, phrase: str) -> bool:
        """Whether the phrase appears at least once with no negation just before it."""
        lowered = normalised(text)
        return any(not self._negated_at(lowered, start) for start in _starts(lowered, phrase))

    def refused(self, text: str, phrase: str) -> bool:
        """Whether the phrase appears, and only ever negated ("I don't want a refund")."""
        lowered = normalised(text)
        starts = _starts(lowered, phrase)
        return bool(starts) and all(self._negated_at(lowered, start) for start in starts)

    def negated_at(self, text: str, start: int) -> bool:
        """Whether the words just before `start` in `text` refuse what begins there."""
        return self._negated_at(normalised(text), start)

    def _negated_at(self, lowered: str, start: int) -> bool:
        """A negation counts only inside the clause: "do not refund A, but refund B" asks for B."""
        clause = _CLAUSE_END.split(lowered[:start])[-1]
        for contrast in self._breaks:
            clause = contrast.split(clause)[-1]
        preceding = " ".join(_WORD.findall(clause)[-self._window :])
        return any(marker.search(preceding) for marker in self._markers)


def _starts(lowered: str, phrase: str) -> list[int]:
    candidate = normalised(phrase.strip())
    if not candidate:
        return []
    return [m.start() for m in re.finditer(rf"\b{re.escape(candidate)}\b", lowered)]
