"""Does this claim talk about that subject, or that criterion?

Deterministic term matching, used wherever the agent must decide coverage
without asking a model: stop rules, gap analysis, the audit's coverage
checks. Terms are compared by a configured stem prefix ("cycles" and
"cycle life" share "cycle"), stop words are ignored, and a phrase's terms
are widened by a configured synonym map (a claim about "minus 20 degrees"
is a claim about cold-weather performance). A phrase matches a text when
any of its terms appears. Simple on purpose — every decision it makes can
be reproduced by hand.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol


class MatchingRules(Protocol):
    """How terms are compared (configured by the agent, `matching:`)."""

    @property
    def token_pattern(self) -> str: ...

    @property
    def stem_chars(self) -> int: ...

    @property
    def min_token_chars(self) -> int: ...

    @property
    def stopwords(self) -> Sequence[str]: ...

    @property
    def synonyms(self) -> Mapping[str, Sequence[str]]: ...


class TermMatcher:
    """Stem-prefix term matching with stop words and synonyms."""

    def __init__(self, settings: MatchingRules) -> None:
        self._token = re.compile(settings.token_pattern, re.IGNORECASE)
        self._stem = settings.stem_chars
        self._min = settings.min_token_chars
        self._stop = {word.casefold() for word in settings.stopwords}
        self._synonyms = {
            self._cut(key): {self._cut(word) for word in words}
            for key, words in settings.synonyms.items()
        }

    def terms(self, text: str) -> set[str]:
        """The distinctive stems of a text."""
        return {
            self._cut(token)
            for token in self._token.findall(text)
            if len(token) >= self._min and token.casefold() not in self._stop
        }

    def phrase_terms(self, phrase: str) -> set[str]:
        """A phrase's stems plus their configured synonyms."""
        stems = self.terms(phrase)
        return stems.union(*(self._synonyms.get(stem, set()) for stem in stems))

    def mentions(self, text: str, phrase: str) -> bool:
        """Whether any term of `phrase` (or a synonym) appears in `text`."""
        wanted = self.phrase_terms(phrase)
        return bool(wanted) and bool(wanted & self.terms(text))

    def mentions_all(self, text: str, phrases: Iterable[str]) -> bool:
        return all(self.mentions(text, phrase) for phrase in phrases)

    def _cut(self, token: str) -> str:
        return token.casefold()[: self._stem]
