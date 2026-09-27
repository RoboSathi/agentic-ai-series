"""Undoing a model's typesetting before anything reads its text.

Some models set their output like a printed page: a narrow no-break space
(U+202F) between a label and its number — "order 10898", "step 1" — and
non-breaking hyphens (U+2011) inside dates. On screen these are ordinary
spaces and hyphens. To everything that reads the text they are not: the
NLI checker scored "order<U+202F>10898 is in<U+202F>pending_review" as
contradicting the very record it restated, the id and amount matchers
look for ASCII, and a no-break space between JSON tokens is invalid JSON.

So the text is normalised once, where it enters — the adapter — and every
consumer (guardrails, the customer, memory, logs) sees the same plain
text. Which characters are rewritten is configuration, not code.
"""

import unicodedata

from agentic_commons.config.settings import OutputNormalisationSettings


class TypographyNormaliser:
    """Maps a model's decorative characters back to plain text."""

    def __init__(self, settings: OutputNormalisationSettings) -> None:
        self._enabled = settings.enabled
        self._space_categories = frozenset(settings.space_categories)
        self._table = {ord(char): None for char in settings.remove} | {
            ord(char): replacement for char, replacement in settings.replacements.items()
        }

    def normalise(self, text: str) -> tuple[str, int]:
        """The text in plain characters, and how many characters were changed."""
        if not self._enabled or not text or text.isascii():
            return text, 0
        parts = [self._plain(char) for char in text]
        changed = sum(1 for char, part in zip(text, parts, strict=True) if part != char)
        return "".join(parts), changed

    def _plain(self, char: str) -> str:
        """One character as plain text: a space, its replacement, nothing, or itself."""
        if ord(char) in self._table:
            return self._table[ord(char)] or ""
        if char != " " and unicodedata.category(char) in self._space_categories:
            return " "
        return char
