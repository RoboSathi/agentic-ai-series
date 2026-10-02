"""Quote matching that tolerates typography, never content.

The ledger's rule is that a span must be an exact substring of the fetched page. A model that
copies a sentence faithfully can still fail it: it types a straight quote where the page has a
curly one, an en dash where the page has an em dash, one space where the page has a newline or a
non-breaking space. None of that changes what the page says.

`canonical_span` finds the span in the page after folding exactly those differences (quote and
dash styles, no-break spaces, zero-width characters, runs of whitespace) and returns the page's
OWN text for it. The ledger then stores, and still strictly verifies, a true substring — the
invariant is unchanged; only the model's keystrokes are forgiven. Case, words and order are never
folded, so a paraphrase or a changed number is still rejected.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
}  # fmt: skip
_SPACES = frozenset("\u00a0\u2002\u2003\u2009\u200a\u202f\u3000")
_INVISIBLE = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")


def _fold(text: str) -> tuple[str, list[int]]:
    """The folded text, and for each folded character the index it came from in `text`."""
    out: list[str] = []
    origin: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char in _INVISIBLE:
            continue
        if char.isspace() or char in _SPACES:
            if not previous_space:
                out.append(" ")
                origin.append(index)
            previous_space = True
            continue
        previous_space = False
        for piece in _FOLD.get(char, char):
            out.append(piece)
            origin.append(index)
    return "".join(out), origin


def canonical_span(page_text: str, span: str) -> str | None:
    """The page's own text for `span`, or None if the page does not contain it."""
    if span and span in page_text:
        return span
    folded_text, origin = _fold(page_text)
    folded_span, _ = _fold(span)
    folded_span = folded_span.strip()
    if not folded_span:
        return None
    start = folded_text.find(folded_span)
    if start < 0:
        return None
    end = start + len(folded_span) - 1
    return page_text[origin[start] : origin[end] + 1]


@dataclass(frozen=True)
class Nearest:
    """How close a rejected span came to anything on the page."""

    coverage: float  # share of the span covered by its longest run found on the page
    longest: int  # that run's length in characters
    context: str  # what the page says around it

    def describe(self) -> str:
        return f"{self.coverage:.0%} of the span is on the page (longest run {self.longest} chars)"


def nearest_match(page_text: str, span: str, context_chars: int = 60) -> Nearest:
    """The longest stretch of `span` that does appear on the page, for the rejection log."""
    folded_text, origin = _fold(page_text)
    folded_span, _ = _fold(span)
    if not folded_span or not folded_text:
        return Nearest(0.0, 0, "")
    block = SequenceMatcher(None, folded_text, folded_span, autojunk=False).find_longest_match(
        0, len(folded_text), 0, len(folded_span)
    )
    at = origin[block.a] if block.size else 0
    context = " ".join(page_text[max(0, at - context_chars) : at + context_chars].split())
    return Nearest(block.size / len(folded_span), block.size, context)
