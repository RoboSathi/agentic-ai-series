"""A direct answer written as prose, turned into points — the safety net behind the prompt.

The answer prompt asks for points; a model sometimes still writes a paragraph. When the body
has no list lines and enough sentences, each sentence becomes one point. It splits only where a
sentence plainly ends — after `.`, `!` or `?`, a space, then a capital letter, a digit or an
opening quote — so a decimal (`96.5`), an initial, `e.g.` or a URL never splits. The Sources
block and anything after it are left exactly as written.
"""

import re

# A sentence boundary: terminal punctuation, whitespace, then what a sentence starts with.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")
# Abbreviations that end with a period but do not end a sentence.
_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|Mr|Ms|Dr|Inc|Ltd|Co|No|St)\.$")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def as_points(text: str, sources_heading: str, marker: str, min_sentences: int) -> str:
    """`text` with a prose body rewritten one sentence per point; anything else unchanged."""
    body, sep, tail = text.partition(sources_heading)
    lines = [line for line in body.strip().splitlines() if line.strip()]
    if not lines or any(_LIST_LINE.match(line) for line in lines):
        return text
    sentences = _sentences(" ".join(line.strip() for line in lines))
    if len(sentences) < min_sentences:
        return text
    points = "\n".join(f"{marker} {sentence}" for sentence in sentences)
    return f"{points}\n\n{sep}{tail}".rstrip() if sep else points


def _sentences(text: str) -> list[str]:
    pieces = _BOUNDARY.split(text)
    merged: list[str] = []
    for piece in pieces:
        if merged and _ABBREVIATION.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [sentence.strip() for sentence in merged if sentence.strip()]


__all__ = ["as_points"]
