"""The parts of a long text that bear on a question, within a size limit.

A model asked to read a 40,000-character page for one question spends most of its time (and
its stalls) on navigation, comments and footers. This cuts the text into windows on paragraph
boundaries, scores each by the question's terms it contains, and keeps the best windows *in
page order* until the limit is reached. Paragraphs are never altered, only chosen, so a claim
quoted from within one is still a verbatim span of the source.
"""

from agent_platform.llm.embeddings import tokenize

# Joins the kept windows; never inside one, so each window stays verbatim.
SEPARATOR = "\n\n"


def relevant_excerpt(text: str, question: str, max_chars: int, window_chars: int) -> str:
    """The text itself when it fits; otherwise its most relevant windows, in order."""
    if len(text) <= max_chars:
        return text
    windows = _windows(text, window_chars)
    terms = set(tokenize(question))
    ranked = sorted(range(len(windows)), key=lambda i: (-len(terms & set(tokenize(windows[i]))), i))
    kept: list[int] = []
    used = 0
    for index in ranked:
        cost = len(windows[index]) + len(SEPARATOR)
        if used + cost > max_chars:
            continue
        kept.append(index)
        used += cost
    return SEPARATOR.join(windows[i] for i in sorted(kept)) or text[:max_chars]


def _windows(text: str, size: int) -> list[str]:
    """Paragraph-aligned windows of at most `size` characters (a longer paragraph is split)."""
    windows: list[str] = []
    current = ""
    for paragraph in (p for p in text.split("\n") if p.strip()):
        for start in range(0, len(paragraph), size):
            piece = paragraph[start : start + size]
            if current and len(current) + 1 + len(piece) > size:
                windows.append(current)
                current = piece
            else:
                current = f"{current}\n{piece}" if current else piece
    if current:
        windows.append(current)
    return windows


__all__ = ["relevant_excerpt"]
