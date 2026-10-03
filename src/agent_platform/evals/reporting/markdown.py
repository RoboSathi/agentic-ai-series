"""Markdown building blocks: tables, rates, code fences."""

import json
from collections.abc import Sequence
from typing import Any

from agent_platform.evals.reporting.stats import wilson_interval

_PIPE = "|"
_ESCAPED_PIPE = "\\|"
_EMPTY = "—"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A GitHub-flavoured Markdown table; an empty one says so."""
    if not rows:
        return "_None._"
    lines = [
        _row(headers),
        _row(["---"] * len(headers)),
        *(_row([cell(value) for value in row]) for row in rows),
    ]
    return "\n".join(lines)


def cell(value: Any) -> str:
    """One table cell: no pipes, no newlines, and an explicit dash for nothing."""
    if value is None or value == "":
        return _EMPTY
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value).replace(_PIPE, _ESCAPED_PIPE)
    return " ".join(text.split())


def rate(passed: int, total: int, z: float) -> str:
    """ "42/50 (84.0%, 95% CI 71–92%)" — a rate is never shown without its sample."""
    if total <= 0:
        return _EMPTY
    low, high = wilson_interval(passed, total, z)
    return f"{passed}/{total} ({passed / total:.1%}, CI {low:.0%}–{high:.0%})"


def fenced(value: Any, language: str = "json") -> str:
    """A fenced block; structured values are pretty-printed."""
    body = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    return f"```{language}\n{body}\n```"


def quote(text: str) -> str:
    """A blockquote that survives multi-line replies."""
    return "\n".join(f"> {line}" if line else ">" for line in (text or _EMPTY).splitlines())


def _row(values: Sequence[Any]) -> str:
    return f"{_PIPE} " + f" {_PIPE} ".join(str(value) for value in values) + f" {_PIPE}"
