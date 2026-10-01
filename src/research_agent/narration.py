"""The words the operator's cards and status block are written in.

Every sentence a card says lives in `config.yaml` under `tui.cards`, keyed
by card and part (`scope_gate.heading`), and is filled in here. Code decides
*what* happened; this module is the only place that says it in English, so
changing a card's wording is a config edit, never a code change.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from research_agent.config import get_agent_settings

Line = dict[str, str]


def say(key: str, **values: Any) -> str:
    """The configured sentence for `key`, with `values` filled in."""
    return get_agent_settings().tui.cards[key].format(**values)


def describe(prefix: str, name: str) -> str:
    """`name` as the card words it under `prefix`, or as plain words when none is configured."""
    return get_agent_settings().tui.cards.get(f"{prefix}.{name}", title_of(name))


def title_of(identifier: str) -> str:
    """An identifier as words: `decision_support` -> `decision support`."""
    return identifier.replace("_", " ")


def toned(text: str, tone: str) -> Line:
    """One line of a card section, coloured by the TUI (`pass`, `fail`, `warn`, `dim`)."""
    return {"text": text, "tone": tone}


def reason_text(code: str, reasons: Mapping[str, str]) -> str:
    """A rejection or gap code in words; the code itself when none is configured."""
    return reasons.get(code, title_of(code))


def clip(text: str, limit: int) -> str:
    """`text` cut to `limit` characters with an ellipsis, for headings and list lines."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: max(0, limit - 1)].rstrip() + "…"


def bullets(items: Iterable[str]) -> list[str]:
    return [f"• {item}" for item in items]
