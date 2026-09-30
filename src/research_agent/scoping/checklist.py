"""The coverage checklist: loading it, and moving items between states.

Item state changes only here, and only in three ways — an answer covers
an item, the user waives it, or scoping closes with it open — so the
rules the termination policy relies on cannot be bypassed by a prompt.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from research_agent.domain.scope import (
    Archetype,
    CoverageItem,
    ItemStatus,
    KnownUnknown,
    UnknownReason,
)


def load_checklist(path: Path, archetype: Archetype) -> list[CoverageItem]:
    """The archetype's items (its own, then the common ones), all open."""
    data = yaml.safe_load(path.read_text())
    return [CoverageItem.model_validate(raw) for raw in _flatten(data[archetype])]


def _flatten(entries: Iterable[Any]) -> list[dict[str, Any]]:
    """YAML anchors splice the common list in as a nested list; lay it flat."""
    flat: list[dict[str, Any]] = []
    for entry in entries:
        flat.extend(_flatten(entry) if isinstance(entry, list) else [entry])
    return flat


def with_status(
    items: Sequence[CoverageItem], item_id: str, status: ItemStatus, value: str | None = None
) -> list[CoverageItem]:
    """A copy with one item moved; unknown ids leave the list unchanged."""
    return [
        item.model_copy(update={"status": status, "value": item.value if value is None else value})
        if item.item_id == item_id
        else item
        for item in items
    ]


def open_items(items: Sequence[CoverageItem], required_only: bool = False) -> list[CoverageItem]:
    return [
        item for item in items if item.status == "open" and (item.required or not required_only)
    ]


def waive(items: Sequence[CoverageItem], target: str, all_keyword: str) -> list[CoverageItem]:
    """Waive one open item by id, or every open item with the `all` keyword."""
    targets = {item.item_id for item in open_items(items)} if target == all_keyword else {target}
    return [
        item.model_copy(update={"status": "waived"}) if item.item_id in targets else item
        for item in items
    ]


def known_unknowns(
    items: Sequence[CoverageItem], closing_reason: UnknownReason
) -> list[KnownUnknown]:
    """Waived items, and required items still open, as known unknowns.

    An optional item left unanswered falls back to its default (any
    recency, no source preference) rather than becoming an unknown: the
    user was not obliged to have an opinion on it.
    """
    return [
        KnownUnknown(
            item_id=item.item_id,
            question=item.prompt,
            reason="waived" if item.status == "waived" else closing_reason,
        )
        for item in items
        if item.status == "waived" or (item.status == "open" and item.required)
    ]


def item(items: Sequence[CoverageItem], item_id: str) -> CoverageItem | None:
    return next((candidate for candidate in items if candidate.item_id == item_id), None)
