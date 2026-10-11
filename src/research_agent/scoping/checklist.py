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

RECENCY_FIELD = "recency_days"
# Statuses that still need a value: asked, or left to the interviewer to propose.
_UNSETTLED = frozenset({"open", "delegated"})


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


def covered(
    items: Sequence[CoverageItem], item_id: str, value: str, *, derived: bool = False
) -> list[CoverageItem]:
    """A copy with one item answered; `derived` when the interviewer proposed the value."""
    return [
        candidate.model_copy(update={"status": "covered", "value": value, "derived": derived})
        if candidate.item_id == item_id
        else candidate
        for candidate in items
    ]


def vague_answer(
    items: Sequence[CoverageItem], item_id: str, delegate_after: int
) -> list[CoverageItem]:
    """An answer that names nothing to research: the item stays open, asked again with
    examples, until `delegate_after` such answers hand it to the interviewer to propose."""
    moved: list[CoverageItem] = []
    for candidate in items:
        if candidate.item_id == item_id:
            count = candidate.vague_answers + 1
            status: ItemStatus = "delegated" if count >= delegate_after else candidate.status
            candidate = candidate.model_copy(update={"vague_answers": count, "status": status})
        moved.append(candidate)
    return moved


def prefill_window(items: Sequence[CoverageItem], days: int | None) -> list[CoverageItem]:
    """A time window the first message stated is not asked about again."""
    if days is None:
        return list(items)
    return [
        item.model_copy(update={"status": "covered", "value": str(days)})
        if item.brief_field == RECENCY_FIELD and item.status == "open"
        else item
        for item in items
    ]


def open_items(items: Sequence[CoverageItem], required_only: bool = False) -> list[CoverageItem]:
    """Open items; with `required_only`, those that must be settled before finishing.

    That is every required item, and every optional one the interviewer may not skip.
    """
    return [
        item
        for item in items
        if item.status in _UNSETTLED and (item.required or not item.skippable or not required_only)
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
            question=item.goal,
            reason="waived" if item.status == "waived" else closing_reason,
        )
        for item in items
        if item.status == "waived" or (item.status in _UNSETTLED and item.required)
    ]


def item(items: Sequence[CoverageItem], item_id: str) -> CoverageItem | None:
    return next((candidate for candidate in items if candidate.item_id == item_id), None)
