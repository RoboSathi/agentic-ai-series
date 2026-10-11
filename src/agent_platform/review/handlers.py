"""The platform's own review kinds: memory a curator must confirm, and quarantines.

An agent adds handlers for kinds that are its own (a refund is the
support agent's); these two exist for every agent.
"""

from typing import Any, Protocol

from agent_platform.config import get_settings
from agent_platform.memory.types import MemoryPrincipal, WriteResult
from agent_platform.review.types import ReviewItem, ReviewOutcome

# The frontmatter field a learned skill's table row quotes.
_DESCRIPTION = "description"


class CuratorPipeline(Protocol):
    """The part of the memory write pipeline a curator decides through."""

    def approve_queued(self, queue_id: int, principal: MemoryPrincipal) -> WriteResult: ...

    def deny_queued(
        self, queue_id: int, principal: MemoryPrincipal, reason: str
    ) -> WriteResult: ...


class MemoryReviewHandler:
    """A fact or a skill memory would not take on its own word: a curator decides."""

    def __init__(self, pipeline: CuratorPipeline) -> None:
        self._pipeline = pipeline
        kinds = get_settings().review.kinds
        self._kinds = frozenset({kinds.memory_fact, kinds.skill})

    @property
    def kinds(self) -> frozenset[str]:
        return self._kinds

    def describe(self, item: ReviewItem) -> list[str]:
        fact, reason = _write(item)
        return [f"{fact.get('key', '')} = {fact.get('value', '')}", reason]

    def columns(self, kind: str) -> list[str]:
        return list(get_settings().review.words.columns[kind])

    def cells(self, item: ReviewItem) -> list[str]:
        fact, reason = _write(item)
        if item.kind == get_settings().review.kinds.skill:
            what = _frontmatter_value(str(fact.get("value", "")), _DESCRIPTION)
            return [str(item.id), str(fact.get("key", "")), what, reason, raised(item)]
        source = dict(fact.get("provenance") or {}).get("source", "")
        return [
            str(item.id),
            item.subject_id or none(),
            f"{fact.get('key', '')} = {fact.get('value', '')}",
            str(source) or none(),
            reason or none(),
            raised(item),
        ]

    async def approve(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome:
        result = self._pipeline.approve_queued(int(item.id or 0), _curator(reviewer))
        status = "approved" if result.written else "failed"
        return _outcome(item, status, result.reason)

    async def deny(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome:
        result = self._pipeline.deny_queued(int(item.id or 0), _curator(reviewer), comment)
        return _outcome(item, "denied", result.reason)


class QuarantineHandler:
    """A message security screening stopped. Nothing is ever re-run.

    Approving records that the screen was wrong (a false positive, worth
    knowing when tuning it); denying records that it was right.
    """

    def __init__(self) -> None:
        self._kind = get_settings().review.kinds.quarantine

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset({self._kind})

    def describe(self, item: ReviewItem) -> list[str]:
        detail = item.detail
        return [str(detail.get("reason", "")), str(detail.get("payload", ""))]

    def columns(self, kind: str) -> list[str]:
        return list(get_settings().review.words.columns[kind])

    def cells(self, item: ReviewItem) -> list[str]:
        detail = item.detail
        screen = dict(detail.get("metadata") or {}).get("screen") or detail.get("reason", "")
        return [
            str(item.id),
            str(screen),
            str(detail.get("payload", "")),
            item.subject_id or none(),
            raised(item),
        ]

    async def approve(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome:
        return _outcome(item, "approved", get_settings().review.words.quarantine_released)

    async def deny(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome:
        return _outcome(item, "denied", get_settings().review.words.quarantine_upheld)


def raised(item: ReviewItem) -> str:
    """When an item was queued, as a table shows it (local time)."""
    return item.created_at.astimezone().strftime(get_settings().review.words.time_format)


def none() -> str:
    """What an empty cell shows."""
    return get_settings().review.words.none


def _write(item: ReviewItem) -> tuple[dict[str, Any], str]:
    """The held write's fact (as stored) and why it was held."""
    write = dict(item.detail.get("write") or {})
    return dict(write.get("fact") or {}), str(write.get("reason", ""))


def _frontmatter_value(document: str, key: str) -> str:
    """One frontmatter field of a skill's text ("" when it has none)."""
    prefix = f"{key}:"
    line = next((ln for ln in document.splitlines() if ln.startswith(prefix)), "")
    return line[len(prefix) :].strip().strip("\"'")


def _curator(reviewer: str) -> MemoryPrincipal:
    return MemoryPrincipal(actor=reviewer, role="curator")


def _outcome(item: ReviewItem, status: str, line: str) -> ReviewOutcome:
    return ReviewOutcome.model_validate(
        {"item_id": item.id, "kind": item.kind, "status": status, "lines": [line] if line else []}
    )
