"""Deciding a review item — the one path every kind goes through.

A person approves or denies with a **comment**, always: a decision with no
reason recorded is a decision nobody can later check, and the comment is
what makes the audit trail an account rather than a log of clicks. The
service refuses a decision without one, refuses an item outside the
reviewer's scope as if it did not exist, hands the decision to the
handler for the item's kind (which does what approving means — issues
the refund, sends the removed sentences, activates the skill), then
settles the item, signs it and audits it.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agent_platform.audit import AuditEntry, AuditLedger, NullAuditLedger
from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.review.queue import ReviewQueue
from agent_platform.review.types import ReviewDecision, ReviewItem, ReviewOutcome

# A reviewer who may see every item: no subject scope at all.
EVERYONE: None = None


class ReviewError(ValueError):
    """A decision that cannot be made as asked."""


@runtime_checkable
class ReviewHandler(Protocol):
    """What approving and denying one or more kinds of item does."""

    @property
    def kinds(self) -> frozenset[str]: ...

    def describe(self, item: ReviewItem) -> list[str]: ...

    async def approve(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome: ...

    async def deny(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome: ...


class ReviewService:
    """Lists what a reviewer may see, and applies their decisions."""

    def __init__(
        self,
        queue: ReviewQueue,
        handlers: Sequence[ReviewHandler],
        audit: AuditLedger | None = None,
    ) -> None:
        self._queue = queue
        self._handlers = {kind: handler for handler in handlers for kind in handler.kinds}
        self._audit = audit or NullAuditLedger()
        self._settings = get_settings().review
        self._logger = get_logger("review")

    @property
    def queue(self) -> ReviewQueue:
        return self._queue

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def visible(self, subject: str | None = EVERYONE, kind: str | None = None) -> list[ReviewItem]:
        """Pending items in scope: one subject's (`""` is the agent's own), or everyone's."""
        kinds = (kind,) if kind else self.kinds
        return self._queue.items(kinds=kinds, subject=subject)

    def item(self, item_id: int, subject: str | None = EVERYONE) -> ReviewItem:
        """One item in scope; an item outside it is reported exactly as a missing one."""
        item = self._queue.get(item_id)
        if item is None or item.kind not in self._handlers or not _in_scope(item, subject):
            raise ReviewError(self._settings.words.unknown_item.format(id=item_id))
        return item

    def describe(self, item: ReviewItem) -> list[str]:
        return self._handlers[item.kind].describe(item)

    async def decide(
        self,
        item_id: int,
        decision: ReviewDecision,
        reviewer: str,
        comment: str,
        subject: str | None = EVERYONE,
    ) -> ReviewOutcome:
        """Apply a reviewer's decision; the comment is required, and recorded with it."""
        note = " ".join(comment.split())
        if len(note) < self._settings.min_comment_chars:
            raise ReviewError(self._settings.words.needs_comment)
        item = self.item(item_id, subject)
        if item.status != "pending":
            raise ReviewError(
                self._settings.words.already_decided.format(id=item_id, status=item.status)
            )
        handler = self._handlers[item.kind]
        outcome = await (handler.approve if decision == "approve" else handler.deny)(
            item, reviewer, note
        )
        self._queue.decide(item_id, outcome.status)
        self._queue.sign(item_id, reviewer, note)
        self._record(item, decision, reviewer, note, outcome)
        return outcome

    def _record(
        self,
        item: ReviewItem,
        decision: ReviewDecision,
        reviewer: str,
        comment: str,
        outcome: ReviewOutcome,
    ) -> None:
        """The decision on the audit trail, under the reviewer's name, with their comment."""
        self._audit.record(
            AuditEntry(
                trace_id=self._settings.trace_id.format(id=item.id),
                actor=reviewer,
                action=self._settings.audit_action.format(kind=item.kind, decision=decision),
                authority_context={
                    "item_id": item.id,
                    "kind": item.kind,
                    "subject_id": item.subject_id,
                    "status": outcome.status,
                },
                decision="allow" if decision == "approve" else "deny",
                rationale=comment,
            )
        )


def _in_scope(item: ReviewItem, subject: str | None) -> bool:
    return subject is EVERYONE or item.subject_id == subject
