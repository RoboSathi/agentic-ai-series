from __future__ import annotations

from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.resilience import DeadLetterQueue


def test_dead_letter_entry_can_be_claimed_and_resolved() -> None:
    pool = SqliteConnectionPool(":memory:")
    queue = DeadLetterQueue(pool, agent_id="customer-support-test")
    entry = queue.enqueue(
        reason="content_moderation",
        payload="[REDACTED] customer message",
        session_id="session-2",
        customer_id=41,
        metadata={"category": "blocked_term"},
    )
    assert entry is not None
    assert entry.status == "pending"
    assert queue.mark_in_review(entry.id, "reviewer-1") is True
    assert queue.resolve(entry.id, "reviewer-1", "reviewed and discarded", discarded=True) is True
    assert queue.pending() == []
