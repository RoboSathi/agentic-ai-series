"""A reply the hallucination check cut goes to the one human review queue.

The card used to say "escalated to human review" when nothing reached a
person. Now a trimmed or withheld reply is queued beside refunds: approving
it says the check was wrong and sends the customer what was removed;
denying it says the check was right, and nothing more is sent.
"""

from agent_platform.guardrails import ResponseGuardrailResult
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.console.review_handler import support_queue_command
from customer_support_agent.persistence.repositories import HitlQueueRepository
from customer_support_agent.persistence.schema import ensure_schema
from customer_support_agent.review.referral import ReplyReferral

_SESSION = "session-1"
_REMOVED = "The $66.56 refund has been processed."


def _trimmed() -> ResponseGuardrailResult:
    return ResponseGuardrailResult(
        response="Your order 10193 has been cancelled.",
        escalated=True,
        grounded=False,
        reason_code="TOOL_EVIDENCE_RECORD_STATE_MISMATCH",
        removed_sentences=[_REMOVED],
        total_sentences=2,
    )


class _Recipient:
    """An agent a decision is reported back through."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def deliver_follow_up(self, session_id: str, reply: str) -> None:
        self.sent.append((session_id, reply))


def _queue(pool: SqliteConnectionPool) -> int:
    request_id = ReplyReferral(HitlQueueRepository(pool)).refer(
        _trimmed(), customer_id=1, session_id=_SESSION
    )
    assert request_id is not None
    return request_id


class TestACutReplyIsQueued:
    def test_it_is_a_reply_review_with_what_was_removed(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        request = HitlQueueRepository(seeded_pool).get(_queue(seeded_pool))
        assert request is not None and request.is_reply
        assert request.order_id is None and request.requested_amount is None
        assert request.session_id == _SESSION
        assert request.detail is not None and request.detail.removed == [_REMOVED]

    def test_a_reply_sent_whole_is_not_queued(self, seeded_pool: SqliteConnectionPool) -> None:
        referral = ReplyReferral(HitlQueueRepository(seeded_pool))
        clean = ResponseGuardrailResult(response="Hello.")
        assert referral.refer(clean, customer_id=1, session_id=_SESSION) is None

    def test_it_never_blocks_a_refund_for_the_same_order(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        _queue(seeded_pool)
        assert HitlQueueRepository(seeded_pool).pending_for_order(10193) is None

    async def test_the_queue_lists_it_with_the_removed_text(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        _queue(seeded_pool)
        result = await support_queue_command(seeded_pool).run([])
        [table] = result.tables
        assert table.title.startswith("reply") and _REMOVED in " ".join(table.rows[0])


class TestTheReviewerDecides:
    async def test_approving_sends_the_customer_what_was_removed(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        request_id, recipient = _queue(seeded_pool), _Recipient()
        result = await support_queue_command(seeded_pool, agent=lambda: recipient).run(
            ["approve", str(request_id), "the check was wrong"]
        )
        assert ": approved by" in result.title
        [(session, reply)] = recipient.sent
        assert session == _SESSION and _REMOVED in reply
        assert any(line.startswith("└─ customer: told:") for line in result.lines)
        assert HitlQueueRepository(seeded_pool).pending() == []

    async def test_denying_sends_nothing(self, seeded_pool: SqliteConnectionPool) -> None:
        request_id, recipient = _queue(seeded_pool), _Recipient()
        result = await support_queue_command(seeded_pool, agent=lambda: recipient).run(
            ["deny", str(request_id), "the check was right"]
        )
        assert ": denied by" in result.title
        assert recipient.sent == []
        assert not any("customer" in line for line in result.lines)

    async def test_the_output_names_no_thread(self, seeded_pool: SqliteConnectionPool) -> None:
        request_id = _queue(seeded_pool)
        result = await support_queue_command(seeded_pool, agent=_Recipient).run(
            ["approve", str(request_id), "fine"]
        )
        assert not any("Resumed" in line for line in result.lines)
        branches = [line for line in result.lines if line.startswith(("├─", "└─"))]
        assert [line.split(":")[0] for line in branches] == [
            "├─ request",
            "├─ decided",
            "├─ effect",
            "└─ customer",
        ]


class TestAnOlderQueueIsUpgraded:
    def test_its_rows_survive_and_it_takes_reply_reviews(self, pool: SqliteConnectionPool) -> None:
        pool.execute_script(
            "CREATE TABLE hitl_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "order_id INTEGER NOT NULL, user_id INTEGER NOT NULL, "
            "requested_amount_cents INTEGER NOT NULL, action TEXT NOT NULL DEFAULT 'refund', "
            "status TEXT NOT NULL, created_at TEXT NOT NULL);"
            "INSERT INTO hitl_queue (order_id, user_id, requested_amount_cents, status, "
            "created_at) VALUES (10001, 1, 2000, 'pending', '2026-01-01');"
        )
        ensure_schema(pool)
        [old] = HitlQueueRepository(pool).pending()
        assert old.order_id == 10001 and old.requested_amount is not None
        assert not list(pool.iter_rows("PRAGMA table_info(hitl_queue)"))
