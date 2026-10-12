"""Why a request is in the review queue: a refund, a cancellation, or a cut reply's finding."""

import pytest

from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.guardrails import ResponseGuardrailResult
from customer_support_agent.config import get_agent_settings
from customer_support_agent.persistence.repositories import HitlQueueRepository
from customer_support_agent.persistence.schema import ensure_schema
from customer_support_agent.review.referral import ReplyReferral


def _cut(reason_code: str) -> ResponseGuardrailResult:
    return ResponseGuardrailResult(
        response="Kept.", escalated=True, reason_code=reason_code, removed_sentences=["Cut."]
    )


class TestACutReplyIsQueuedWithWhatWasFound:
    @pytest.mark.parametrize(
        "code, reason",
        [
            ("CLAIM_UNSUPPORTED", "hallucination_unsupported"),
            ("LOW_CONFIDENCE_DEGENERATE", "low_confidence"),
            ("LOW_CONFIDENCE_SPECIFIC", "low_confidence"),
            ("TOOL_EVIDENCE_RECORD_STATE_MISMATCH", "hallucination_contradicted"),
        ],
    )
    def test_the_reason_names_the_finding(
        self, seeded_pool: SqliteConnectionPool, code: str, reason: str
    ) -> None:
        queue = HitlQueueRepository(seeded_pool)
        request_id = ReplyReferral(queue).refer(_cut(code), customer_id=1, session_id="s")
        assert request_id is not None
        stored = queue.get(request_id)
        assert stored is not None and stored.reason == reason


class TestARefundIsQueuedAsARefund:
    @pytest.mark.parametrize(
        "action, reason", [("refund", "refund"), ("cancel_and_refund", "cancellation")]
    )
    def test_the_reason_names_the_action(self, action: str, reason: str) -> None:
        assert get_agent_settings().review_reasons.for_action(action) == reason


class TestAnOlderQueueMovesIntoTheReviewQueue:
    def test_its_rows_become_review_items_and_the_table_goes(
        self, pool: SqliteConnectionPool
    ) -> None:
        pool.execute_script(
            "CREATE TABLE hitl_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, "
            "user_id INTEGER NOT NULL, requested_amount_cents INTEGER, action TEXT NOT NULL "
            "DEFAULT 'refund', status TEXT NOT NULL, created_at TEXT NOT NULL, session_id TEXT, "
            "detail TEXT);"
            "INSERT INTO hitl_queue (user_id, action, status, created_at) "
            "VALUES (1, 'refund', 'pending', '2026-10-10');"
        )
        ensure_schema(pool)
        [moved] = HitlQueueRepository(pool).pending()
        assert moved.user_id == 1 and moved.reason == ""
        assert not list(pool.iter_rows("PRAGMA table_info(hitl_queue)"))


class TestARetiredRefundStatusIsRewritten:
    def test_an_approved_order_reads_as_completed(self, seeded_pool: SqliteConnectionPool) -> None:
        """`approved` was only ever seeded; an older database still holding it must still open."""
        seeded_pool.execute("UPDATE orders SET refund_status = 'approved' WHERE order_id = 10001")
        ensure_schema(seeded_pool)
        [row] = list(
            seeded_pool.iter_rows("SELECT refund_status FROM orders WHERE order_id = 10001")
        )
        assert row["refund_status"] == "completed"
