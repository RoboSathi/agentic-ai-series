"""Reading the audit trail (audit/view.py): every ledger as one, coloured by outcome.

Two real SQLite ledgers stand in for the shared and the agent's own: what
is under test is the merge, the filters and the rendering.
"""

from datetime import timedelta

import pytest

from agent_platform.audit import AuditEntry, SqliteAuditLedger
from agent_platform.audit.view import (
    AuditQuery,
    AuditTrail,
    colour,
    since,
    table_lines,
    trace_tree_lines,
)
from agent_platform.config import get_settings
from agent_platform.observability.types import utc_now
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool


def _ledger() -> SqliteAuditLedger:
    return SqliteAuditLedger(SqliteConnectionPool(IN_MEMORY), agent_id="a")


def _entry(
    action: str, decision: str = "allow", minutes_ago: int = 0, **context: str
) -> AuditEntry:
    return AuditEntry.model_validate(
        {
            "trace_id": "t1",
            "actor": "x",
            "action": action,
            "decision": decision,
            "authority_context": context,
            "rationale": f"why {action}",
            "timestamp": utc_now() - timedelta(minutes=minutes_ago),
        }
    )


@pytest.fixture
def trail() -> AuditTrail:
    shared, own = _ledger(), _ledger()
    shared.record(_entry("support.process_refund", "escalate", 30, standing="RATIFIABLE"))
    own.record(_entry("review.refund.approve", "allow", 10))
    own.record(_entry("memory.write", "deny", 60 * 48))
    return AuditTrail([shared, own])


def test_both_ledgers_are_one_trail_newest_first(trail: AuditTrail) -> None:
    actions = [e.action for e in trail.entries(AuditQuery(limit=10))]
    assert actions == ["review.refund.approve", "support.process_refund", "memory.write"]


def test_it_narrows_by_kind_since_and_decision(trail: AuditTrail) -> None:
    assert [e.action for e in trail.entries(AuditQuery(limit=10, kind="review."))] == [
        "review.refund.approve"
    ]
    assert len(trail.entries(AuditQuery(limit=10, since=since("24h")))) == 2
    assert [e.decision for e in trail.entries(AuditQuery(limit=10, decision="deny"))] == ["deny"]


def test_one_trace_reads_in_order_with_its_reasons(trail: AuditTrail) -> None:
    lines = trace_tree_lines("t1", trail.trace("t1"))
    assert lines[0].split(" · ")[1] == "memory.write"
    assert any("standing: RATIFIABLE" in line for line in lines)


def test_an_outcome_is_coloured_by_standing_then_decision() -> None:
    view = get_settings().audit.view
    assert colour(_entry("a", "allow", standing="VOID")) == view.standing_colours["VOID"]
    assert colour(_entry("a", "deny")) == view.decision_colours["deny"]


def test_the_table_has_the_configured_headers(trail: AuditTrail) -> None:
    lines = table_lines(trail.entries(AuditQuery(limit=10)))
    assert all(header in lines[1] for header in get_settings().commands.audit.headers)


def test_since_accepts_a_date() -> None:
    assert since("2026-10-11").date().isoformat() == "2026-10-11"
