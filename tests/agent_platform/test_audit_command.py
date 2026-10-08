"""`/audit`: the trail the agents actually write, newest first, with each decision's standing.

Every test records through `DuckDbAuditLedger.record`, the agents' own path, into an
in-memory store, and reads it back through the command.
"""

from datetime import timedelta

import pytest

from agent_platform.audit import STANDING_FIELD, AuditEntry, Decision, DuckDbAuditLedger
from agent_platform.commands import CommandError
from agent_platform.commands.inspection import AuditCommand
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import utc_now


def _ledger() -> DuckDbAuditLedger:
    return DuckDbAuditLedger(DuckDbObservabilityStore(":memory:"), agent_id="agent-under-test")


def _record(
    ledger: DuckDbAuditLedger, action: str, decision: Decision, standing: str, trace: str
) -> None:
    ledger.record(
        AuditEntry(
            trace_id=trace,
            actor="agent-under-test",
            action=action,
            authority_context={STANDING_FIELD: standing},
            decision=decision,
            rationale=f"{action} rationale",
            timestamp=utc_now() + timedelta(seconds=ledger.count()),
        )
    )


@pytest.fixture
def ledger() -> DuckDbAuditLedger:
    trail = _ledger()
    _record(trail, "support.get_order_status", "allow", "VALID", "t1")
    _record(trail, "support.process_refund", "escalate", "RATIFIABLE", "t1")
    _record(trail, "support.process_refund", "deny", "VOID", "t2")
    return trail


def _text(lines: list[str]) -> str:
    return "\n".join(lines)


async def test_lists_what_the_agents_recorded_newest_first(ledger: DuckDbAuditLedger) -> None:
    text = _text((await AuditCommand(ledger).run([])).lines)
    assert text.index("VOID") < text.index("RATIFIABLE") < text.index("VALID")


async def test_every_row_shows_its_standing_and_trace(ledger: DuckDbAuditLedger) -> None:
    text = _text((await AuditCommand(ledger).run([])).lines)
    for expected in ("VALID", "RATIFIABLE", "VOID", "t1", "t2"):
        assert expected in text


async def test_a_limit_keeps_only_the_newest(ledger: DuckDbAuditLedger) -> None:
    text = _text((await AuditCommand(ledger).run(["1"])).lines)
    assert "VOID" in text and "RATIFIABLE" not in text


async def test_narrows_by_decision(ledger: DuckDbAuditLedger) -> None:
    result = await AuditCommand(ledger).run(["escalate"])
    text = _text(result.lines)
    assert "RATIFIABLE" in text and "VOID" not in text and "escalate" in result.title


async def test_narrows_by_standing(ledger: DuckDbAuditLedger) -> None:
    text = _text((await AuditCommand(ledger).run(["VOID"])).lines)
    assert "VOID" in text and "VALID" not in text


async def test_trace_shows_one_trace_in_causal_order(ledger: DuckDbAuditLedger) -> None:
    result = await AuditCommand(ledger).run(["trace", "t1"])
    text = _text(result.lines)
    assert "t2" not in text
    assert text.index("VALID") < text.index("RATIFIABLE")


async def test_an_unknown_filter_is_an_error_not_an_empty_table(
    ledger: DuckDbAuditLedger,
) -> None:
    with pytest.raises(CommandError):
        await AuditCommand(ledger).run(["maybe"])


async def test_trace_without_an_id_is_an_error(ledger: DuckDbAuditLedger) -> None:
    with pytest.raises(CommandError):
        await AuditCommand(ledger).run(["trace"])


async def test_another_agents_entries_are_not_shown(ledger: DuckDbAuditLedger) -> None:
    _record(_ledger(), "web.fetch", "allow", "VALID", "t3")
    text = _text((await AuditCommand(ledger).run([])).lines)
    assert "web.fetch" not in text
