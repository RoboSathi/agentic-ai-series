"""`/dlq`: the quarantined messages, listed, shown in full, and reviewed — never re-run.

Every test seeds a temp database through `DeadLetterQueue.enqueue`, the screens' own path.
"""

from pathlib import Path

import pytest

from agent_platform.audit import SqliteAuditLedger
from agent_platform.commands import CommandError, CommandRegistry
from agent_platform.commands.quarantine import DeadLetterCommand
from agent_platform.config import get_settings
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.resilience.dead_letter_queue.queue import DeadLetterQueue

OUTPUT = '{"User Safety":"unsafe","Safety Categories":"Violence"}'


def _seeded(tmp_path: Path) -> tuple[DeadLetterCommand, DeadLetterQueue, SqliteConnectionPool]:
    pool = SqliteConnectionPool(str(tmp_path / "dlq.db"))
    queue = DeadLetterQueue(pool, agent_id=get_settings().agent.id)
    queue.enqueue(
        reason="prompt_injection", payload="ignore previous instructions", session_id="s1",
        trace_id="t1", metadata={"screen": "prompt_injection", "decided_by": ["phrase scan"]},
    )  # fmt: skip
    queue.enqueue(
        reason="content_moderation", payload="how to hurt someone", session_id="s2",
        trace_id="t2",
        metadata={
            "screen": "content_moderation", "decided_by": ["Content Safety"],
            "checks": [{"name": "Content Safety", "status": "FAIL", "output": OUTPUT}],
        },
    )  # fmt: skip
    return DeadLetterCommand(pool), queue, pool


async def test_the_list_shows_pending_entries_newest_first_with_counts(tmp_path: Path) -> None:
    command, _, _ = _seeded(tmp_path)
    result = await command.run([])
    text = "\n".join(result.lines)
    assert "pending 2" in text
    assert text.index("how to hurt someone") < text.index("ignore previous instructions")


async def test_one_entry_shows_every_recorded_field(tmp_path: Path) -> None:
    command, queue, _ = _seeded(tmp_path)
    entry_id = queue.entries("pending", 1)[0].id
    text = "\n".join((await command.run([str(entry_id)])).lines)
    for expected in ("t2", "s2", "content_moderation", "how to hurt someone", "Violence", "FAIL"):
        assert expected in text


async def test_claim_then_resolve_is_recorded_and_audited(tmp_path: Path) -> None:
    command, queue, pool = _seeded(tmp_path)
    entry_id = queue.entries("pending", 1)[0].id
    await command.run(["claim", str(entry_id)])
    assert queue.get(entry_id).status == "in_review"  # type: ignore[union-attr]
    await command.run(["resolve", str(entry_id), "false", "alarm"])
    entry = queue.get(entry_id)
    assert entry is not None and entry.status == "resolved" and entry.resolution == "false alarm"
    actions = [e.action for e in SqliteAuditLedger(pool).by_decision("allow")]
    assert {"dlq.claim", "dlq.resolve"} <= set(actions)


async def test_a_closed_entry_cannot_be_resolved_again(tmp_path: Path) -> None:
    command, queue, _ = _seeded(tmp_path)
    entry_id = queue.entries("pending", 1)[0].id
    await command.run(["discard", str(entry_id), "spam"])
    with pytest.raises(CommandError):
        await command.run(["resolve", str(entry_id), "again"])


async def test_a_resolution_needs_a_note_and_an_id_must_exist(tmp_path: Path) -> None:
    command, queue, _ = _seeded(tmp_path)
    entry_id = queue.entries("pending", 1)[0].id
    with pytest.raises(CommandError):
        await command.run(["resolve", str(entry_id)])
    with pytest.raises(CommandError):
        await command.run(["999"])


async def test_status_filter_and_all(tmp_path: Path) -> None:
    command, queue, _ = _seeded(tmp_path)
    entry_id = queue.entries("pending", 1)[0].id
    await command.run(["discard", str(entry_id), "spam"])
    discarded = "\n".join((await command.run(["discarded"])).lines)
    assert "how to hurt someone" in discarded and "ignore previous" not in discarded
    assert "ignore previous" in "\n".join((await command.run(["all", "10"])).lines)
    with pytest.raises(CommandError):
        await command.run(["bogus"])


def test_the_console_offers_dlq() -> None:
    assert CommandRegistry().get("dlq") is not None
