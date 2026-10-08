"""The research agent's audit entries sit under the turn that caused them, naming their run."""

from agent_platform.audit import AuditEntry
from agent_platform.logging.trace_context import bound_log_context
from research_agent.observability.telemetry import RUN_KEY, RunTelemetry


class _Ledger:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        return [entry for entry in self.entries if entry.trace_id == trace_id]


def test_inside_a_turn_the_entry_carries_the_turns_trace_and_its_run() -> None:
    ledger = _Ledger()
    telemetry = RunTelemetry(None, ledger, trace_id="run-abc")
    with bound_log_context(agent="research_agent", session="s", trace_id="turn-trace"):
        telemetry.audit("web.search", "allow", "ok")
    (entry,) = ledger.entries
    assert entry.trace_id == "turn-trace"
    assert entry.authority_context[RUN_KEY] == "run-abc"


def test_outside_a_turn_the_run_is_the_trace() -> None:
    ledger = _Ledger()
    RunTelemetry(None, ledger, trace_id="run-abc").audit("run.finished", "allow", "done")
    assert ledger.entries[0].trace_id == "run-abc"
