"""What one dream changed, as data and as a tree."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_commons.text_tree import TreeNode, tree_lines

# mem0's reconcile vocabulary, plus the curator hand-off governance adds.
ReconcileOp = Literal["ADD", "UPDATE", "DELETE", "NOOP", "QUEUED", "REFUSED"]
Store = Literal["Episodic", "Semantic", "Procedural"]

_MARKS: dict[ReconcileOp, str] = {
    "ADD": "+",
    "UPDATE": "~",
    "DELETE": "-",
    "NOOP": "=",
    "QUEUED": "?",
    "REFUSED": "✗",
}
_STORES: tuple[Store, ...] = ("Episodic", "Semantic", "Procedural")
_AGENT_WIDE = "agent-wide"
# Events recorded before turn events carried a subject, with no log to recover it from.
_UNKNOWN_CUSTOMER = "customer unknown"


class DreamEntry(BaseModel):
    """One thing a dream did (or chose not to do) to one store."""

    store: Store
    op: ReconcileOp
    label: str
    subject: str = ""
    session: str = ""


class DreamReport(BaseModel):
    """Everything one dream read and changed."""

    started_at: datetime
    events: int = 0
    sessions: int = 0
    entries: list[DreamEntry] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    pruned: int = 0
    notes: list[str] = Field(default_factory=list)

    def add(self, entry: DreamEntry) -> None:
        self.entries.append(entry)

    def count(self, op: ReconcileOp) -> int:
        return sum(1 for entry in self.entries if entry.op == op)


def change_tree_lines(report: DreamReport) -> list[str]:
    """The dream as a tree: store → customer → session → change."""
    root = TreeNode(
        f"Dream · {report.started_at:%Y-%m-%d %H:%M} · "
        f"{plural(report.events, 'turn event')} from {plural(report.sessions, 'session')}"
    )
    for store in _STORES:
        _store_branch(root.add(store), [e for e in report.entries if e.store == store], store)
    if report.skipped:
        skipped = root.add(
            f"Skipped — {plural(len(report.skipped), 'session')} with nothing worth remembering"
        )
        for session in report.skipped:
            skipped.add(f"session {session}")
    return [*tree_lines(root), _totals(report), *report.notes]


def _store_branch(branch: TreeNode, entries: list[DreamEntry], store: Store) -> None:
    if not entries:
        branch.add("no changes")
        return
    by_subject: dict[str, dict[str, list[DreamEntry]]] = {}
    for entry in entries:
        by_subject.setdefault(entry.subject, {}).setdefault(entry.session, []).append(entry)
    for subject, by_session in sorted(by_subject.items()):
        nobody = _AGENT_WIDE if store == "Procedural" else _UNKNOWN_CUSTOMER
        person = branch.add(f"customer {subject}" if subject else nobody)
        for session, items in by_session.items():
            parent = person.add(f"session {session}") if session else person
            for entry in items:
                parent.add(f"{_MARKS[entry.op]} {entry.label}")


def plural(number: int, noun: str) -> str:
    """`1 session`, `2 sessions`."""
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _totals(report: DreamReport) -> str:
    return (
        f"Totals: {report.count('ADD')} added · {report.count('UPDATE')} updated · "
        f"{report.count('QUEUED')} queued for curator · {report.count('REFUSED')} refused · "
        f"{report.count('NOOP')} unchanged · {report.pruned} old events pruned"
    )
