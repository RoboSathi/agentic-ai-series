"""Golden dataset provenance ledger (E11's exit artifact).

Golden-label provenance is platform-owned agent state. The records are scoped
by ``agent_id`` so a reusable common library can host more than one agent in
the same SQLite file without cross-agent label collisions.
"""

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_commons.logging import get_logger
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

LabelBasis = Literal["written_policy", "historical_practice", "human_judgment"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_golden_labels (
    agent_id    TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    label       TEXT NOT NULL,
    basis       TEXT NOT NULL,
    labeller    TEXT NOT NULL,
    source      TEXT NOT NULL,
    rationale   TEXT NOT NULL,
    superseded  TEXT,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_platform_golden_labels_agent
    ON platform_golden_labels (agent_id, recorded_at);
"""

_INSERT = """
INSERT OR REPLACE INTO platform_golden_labels
    (agent_id, case_id, label, basis, labeller, source, rationale, superseded, recorded_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class LabelProvenance(BaseModel):
    """Where one golden label came from, and on what basis."""

    case_id: str
    label: str
    basis: LabelBasis
    labeller: str
    source: str = ""
    rationale: str = ""
    superseded: str | None = None
    recorded_at: datetime = Field(default_factory=utc_now)


class BiasReport(BaseModel):
    """How much of a golden set rests on practice rather than policy."""

    total: int
    by_basis: dict[str, int] = Field(default_factory=dict)
    practice_ratio: float = 0.0
    relabelled: int = 0


class GoldenDatasetProvenanceLedger:
    """Records and audits golden-label provenance for one agent."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._logger = get_logger("evals")
        self._store.execute_script(_SCHEMA)
        self._migrate_legacy_table()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def record(self, provenance: LabelProvenance) -> None:
        """Record a label's provenance, replacing a prior entry for the case."""
        self._store.execute(
            _INSERT,
            (
                self._agent_id,
                provenance.case_id,
                provenance.label,
                provenance.basis,
                provenance.labeller,
                provenance.source,
                provenance.rationale,
                provenance.superseded,
                provenance.recorded_at.isoformat(),
            ),
        )

    def relabel(self, case_id: str, label: str, labeller: str, rationale: str) -> LabelProvenance:
        """Correct a label to written policy, preserving what it replaced."""
        previous = self.get(case_id)
        updated = LabelProvenance(
            case_id=case_id,
            label=label,
            basis="written_policy",
            labeller=labeller,
            source=previous.source if previous else "",
            rationale=rationale,
            superseded=previous.label if previous else None,
        )
        self.record(updated)
        return updated

    def get(self, case_id: str) -> LabelProvenance | None:
        """Provenance for one case, if recorded."""
        row = self._store.query_one(
            "SELECT case_id, label, basis, labeller, source, rationale, superseded, recorded_at "
            "FROM platform_golden_labels WHERE agent_id = ? AND case_id = ?",
            (self._agent_id, case_id),
        )
        return LabelProvenance.model_validate(row) if row else None

    def all_labels(self) -> list[LabelProvenance]:
        """Every recorded label for this agent."""
        return [
            LabelProvenance.model_validate(row)
            for row in self._store.iter_rows(
                "SELECT case_id, label, basis, labeller, source, rationale, superseded, recorded_at "
                "FROM platform_golden_labels WHERE agent_id = ? ORDER BY case_id",
                (self._agent_id,),
            )
        ]

    def bias_report(self) -> BiasReport:
        """What share of the golden set encodes practice rather than policy."""
        labels = self.all_labels()
        by_basis: dict[str, int] = {}
        for entry in labels:
            by_basis[entry.basis] = by_basis.get(entry.basis, 0) + 1
        total = len(labels)
        practice = by_basis.get("historical_practice", 0)
        return BiasReport(
            total=total,
            by_basis=by_basis,
            practice_ratio=practice / total if total else 0.0,
            relabelled=sum(entry.superseded is not None for entry in labels),
        )

    def export(self) -> str:
        """The ledger as JSON — E11's checked-in exit artifact."""
        return json.dumps(
            [entry.model_dump(mode="json") for entry in self.all_labels()],
            indent=2,
        )

    def _migrate_legacy_table(self) -> None:
        """Copy legacy golden-label rows into the agent-scoped platform table."""
        if not self._table_exists("golden_labels"):
            return
        columns = self._table_columns("golden_labels")
        if "agent_id" in columns:
            rows = self._store.iter_rows(
                "SELECT agent_id, case_id, label, basis, labeller, source, rationale, superseded, recorded_at "
                "FROM golden_labels WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._store.iter_rows(
                "SELECT case_id, label, basis, labeller, source, rationale, superseded, recorded_at "
                "FROM golden_labels"
            )
        for row in rows:
            self._store.execute(
                "INSERT OR IGNORE INTO platform_golden_labels "
                "(agent_id, case_id, label, basis, labeller, source, rationale, superseded, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("agent_id", self._agent_id),
                    row["case_id"],
                    row["label"],
                    row["basis"],
                    row["labeller"],
                    row["source"],
                    row["rationale"],
                    row["superseded"],
                    row["recorded_at"],
                ),
            )

    def _table_exists(self, table: str) -> bool:
        row = self._store.query_one(
            "SELECT 1 AS hit FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        )
        return row is not None

    def _table_columns(self, table: str) -> set[str]:
        return {str(row["name"]) for row in self._store.iter_rows(f"PRAGMA table_info({table})")}
