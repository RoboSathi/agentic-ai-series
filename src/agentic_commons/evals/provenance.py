"""Golden dataset provenance ledger (E11's exit artifact).

A golden dataset inherits the biases of whoever labelled it. When the
labels record what agents *did* rather than what policy *says*, the eval
certifies the existing practice and quietly punishes correct behaviour.
The ledger makes each label's origin, labeller and basis inspectable, so
that failure is findable rather than baked in.
"""

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_commons.logging import get_logger
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool

# E11's three-bucket relabelling framework.
LabelBasis = Literal["written_policy", "historical_practice", "human_judgment"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS golden_labels (
    case_id     TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    basis       TEXT NOT NULL,
    labeller    TEXT NOT NULL,
    source      TEXT NOT NULL,
    rationale   TEXT NOT NULL,
    superseded  TEXT,
    recorded_at TEXT NOT NULL
);
"""

_INSERT = """
INSERT OR REPLACE INTO golden_labels
    (case_id, label, basis, labeller, source, rationale, superseded, recorded_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
    """Records and audits the provenance of every golden label."""

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        self._pool = pool or SqliteConnectionPool()
        self._logger = get_logger("evals")
        self._pool.execute_script(_SCHEMA)

    def record(self, provenance: LabelProvenance) -> None:
        """Record a label's provenance, replacing any prior entry for the case."""
        self._pool.execute(
            _INSERT,
            (
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
        """Correct a label to written policy, preserving what it replaced.

        The superseded value is kept rather than overwritten: a corrected
        eval set whose corrections are invisible cannot be reviewed.
        """
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
        row = self._pool.query_one("SELECT * FROM golden_labels WHERE case_id = ?", (case_id,))
        return LabelProvenance.model_validate(row) if row else None

    def all_labels(self) -> list[LabelProvenance]:
        """Every recorded label."""
        return [
            LabelProvenance.model_validate(row)
            for row in self._pool.iter_rows("SELECT * FROM golden_labels")
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
