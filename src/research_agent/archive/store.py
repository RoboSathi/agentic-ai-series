"""Archiving a finished run: exactly three artifacts and one index row.

Kept: the Brief, the Rubric and the Report — frozen, hashed, written to
the run's directory with a Markdown rendering each. Indexed: one row (id,
archetype, status, budget spent, unmet items, artifact paths) as an
episodic memory event. Remembered: the report as a semantic fact, so a
later session can find it. Discarded: the Evidence Ledger — it is
run-scoped working state, and the report's citations already carry what
an auditor needs.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from agent_platform.artifacts import DirectoryArtifactStore, FrozenArtifact
from agent_platform.logging import instrument
from agent_platform.memory import MemoryFacade
from agent_platform.memory.types import MemoryCommand, MemoryPrincipal
from research_agent.archive.render import render_brief, render_report, render_rubric
from research_agent.domain.report import Report
from research_agent.domain.rubric import Contract
from research_agent.rubric.freeze import BRIEF_ARTIFACT, RUBRIC_ARTIFACT

REPORT_ARTIFACT = "report"
INDEX_CATEGORY = "research_run"
ARCHIVIST = "research_archive"


class RunIndexRow(BaseModel):
    """The one row a finished run leaves in the index."""

    run_id: str
    archetype: str
    topic: str
    status: str
    contract_hash: str
    budget_spent: dict[str, float] = Field(default_factory=dict)
    unmet_items: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


@instrument("research_agent")
class ResearchArchive:
    """Writes the three artifacts, the index row and the semantic fact."""

    def __init__(
        self,
        store: DirectoryArtifactStore,
        memory: MemoryFacade | None,
        index_session: str,
        semantic_pointer: bool = False,
    ) -> None:
        self._store = store
        self._memory = memory
        self._index_session = index_session
        # The episodic row is the run pointer; a semantic "fact" about it is opt-in.
        self._semantic_pointer = semantic_pointer

    def archive(self, contract: Contract, report: Report, spent: dict[str, float]) -> RunIndexRow:
        run_id = contract.contract_id
        paths = [
            self._store.write(
                run_id,
                FrozenArtifact.freeze(BRIEF_ARTIFACT, run_id, contract.brief),
                render_brief(contract),
            ),
            self._store.write(
                run_id,
                FrozenArtifact.freeze(RUBRIC_ARTIFACT, run_id, contract.rubric),
                render_rubric(contract),
            ),
            self._store.write(
                run_id,
                FrozenArtifact.freeze(REPORT_ARTIFACT, run_id, report),
                render_report(report),
            ),
        ]
        row = RunIndexRow(
            run_id=run_id,
            archetype=contract.brief.archetype,
            topic=contract.brief.topic,
            status=report.status,
            contract_hash=contract.contract_hash,
            budget_spent=spent,
            unmet_items=[limit.item_id for limit in report.limits],
            artifact_paths=[self._relative(str(p)) for p in paths],
        )
        self._index(row, report)
        return row

    def history(self, limit: int) -> list[RunIndexRow]:
        """The most recent index rows, newest first."""
        if self._memory is None or self._memory.episodic is None:
            return []
        events = self._memory.episodic.events(self._index_session)
        rows = [
            RunIndexRow.model_validate_json(e.value) for e in events if e.category == INDEX_CATEGORY
        ]
        return list(reversed(rows))[:limit]

    def report(self, run_id: str) -> str | None:
        return self._store.rendered(run_id, REPORT_ARTIFACT)

    def root(self) -> Path:
        return self._store.root

    def _relative(self, path: str) -> str:
        """The path under the artifact root, so no machine-specific prefix is remembered."""
        try:
            return str(Path(path).relative_to(self._store.root))
        except ValueError:
            return path

    def _index(self, row: RunIndexRow, report: Report) -> None:
        if self._memory is None:
            return
        principal = MemoryPrincipal(actor=ARCHIVIST, role="agent")
        self._memory.record_event(
            row.model_dump_json(),
            ARCHIVIST,
            self._index_session,
            principal=principal,
            category=INDEX_CATEGORY,
            key=f"run:{row.run_id}",
        )
        # A run that established nothing is indexed (so /history shows it) but not
        # remembered as a fact a later session should rely on.
        if self._semantic_pointer and self._memory.semantic is not None and report.citations:
            self._memory.submit(
                MemoryCommand(
                    principal=principal,
                    operation="create",
                    memory_type="semantic",
                    key=f"{row.run_id}:{REPORT_ARTIFACT}",
                    value=json.dumps(
                        {
                            "title": report.title,
                            "status": report.status,
                            "topic": row.topic,
                            "path": row.artifact_paths[-1],
                        }
                    ),
                )
            )
