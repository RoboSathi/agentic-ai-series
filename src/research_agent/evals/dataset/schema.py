"""What a research eval case carries in its agent-owned `payload`.

The platform stores and hashes a case's payload without reading it. Here
it is typed: the scripted user's answers (by checklist item), the budget
the case runs under, an optional planted flaw, and the properties the
oracle expects of the outcome.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from agentic_commons.evals.contracts import GoldenCase
from research_agent.domain.report import RunStatus
from research_agent.fixture_world.scripted_user import UserScript

FlawKind = Literal[
    "unsupported_claim", "fabricated_citation", "missed_item", "false_coverage", "tradeoffs_omitted"
]


class PlantedFlaw(BaseModel):
    """A defect written into a delivered report, for the audit to catch.

    `text` is the sentence planted (`{entry}` becomes a real citation id);
    `target` is the phrase whose lines are removed; `markers` open the
    lines a flaw strips. What the flaw says is data, never code.
    """

    kind: FlawKind
    text: str = ""
    target: str = ""
    markers: list[str] = Field(default_factory=list)


class ResearchExpectation(BaseModel):
    """What must be true of the outcome, beyond the invariants every case gets."""

    status: RunStatus
    archetype: str = ""
    # Security: strings a hostile page tried to get into the report.
    report_excludes: list[str] = Field(default_factory=list)


class ResearchCasePayload(BaseModel):
    """The typed payload of one research case."""

    script: UserScript
    budget: dict[str, float] = Field(default_factory=dict)
    planted_flaw: PlantedFlaw | None = None
    expected: ResearchExpectation

    @classmethod
    def of(cls, case: GoldenCase) -> "ResearchCasePayload":
        return cls.model_validate(case.payload)


class _CaseFile(BaseModel):
    cases: list[GoldenCase]


def load_cases(path: Path, suite: str) -> list[GoldenCase]:
    """Load and validate a suite file; every payload must parse."""
    cases = _CaseFile.model_validate(yaml.safe_load(path.read_text())).cases
    for case in cases:
        ResearchCasePayload.of(case)
        if case.suite != suite:
            raise ValueError(f"case {case.case_id} belongs to '{case.suite}', not '{suite}'")
    return cases
