"""The research agent's ubiquitous language, as types.

Scope Brief, Rubric, Research Contract, Evidence Ledger (commons), Round,
Batch, Independent Audit, Gap, Known unknown — see design_docs/ubiquitous_language.md.
"""

from research_agent.domain.budget import Budget
from research_agent.domain.evidence import ClaimCandidate, ExtractionResult
from research_agent.domain.plan import PlanDag, PlanValidation, SubQuestion
from research_agent.domain.report import (
    COVERAGE_AND_LIMITS,
    Citation,
    LimitEntry,
    Report,
    ReportSection,
    RunStatus,
    SessionPhase,
)
from research_agent.domain.rubric import Contract, ItemOrigin, Rubric, RubricCheck, RubricItem
from research_agent.domain.scope import (
    RESEARCH_ARCHETYPES,
    Archetype,
    CoverageItem,
    KindVerdict,
    KnownUnknown,
    ScopeBrief,
    SourcePolicy,
)

__all__ = [
    "COVERAGE_AND_LIMITS",
    "RESEARCH_ARCHETYPES",
    "Archetype",
    "Budget",
    "Citation",
    "ClaimCandidate",
    "Contract",
    "CoverageItem",
    "ExtractionResult",
    "KindVerdict",
    "ItemOrigin",
    "KnownUnknown",
    "LimitEntry",
    "PlanDag",
    "PlanValidation",
    "Report",
    "ReportSection",
    "Rubric",
    "RubricCheck",
    "RubricItem",
    "RunStatus",
    "ScopeBrief",
    "SessionPhase",
    "SourcePolicy",
    "SubQuestion",
]
