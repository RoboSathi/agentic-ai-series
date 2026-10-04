"""What flows through the two graphs, and what their checkpoints hold.

The scoping graph's state is a pydantic model: one session's interview,
checklist and (once approved) contract. The run graph's state is a
TypedDict because batches fan out with LangGraph `Send`s whose results are
merged by reducers; its values are the same pydantic domain types.

Neither holds a live collaborator (a gateway, a ledger object, a model
client). What a resumed run needs back — the ledger, the spend, the trace
— is snapshotted into run state at every batch boundary.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from agent_platform.evidence import LedgerSnapshot
from research_agent.audit.independent import AuditResult
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.report import Report, RunStatus
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import Archetype, CoverageItem, ScopeBrief
from research_agent.domain.trace import RunTrace
from research_agent.research.worker import NodeResult

ScopePhase = Literal["KIND", "INTERVIEW", "RUBRIC_REVIEW", "APPROVED", "REFUSED"]


class ScopingState(BaseModel):
    """One scope's interview, from first message to approved contract."""

    session_id: str = ""
    first_message: str = ""
    phase: ScopePhase = "KIND"
    archetype: Archetype | None = None
    topic: str = ""
    # A time window read from the first message, in days.
    recent_days: int | None = None
    items: list[CoverageItem] = Field(default_factory=list)
    turn: int = 0
    asked_item_id: str = ""
    since_challenge: int = 0
    draft_plan: list[str] = Field(default_factory=list)
    # The interviewer proposed to finish (with `draft_plan`): a "yes" now closes the interview.
    proposed: bool = False
    user_input: str = ""
    reply: str = ""
    # Said before the next question (e.g. which items a waiver moved).
    notice: str = ""
    brief: ScopeBrief | None = None
    rubric: Rubric | None = None
    contract: Contract | None = None


def _merge(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


class RunState(TypedDict, total=False):
    """One autonomous run, checkpointed at every batch boundary."""

    contract: Contract
    plans: list[PlanDag]
    round: int
    batch: int
    round_nodes: list[SubQuestion]
    done: Annotated[dict[str, str], _merge]
    results: Annotated[list[NodeResult], operator.add]
    task: SubQuestion
    ledger: LedgerSnapshot
    spent: dict[str, float]
    trace: RunTrace
    gaps: list[str]
    entities: int
    report: Report | None
    audit: AuditResult | None
    # Why the independent audit could not run at all (empty: it ran, or was withheld).
    audit_error: str
    gap_fill_round: int
    status: RunStatus | None
    stop_reason: str
    toggles: dict[str, bool]
    index: dict[str, Any]


class StateTypes(BaseModel):
    """Every model a checkpoint may contain — registered with the serializer."""

    scoping: ScopingState
    contract: Contract
    plan: PlanDag
    node: SubQuestion
    result: NodeResult
    ledger: LedgerSnapshot
    trace: RunTrace
    report: Report
    audit: AuditResult
