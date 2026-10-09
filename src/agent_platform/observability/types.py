"""Observability value types (LLD §6)."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

StatusKind = Literal[
    "started",
    "completed",
    "failed",
    "warning",
    "waiting",
    "blocked",
]

EventKind = Literal[
    "stage",
    "decision",
    "security",
    "authorization",
    "intent",
    "memory",
    "context",
    "model",
    "tool_request",
    "tool_result",
    "gatekeeper",
    "executor",
    "resilience",
    "loop_control",
    "grounding",
    "escalation",
    "system",
    # The emitter wrote the card itself: its summary is the heading and its
    # fields are titled sections of lines, rendered as given.
    "narrative",
]

CostComponent = Literal[
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "retrieval",
    "evaluator_calls",
    "human_review",
    "orchestration",
    "infrastructure",
]


def utc_now() -> datetime:
    """Timezone-aware now — never a naive timestamp in a trace or ledger."""
    return datetime.now(timezone.utc)


class StatusEvent(BaseModel):
    """One harness stage starting, finishing or failing (LLD §20).

    This is what the TUI's live feed renders. It lives in `observability`
    rather than `tui` so any layer can emit one without importing the UI.
    """

    agent_id: str = ""
    run_id: str = ""
    stage: str
    status: StatusKind
    timestamp: datetime = Field(default_factory=utc_now)
    event_kind: EventKind = "stage"
    tokens_delta: int | None = None
    detail: str | None = None
    summary: str | None = None
    # A presentation group (a colour name): the feed colours a group's cards alike.
    group: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """One traced hop (E36).

    Carries everything needed to reconstruct *why* a decision happened, not
    merely that a call occurred — a trace that can't answer "why" is
    telemetry, not audit.
    """

    agent_id: str = ""
    trace_id: str
    span_id: str
    parent_id: str | None = None
    name: str
    actor: str = ""
    authority_context: dict[str, Any] = Field(default_factory=dict)
    tool: str | None = None
    policy_version: str = ""
    model_version: str = ""
    state_transition: str | None = None
    result: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class CostStack(BaseModel):
    """Per-request cost broken down by component (E44)."""

    input_tokens: float = 0.0
    output_tokens: float = 0.0
    tool_calls: float = 0.0
    retrieval: float = 0.0
    evaluator_calls: float = 0.0
    human_review: float = 0.0
    orchestration: float = 0.0
    infrastructure: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(self.model_dump().values()))


class SLOMeasurement(BaseModel):
    """One recorded observation for an SLO metric."""

    metric: str
    value: float
    timestamp: datetime = Field(default_factory=utc_now)


class SLOStatus(BaseModel):
    """Current standing of one SLO against its target."""

    metric: str
    target: float
    observed: float
    sample_count: int
    error_budget_remaining: float
    within_target: bool
