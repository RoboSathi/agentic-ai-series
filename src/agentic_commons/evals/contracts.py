"""The eval framework's vocabulary, as types (LLD §18).

These are the nouns every stage speaks — a golden case, the transcript
of the agent attempting it, the scores graders give that transcript —
and the Protocols an agent implements to take part. The platform owns
the shapes; an agent owns what goes inside the opaque `payload` and
`side_effects` fields, which is how a golden case can say "refund order
90001" without this package ever learning what an order is.
"""

import hashlib
import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.evals.types import ScorerTier

GraderKind = Literal["oracle", "judge", "deepeval", "promptfoo"]
Severity = Literal["hard", "soft"]
StageName = Literal["generate", "oracle", "security", "judge", "deepeval"]


def canonical_hash(value: Any) -> str:
    """SHA-256 of a value's canonical JSON — stable across key order and runs."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


class GoldenCase(BaseModel):
    """One curated input with the expectations an agent attaches to it.

    `turns` are what a user says, in order. `payload` is agent-owned:
    fixture references, expected outcomes, rubric overrides. The platform
    stores and hashes it but never reads inside it.
    """

    suite: str
    case_id: str
    category: str
    tags: list[str] = Field(default_factory=list)
    turns: list[str]
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def case_hash(self) -> str:
        """Changes whenever anything about the case changes."""
        return canonical_hash(self.model_dump(mode="json"))


class ToolCallRecord(BaseModel):
    """One tool invocation the agent attempted, and what became of it."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = ""
    decision: str = ""
    reason_code: str = ""
    output: str = ""


class UsageRecord(BaseModel):
    """What a case cost to generate."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class Transcript(BaseModel):
    """Everything the agent did for one case — the unit every grader reads.

    `evidence` is what a reply may legitimately be grounded on (tool
    outputs, policy text). `side_effects` is the agent's own snapshot of
    the state it changed, which is what a deterministic oracle checks.
    """

    suite: str
    case_id: str
    replies: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    side_effects: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: UsageRecord = Field(default_factory=UsageRecord)
    error: str = ""

    @property
    def reply(self) -> str:
        """The final reply — what the user was left with."""
        return self.replies[-1] if self.replies else ""


class Score(BaseModel):
    """One grader's judgement of one metric on one case."""

    case_id: str
    metric: str
    grader: str
    kind: GraderKind
    severity: Severity
    value: float
    passed: bool
    threshold: float | None = None
    tier: ScorerTier
    rationale: str = ""
    cost_usd: float = 0.0
    llm_calls: int = 0


class RubricCriterion(BaseModel):
    """One quality dimension a judge scores on a 1..scale_max scale."""

    name: str
    description: str
    threshold: float


class Rubric(BaseModel):
    """What "good" means for one suite, in the agent's own words."""

    suite: str
    scale_max: int
    criteria: list[RubricCriterion]
    hallucination_threshold: float


class QualityMetricSpec(BaseModel):
    """One third-party quality metric (e.g. a DeepEval metric) to run."""

    name: str
    kind: str
    threshold: float
    criteria: str = ""
    evaluation_steps: list[str] = Field(default_factory=list)
    uses_evidence: bool = False


@runtime_checkable
class AgentUnderTest(Protocol):
    """Runs one golden case against a freshly isolated agent."""

    async def run(self, case: GoldenCase) -> Transcript: ...


@runtime_checkable
class Grader(Protocol):
    """Turns a case and its transcript into scores."""

    name: str
    kind: GraderKind
    version: str

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]: ...


@runtime_checkable
class EvalPlugin(Protocol):
    """Everything agent-specific the eval framework needs, behind one seam.

    Named in configuration (`evals.plugin`), exactly as `AgentHost` is for
    the shell, so the platform runs an agent's evals without importing it.
    """

    @property
    def agent_id(self) -> str: ...

    def load_cases(self, suite: str) -> list[GoldenCase]: ...

    def prepare(self) -> None:
        """Build whatever fixtures cases run against (idempotent)."""
        ...

    def agent_under_test(self) -> AgentUnderTest: ...

    def oracle(self) -> Grader: ...

    def judge_rubric(self, suite: str) -> Rubric: ...

    def quality_metrics(self) -> list[QualityMetricSpec]: ...

    def security_config(self) -> str:
        """Path of the agent's Promptfoo config."""
        ...

    def fingerprint_inputs(self) -> dict[str, str]:
        """Agent facts that change what a transcript would be (model id, …)."""
        ...
