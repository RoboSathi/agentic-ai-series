"""The eval framework's vocabulary, as types (LLD §18).

These are the nouns every stage speaks — a golden case, the transcript
of the agent attempting it, the scores graders give that transcript —
and the Protocols an agent implements to take part. The platform owns
the shapes; an agent owns what goes inside the opaque `payload` and
`side_effects` fields, which is how a golden case can say "refund order
90001" without this package ever learning what an order is.
"""

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Re-exported: the helper moved to the neutral `agent_platform.primitives.hashing`.
from agent_platform.primitives.hashing import canonical_hash as canonical_hash
from agent_platform.scoring.types import ScorerTier

GraderKind = Literal["oracle", "deepeval", "promptfoo"]
Severity = Literal["hard", "soft"]
StageName = Literal["generate", "oracle", "deepeval", "promptfoo"]
# Who decided what a case's expected outcome is.
LabelBasis = Literal["written_policy", "expert_review", "production_incident"]


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
    # Which of the case's turns (0-based) the call was made in.
    turn: int = 0
    status: str = ""
    decision: str = ""
    # The standing the decision gave the action (VOID, RATIFIABLE, VOIDABLE, VALID, …).
    standing: str = ""
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


class RubricBand(BaseModel):
    """One anchored band of a G-Eval's 0–10 scale: what a score in it looks like."""

    score_range: tuple[int, int]
    expected_outcome: str


class QualityMetricSpec(BaseModel):
    """One DeepEval metric to run over the golden transcripts.

    `kind` names the DeepEval metric (`conversational_geval`,
    `turn_faithfulness`, `turn_relevancy`). A G-Eval grades one dimension:
    its `criteria`, the `evaluation_steps` a judge follows, and the `rubric`
    bands that say what each score means — an unanchored scale is one a
    judge cannot apply the same way twice.
    """

    name: str
    kind: str
    threshold: float
    criteria: str = ""
    evaluation_steps: list[str] = Field(default_factory=list)
    rubric: list[RubricBand] = Field(default_factory=list)
    # What a G-Eval judge is shown (DeepEval `MultiTurnParams` values); empty means
    # `evals.deepeval.geval_params`.
    params: list[str] = Field(default_factory=list)
    # Run by default; the rest run only in a full run (`evals.core`).
    core: bool = False


class ConversationFrame(BaseModel):
    """What a DeepEval conversation needs from the agent beyond the transcript.

    `chatbot_role` is the agent's job in one sentence; `scenario` and
    `expected_outcome` come from the case, so a judge grades against the
    outcome the written policy requires, not against its own idea of good.
    """

    chatbot_role: str
    scenario: str = ""
    expected_outcome: str = ""


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
class BatchGrader(Grader, Protocol):
    """A grader that scores many transcripts in one call (a framework with its own loop).

    `batch_size` transcripts are handed over at a time; the stage saves each batch's scores
    as it finishes, so progress moves and an interrupted run resumes from the last batch.
    """

    batch_size: int

    async def grade_batch(
        self, items: list[tuple[GoldenCase, Transcript]]
    ) -> list[list[Score]]: ...


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

    def quality_metrics(self) -> list[QualityMetricSpec]: ...

    def conversation_frame(self, case: GoldenCase) -> ConversationFrame:
        """The role, scenario and expected outcome DeepEval grades a case against."""
        ...

    def security_config(self) -> str:
        """Path of the agent's Promptfoo config."""
        ...

    def fingerprint_inputs(self) -> dict[str, str]:
        """Agent facts that change what a transcript would be (model id, seed, …)."""
        ...


# -- optional plugin hooks -----------------------------------------------------
# Checked with `isinstance`; a plugin without them simply has no such stage.


@runtime_checkable
class IntentBakeoffPlugin(Protocol):
    """A plugin that can score its intent classifiers side by side (`make intent-eval <agent>`)."""

    async def intent_bakeoff(self) -> Any:
        """An `agent_platform.evals.intent.BakeoffResult`."""
        ...
