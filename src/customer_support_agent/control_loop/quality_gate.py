"""The Output Quality Gate: plan, critique, score (E06, FR-16).

Not a loop, and no longer named as one. What happens here is one pass: a
plan is looked up, a draft is critiqued once, at most one repair is
attempted, and the result is scored by something independent. The
iteration in this agent lives elsewhere — the Tool-Use Loop in the graph.

Three roles, three Protocols, three swappable implementations:

- a **Planner** states the ordered steps *before* acting, so the work is
  a committed plan rather than whatever the model reaches for next. This
  one is a config lookup, not a decomposition: the order tools must run
  in is a property of the domain, not a judgement call;
- a **Reflector** critiques a draft reply after the work, against what
  the tools actually returned;
- an **Evaluator** scores the reply with something independent of the
  thing that produced it.

They are separate objects rather than three methods on the agent because
the pairing is the point. A generator that grades its own output is a
generator with an opinion: the evaluator has to be able to come from
somewhere else — a different model, a deterministic rule, a human — and
that is only true if it is injected.

None of these calls a model. The planner reads the classified intent,
the reflector applies deterministic checks, and the evaluator uses the
NOLLM cascade, whose cheap tiers are exact assertions and embeddings.
A deliberation loop that costs three completions per turn to improve one
reply is a loop that will be switched off in production.
"""

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.evals import Claim, NollmScoringCascadeImpl
from agent_platform.logging import get_logger, instrument
from customer_support_agent.config import get_agent_settings


class PlanStep(BaseModel):
    """One step of a plan, and the tool it expects to need."""

    description: str
    tool: str = ""


class Plan(BaseModel):
    """An ordered decomposition, committed before execution begins."""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def is_single_step(self) -> bool:
        """Whether this needs no decomposition at all.

        Most support requests are one step. Saying so explicitly is what
        stops the loop dressing "where is my order" up as a project.
        """
        return len(self.steps) <= 1


class Critique(BaseModel):
    """What a reflector thinks is wrong with a draft, if anything."""

    acceptable: bool = True
    problems: list[str] = Field(default_factory=list)
    # Which checks produced this verdict, so a report can say what was looked at.
    checks: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(self.problems)


@runtime_checkable
class Planner(Protocol):
    """Decomposes a request into ordered steps."""

    def plan(self, query: str, intent: str, allowed_tools: list[str]) -> Plan: ...


@runtime_checkable
class Reflector(Protocol):
    """Critiques a draft reply against the evidence behind it."""

    def reflect(self, draft: str, evidence: list[str]) -> Critique: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a reply with something independent of what produced it."""

    async def evaluate(self, reply: str, evidence: list[str]) -> float: ...


@instrument("customer_support_agent")
class PlanLookup:
    """Plans from the classified intent, deterministically.

    The intent already decided which tools this turn may use, and the
    order those tools must run in is a property of the domain, not a
    judgement call: eligibility is checked before a refund is issued,
    every time, whatever a model would have chosen. So the plan is a
    lookup, not a completion.
    """

    def __init__(self) -> None:
        self._templates = get_agent_settings().plans
        self._logger = get_logger("customer_support_agent")

    def plan(self, query: str, intent: str, allowed_tools: list[str]) -> Plan:
        """The ordered steps this intent implies, filtered to what is allowed."""
        template = self._templates.get(intent, [])
        steps = [
            PlanStep(description=step.description, tool=step.tool)
            for step in template
            if not step.tool or step.tool in allowed_tools
        ]
        plan = Plan(goal=query, steps=steps)
        self._logger.debug("Planned the turn", intent=intent, steps=len(plan.steps))
        return plan


# The rules the draft check applies, in order, as a report names them.
_DRAFT_CHECKS = ("empty", "deliberation", "order_ids")


@instrument("customer_support_agent")
class GroundedCritic:
    """Critiques a draft against the tool results behind it.

    Deterministic checks only, and each one is a mistake this agent has
    actually made: replying with the model's deliberation instead of an
    answer, quoting an order id no tool returned, or answering at all
    when no tool succeeded.
    """

    def __init__(self) -> None:
        settings = get_agent_settings()
        self._markers = [marker.casefold() for marker in settings.deliberation_markers]
        # An order id, in this agent's id shape. The follow-up pattern
        # (any run of four or more digits) read a year in "August 8, 2026"
        # as an order no tool had returned.
        self._reference = re.compile(settings.order_reference.id_shape)
        self._logger = get_logger("customer_support_agent")

    def reflect(self, draft: str, evidence: list[str]) -> Critique:
        """What is wrong with this draft, if anything."""
        problems: list[str] = []
        lowered = draft.casefold()
        if not draft.strip():
            problems.append("the draft is empty")
        if any(marker in lowered for marker in self._markers):
            problems.append("the draft reads as deliberation rather than a reply")
        for identifier in set(self._reference.findall(draft)):
            if not any(identifier in text for text in evidence):
                problems.append(f"order {identifier} appears in the reply but in no record")
        if problems:
            self._logger.info("Draft check failed", problems=problems)
        return Critique(acceptable=not problems, problems=problems, checks=list(_DRAFT_CHECKS))


@instrument("customer_support_agent")
class IndependentScorer:
    """Scores a reply with the NOLLM cascade — independent of the model.

    Independent is the whole requirement. Asking the model that wrote the
    reply whether the reply is good returns an opinion correlated with
    the thing being measured.
    """

    def __init__(self, cascade: NollmScoringCascadeImpl) -> None:
        self._cascade = cascade
        self._logger = get_logger("customer_support_agent")

    async def evaluate(self, reply: str, evidence: list[str]) -> float:
        """How well the evidence supports the reply, in [0, 1]."""
        if not reply.strip() or not evidence:
            return 0.0
        verdict = await self._cascade.resolve(Claim(text=reply, evidence=evidence))
        return verdict.score if verdict.label == "supported" else 0.0


@instrument("customer_support_agent")
class OutputQualityGate:
    """Plan, then reflect on the result, then score it.

    Composed of the three roles rather than implementing them, so any one
    can be replaced without touching the other two — a hosted planner, a
    model-backed reflector, a human evaluator.

    Bounded by construction: it critiques once. An unbounded revise loop
    is the generate-test-debug cycle with no exit condition, which is how
    an agent spends a customer's afternoon improving a sentence — and is
    why this is a gate rather than a loop.
    """

    def __init__(
        self,
        planner: Planner | None = None,
        reflector: Reflector | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self._planner = planner or PlanLookup()
        self._reflector = reflector or GroundedCritic()
        self._evaluator = evaluator
        self._logger = get_logger("customer_support_agent")

    def plan(self, query: str, intent: str, allowed_tools: list[str]) -> Plan:
        """The plan for this turn."""
        return self._planner.plan(query, intent, allowed_tools)

    def reflect(self, draft: str, evidence: list[str]) -> Critique:
        """The critique of a draft reply."""
        return self._reflector.reflect(draft, evidence)

    async def score(self, reply: str, evidence: list[str]) -> float:
        """The independent score, or 1.0 when no evaluator is wired.

        Not 0.0: an absent evaluator means "unmeasured", and treating
        unmeasured as failing would make every reply look wrong the
        moment the evaluator is switched off.
        """
        if self._evaluator is None:
            return 1.0
        return await self._evaluator.evaluate(reply, evidence)
