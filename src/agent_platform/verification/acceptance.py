"""Acceptance checks: named functions that decide whether an artifact meets a bar.

A *check spec* says which named check to run, of which kind, with which
parameters and threshold, and whether failing it is disqualifying. The
registry maps names to functions and evaluates a list of specs against a
caller-supplied context. The kinds exist so a caller can run the cheap
ones first and choose where independence matters:

    deterministic   string and number rules; free, exact, always run first
    nli             entailment between a claim and a quotation
    judge           a model's judgement, used only where nothing cheaper can decide

The registry knows nothing about what is being accepted. The context type
is the caller's; so are the check functions.
"""

from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.logging import get_logger

CheckKind = Literal["deterministic", "nli", "judge"]
C = TypeVar("C")

_UNREGISTERED = "no check registered under this name"


class CheckSpec(BaseModel):
    """One check to run, how strict to be, and whether it is disqualifying."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    kind: CheckKind
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    threshold: float
    must_pass: bool = True


class CheckOutcome(BaseModel):
    """What a check function measured."""

    score: float
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)
    # False when there was nothing to check (no claims, no links): the score is then
    # a vacuous 1.0 that must not count as a pass in any aggregate.
    applicable: bool = True


class CheckVerdict(BaseModel):
    """A spec, its outcome, and whether the outcome clears the threshold."""

    item_id: str
    kind: CheckKind
    name: str
    passed: bool
    score: float
    threshold: float
    must_pass: bool
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)
    applicable: bool = True


CheckFunction = Callable[[CheckSpec, C], Awaitable[CheckOutcome]]


class AcceptanceCheckRegistry(Generic[C]):
    """Named checks, evaluated in the order the caller's kinds allow."""

    def __init__(self) -> None:
        self._checks: dict[tuple[CheckKind, str], CheckFunction[C]] = {}
        self._logger = get_logger("verification")

    def register(self, kind: CheckKind, name: str, check: CheckFunction[C]) -> None:
        self._checks[(kind, name)] = check

    def names(self, kind: CheckKind | None = None) -> list[str]:
        return sorted(name for key_kind, name in self._checks if kind in (None, key_kind))

    def has(self, kind: CheckKind, name: str) -> bool:
        return (kind, name) in self._checks

    async def evaluate(
        self, specs: Sequence[CheckSpec], context: C, kinds: Iterable[CheckKind] | None = None
    ) -> list[CheckVerdict]:
        """Every spec of an allowed kind, evaluated; an unknown check fails rather than passes."""
        allowed = set(kinds) if kinds is not None else None
        chosen = [spec for spec in specs if allowed is None or spec.kind in allowed]
        return [await self._one(spec, context) for spec in chosen]

    async def _one(self, spec: CheckSpec, context: C) -> CheckVerdict:
        check = self._checks.get((spec.kind, spec.name))
        outcome = (
            CheckOutcome(score=0.0, detail=_UNREGISTERED)
            if check is None
            else await check(spec, context)
        )
        verdict = CheckVerdict(
            item_id=spec.item_id,
            kind=spec.kind,
            name=spec.name,
            passed=check is not None and outcome.score >= spec.threshold,
            score=outcome.score,
            threshold=spec.threshold,
            must_pass=spec.must_pass,
            detail=outcome.detail,
            evidence=outcome.evidence,
            applicable=outcome.applicable,
        )
        self._logger.debug(
            "Acceptance check", item=spec.item_id, check=spec.name, passed=verdict.passed
        )
        return verdict
