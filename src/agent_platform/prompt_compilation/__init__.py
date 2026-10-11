"""Compiled prompts — PARKED (LLD §17).

Interface-only this phase; the contracts below are normative.

The idea (E21): stop hand-writing prompt strings and instead declare a
typed input→output signature, then let an optimizer compile the wording
against a metric and a training set. The wording becomes an artifact
with a version and an eval score rather than a string someone tuned and
nobody can re-derive.

Which makes a compiled module a procedural-memory artifact, not a new
kind of thing — so when this lands, every `CompiledModule` registers
with `memory.SkillStore` (E14) and with the deployment version
registry (E39), rather than growing a third place where versioned
know-how lives.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Example(BaseModel):
    """One training example: inputs and the output they should produce."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None


@runtime_checkable
class Signature(Protocol):
    """A typed input→output contract, independent of wording.

    The point of the indirection: the contract is what callers depend
    on, so the wording underneath can be recompiled without any caller
    changing.
    """

    @property
    def input_fields(self) -> list[str]: ...

    @property
    def output_fields(self) -> list[str]: ...


@runtime_checkable
class EvalMetric(Protocol):
    """Scores a produced output against an expected one."""

    def __call__(self, produced: Any, expected: Any) -> float: ...


@runtime_checkable
class CompiledModule(Protocol):
    """A compiled, versioned, runnable prompt."""

    version: str

    async def run(self, **inputs: Any) -> Any: ...


@runtime_checkable
class Optimizer(Protocol):
    """Builds a compiled module from a signature, a metric and examples."""

    def compile(
        self, signature: Signature, metric: EvalMetric, trainset: list[Example]
    ) -> CompiledModule: ...


__all__ = [
    "CompiledModule",
    "EvalMetric",
    "Example",
    "Optimizer",
    "Signature",
]
