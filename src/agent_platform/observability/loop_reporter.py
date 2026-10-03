"""Cards for one iteration loop, numbered by step — shared by every agent.

A loop that calls a model, acts on what it asked for and reads the result
emits several cards per iteration. Numbering them ("Step 2 · …") is what
makes "the model asked twice" visible, and saying the cap beside the number
("2 of 6") is what makes a loop that ran out visible. The wording is
configuration, so it cannot drift from what the loop does.

`mark_step` is the rule; `LoopReporter` applies it to every card a loop
emits through a `StageReporter`. An agent whose loop has its own card
helpers calls `mark_step`; one starting fresh holds a `LoopReporter`.
"""

from collections.abc import Collection, Mapping
from typing import Any

from agent_platform.config import get_settings
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import EventKind, StatusKind

# The field a numbered card carries its "step N of M" in.
LOOP_STEP_FIELD = "loop_step"


def mark_step(
    stage: str,
    summary: str | None,
    fields: Mapping[str, Any] | None,
    *,
    step: int | None,
    max_steps: int,
    stages: Collection[str],
    detail: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """A card's summary and fields, marked as part of loop step `step` when its stage is."""
    if step is None or stage not in stages:
        return summary, None if fields is None else dict(fields)
    settings = get_settings().observability
    prefix = settings.loop_step_prefix.format(step=step)
    template = detail or settings.loop_step_detail
    return (prefix + summary if summary else prefix.rstrip(" ·")), {
        LOOP_STEP_FIELD: template.format(step=step, max=max_steps),
        **(fields or {}),
    }


class LoopReporter:
    """Emits a loop's cards, each numbered by the step it belongs to.

    `detail` is the agent's description of one iteration ("{step} of {max}
    (thought → action → observation)"); the platform's is the default.
    """

    def __init__(
        self,
        reporter: StageReporter | None,
        *,
        max_steps: int,
        stages: Collection[str],
        detail: str | None = None,
    ) -> None:
        self._reporter = reporter
        self._max_steps = max_steps
        self._stages = frozenset(stages)
        self._detail = detail
        self.step: int | None = None

    def begin(self, step: int) -> None:
        """Number every loop card that follows with `step`."""
        self.step = step

    def emit(
        self,
        stage: str,
        status: StatusKind = "completed",
        *,
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
        event_kind: EventKind = "stage",
        tokens: int | None = None,
        group: str = "",
        quiet: bool = False,
    ) -> None:
        """One card, numbered when its stage belongs to the loop."""
        if self._reporter is None:
            return
        summary, marked = mark_step(
            stage,
            summary,
            fields,
            step=self.step,
            max_steps=self._max_steps,
            stages=self._stages,
            detail=self._detail,
        )
        self._reporter.emit(
            stage,
            status,
            tokens_delta=tokens,
            event_kind=event_kind,
            summary=summary,
            fields=marked,
            group=group,
            quiet=quiet,
        )


__all__ = ["LOOP_STEP_FIELD", "LoopReporter", "mark_step"]
