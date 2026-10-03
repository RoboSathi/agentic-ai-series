"""The window a model is handed, in one call: assemble, fit, check its health, report it.

Every agent that hands a model retrieved material does the same four things — gather
candidates through the five-stage pipeline, let the strategies fit them to the budget,
diagnose and mitigate failure modes, and say what happened on a `context_assembly` card.
They live here once, so every agent draws the same card from the same logic, and only the
*sources* differ per agent.

A withheld piece is never built: no strategies means the pipeline's own fit is final; no
detector means no health check and no findings.
"""

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.context.failure import ContextFailureDetector
from agent_platform.context.pipeline import ContextAssembler
from agent_platform.context.strategies import StrategySelector
from agent_platform.context.types import AssembledContext, ContextFinding, Task
from agent_platform.llm.embeddings import tokenize
from agent_platform.logging import get_logger, instrument
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import EventKind

CONTEXT_STAGE = "context_assembly"
CONTEXT_KIND: EventKind = "context"


class WindowResult(BaseModel):
    """The fitted, checked window, and every failure the check found in it."""

    window: AssembledContext
    findings: list[ContextFinding] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return self.window.text

    @property
    def modes(self) -> list[str]:
        return sorted({finding.mode for finding in self.findings})


@instrument("context")
class ContextWindow:
    """Assemble → fit → health check → card, for any agent's sources."""

    def __init__(
        self,
        pipeline: ContextAssembler,
        strategies: StrategySelector | None = None,
        health: ContextFailureDetector | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._strategies = strategies
        self._health = health
        self._logger = get_logger("context")

    async def build(self, task: Task, reporter: StageReporter | None = None) -> WindowResult:
        words = get_settings().context.window
        if reporter is not None:
            reporter.started(CONTEXT_STAGE, event_kind=CONTEXT_KIND, summary=words.started)
        assembled = await self._pipeline.assemble(task)
        fitted = self._strategies.fit(assembled) if self._strategies is not None else assembled
        result = self._checked(fitted, task)
        self._report(result, reporter)
        return result

    def _checked(self, fitted: AssembledContext, task: Task) -> WindowResult:
        """Diagnose the window and mitigate before the model sees it (when declared)."""
        if self._health is None:
            return WindowResult(window=fitted)
        findings = self._health.diagnose(fitted, set(tokenize(task.query)))
        if findings:
            self._logger.warning(
                "Context window shows failure modes",
                findings=[finding.model_dump() for finding in findings],
            )
        window, findings = self._health.mitigate(fitted, findings)
        return WindowResult(window=window, findings=findings)

    def _report(self, result: WindowResult, reporter: StageReporter | None) -> None:
        """A warning card when the window needed attention; a quiet record otherwise."""
        if reporter is None:
            return
        words = get_settings().context.window
        window = result.window
        cost: dict[str, Any] = {
            "chunks": len(window.chunks),
            "dropped": len(window.dropped),
            "used_tokens": window.used_tokens,
            "budget_tokens": window.budget_tokens,
        }
        if result.modes and not _routine(result.findings):
            reporter.warning(
                CONTEXT_STAGE,
                event_kind=CONTEXT_KIND,
                summary=words.unhealthy.format(
                    modes=", ".join(result.modes), action=_acted(result.findings)
                ),
                fields={"findings": [_finding_line(f) for f in result.findings], **cost},
            )
            return
        reporter.emit(
            CONTEXT_STAGE,
            "completed",
            event_kind=CONTEXT_KIND,
            summary=words.assembled,
            fields=cost,
            quiet=words.quiet_when_healthy,
        )


def _routine(findings: list[ContextFinding]) -> bool:
    """Whether every finding is ordinary retrieval noise that was already removed.

    Retrieval over-fetches on purpose and the health check trims what is off the task — that
    is the pipeline working, not a fault. Poisoning, a clash between sources, rot, or anything
    left in the window still warrants a warning.
    """
    words = get_settings().context.window
    return all(
        finding.mode in words.routine_modes and finding.action != words.not_acted_on
        for finding in findings
    )


def _acted(findings: list[ContextFinding]) -> str:
    """What mitigation did, in a few words."""
    words = get_settings().context.window
    dropped = sum(1 for finding in findings if finding.action.startswith(words.dropped_action))
    return words.removed.format(count=dropped) if dropped else words.nothing_removed


def _finding_line(finding: ContextFinding) -> str:
    return get_settings().context.window.finding_line.format(**finding.model_dump())


__all__ = ["CONTEXT_STAGE", "ContextWindow", "WindowResult"]
