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
from agent_platform.logging import get_logger, instrument
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import EventKind

CONTEXT_STAGE = "context_assembly"
_WARN = "warn"
# Removed first, then flagged, then kept: what changed the window leads the card.
_TONE_ORDER = ("fail", "warn", "pass")
CONTEXT_KIND: EventKind = "context"
# A card its emitter lays out itself: titled sections of toned lines.
NARRATIVE_KIND: EventKind = "narrative"


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
        fitted = await self._pipeline.assemble(task, strategies=self._strategies)
        result = await self._checked(fitted, task)
        self._report(result, reporter)
        return result

    def observe_quality(self, session_id: str, score: float) -> None:
        """One answer's quality in [0, 1], for the rot check (nothing without a detector)."""
        if self._health is not None:
            self._health.rot.record(session_id, score)

    async def _checked(self, fitted: AssembledContext, task: Task) -> WindowResult:
        """Diagnose the window and mitigate before the model sees it (when declared)."""
        if self._health is None:
            return WindowResult(window=fitted)
        findings = await self._health.diagnose(fitted, task)
        if findings:
            self._logger.warning(
                "Context window shows failure modes", findings=finding_tree(findings)
            )
        window, findings = self._health.mitigate(fitted, findings)
        self._health.rot.window_used(task.session_id, window.used_tokens)
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
            # A short, coloured card: one line per source and what was done to it. Every
            # statement behind each line is in the context log (`finding_tree`).
            reporter.warning(
                CONTEXT_STAGE,
                event_kind=NARRATIVE_KIND,
                summary=words.unhealthy.format(
                    modes=", ".join(result.modes), action=_acted(result.findings)
                ),
                fields={
                    words.size_title: [words.size_line.format(**cost)],
                    words.findings_title: source_lines(result.findings),
                    words.detail_title: [words.detail_note],
                },
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


def source_lines(findings: list[ContextFinding]) -> list[dict[str, str]]:
    """One toned line per source and what was done to it, worst first.

    `✗ kb:refunds.md#0 · clash · dropped (lower authority) · 2 statement(s)` in red: a source
    removed or cut is red, a flagged one yellow, a kept one green.
    """
    settings = get_settings().context
    words, tones = settings.window, _action_tones()
    grouped: dict[tuple[str, str, str], int] = {}
    for finding in findings:
        key = (finding.source, finding.mode, finding.action)
        grouped[key] = grouped.get(key, 0) + 1
    lines = [
        {
            "text": words.source_line.format(
                mark=words.marks[tones.get(action, _WARN)],
                source=source,
                mode=mode,
                action=action,
                count=count,
            ),
            "tone": tones.get(action, _WARN),
        }
        for (source, mode, action), count in grouped.items()
    ]
    return sorted(lines, key=lambda line: _TONE_ORDER.index(line["tone"]))


def _action_tones() -> dict[str, str]:
    """An action's words → its tone (`context.findings.action_tones`, by action key)."""
    findings = get_settings().context.findings
    return {text: findings.action_tones[key] for key, text in findings.actions.items()}


def finding_tree(findings: list[ContextFinding]) -> dict[str, dict[str, Any]]:
    """Findings as a person reads them: one numbered entry per problem, not one per chunk.

    A clash is found once per side; it is shown once — the statements that disagree (each on
    its own line), how sure the check was, and every source with what was done to it:

        1. clash · statements contradict
           ├─ statements
           │  ├─ 1: A refund applies once an order is settled.
           │  └─ 2: A refund applies once the order is delivered.
           ├─ measured: NLI contradiction 0.99
           └─ sources
              ├─ kb:refunds.md#0: flagged (equal authority)
              └─ kb:faq.md#2: flagged (equal authority)
    """
    words = get_settings().context.window
    labels = words.finding_labels
    grouped: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    headings: dict[tuple[str, str, str, tuple[str, ...]], str] = {}
    for finding in findings:
        key = (finding.mode, finding.rule, finding.detail, tuple(finding.statements))
        if key not in grouped:
            headings[key] = words.finding_heading.format(**finding.model_dump())
            grouped[key] = _entry(finding, labels)
        grouped[key][labels["sources"]][finding.source] = finding.action
    return {
        f"{number}. {headings[key]}": entry
        for number, (key, entry) in enumerate(grouped.items(), start=1)
    }


def _entry(finding: ContextFinding, labels: dict[str, str]) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if finding.statements:
        entry[labels["statements"]] = {
            str(number): text for number, text in enumerate(finding.statements, start=1)
        }
    entry[labels["detail"]] = finding.detail
    entry[labels["sources"]] = {}
    return entry


__all__ = ["CONTEXT_STAGE", "ContextWindow", "WindowResult"]
