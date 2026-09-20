"""Generic deterministic execution boundary.

The Executor knows nothing about orders, refunds, users, or any other agent's
business state. It accepts only an ``ApprovedToolCall`` produced after the
Gatekeeper and delegates domain work to an injected agent-owned handler.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_commons.config import get_settings
from agentic_commons.executor.types import ApprovedToolCall, ExecutionAuditEvent, ExecutionOutcome
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.observability import StageReporter
from agentic_commons.observability.types import utc_now
from agentic_commons.resilience.idempotency.guard import IdempotencyGuard
from agentic_commons.tools.types import ToolCall, ToolResult


@runtime_checkable
class ToolExecutionHandler(Protocol):
    """Agent-owned adapter that performs one already-approved domain call."""

    async def execute(self, call: ToolCall) -> Any: ...


@runtime_checkable
class ExecutionAuditHook(Protocol):
    """Optional observer for execution lifecycle events."""

    def record(self, event: ExecutionAuditEvent) -> None: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """The only generic API that crosses into tool execution."""

    async def execute(self, approved: ApprovedToolCall) -> ExecutionOutcome: ...


@instrument("executor")
class ToolExecutorImpl:
    """Reference execution boundary: approved call → one normalized result."""

    def __init__(
        self,
        handler: ToolExecutionHandler,
        idempotency: IdempotencyGuard | None = None,
        audit_hook: ExecutionAuditHook | None = None,
        reporter: StageReporter | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._handler = handler
        self._idempotency = idempotency
        timeout = get_settings().resilience.timeout
        self._timeout_enabled = (
            timeout.enabled if timeout_seconds is None else timeout_seconds > 0
        )
        self._timeout_seconds = (
            timeout.default_seconds if timeout_seconds is None else float(timeout_seconds)
        )
        self._audit_hook = audit_hook
        self._reporter = reporter
        self._agent_id = get_settings().agent.id
        self._logger = get_logger("executor")

    @logged("executor")
    async def execute(self, approved: ApprovedToolCall) -> ExecutionOutcome:
        """Execute one call, with idempotency and timeout boundaries."""
        call = approved.call
        if not approved.decision.allowed:
            raise ValueError("Executor received a tool call that was not approved")
        if approved.decision.action != call.action:
            raise ValueError("ApprovedToolCall action does not match its Gatekeeper decision")

        key = self._idempotency_key(approved)
        started = utc_now()
        start_monotonic = time.monotonic()
        self._emit(
            ExecutionAuditEvent(
                action=call.action,
                actor=approved.authority.actor,
                phase="started",
                idempotency_key=key,
            )
        )
        if self._reporter is not None:
            self._reporter.started(
                "executor",
                event_kind="executor",
                summary=f"Executing approved tool: {call.action}",
                fields={
                    "action": call.action,
                    "actor": approved.authority.actor,
                    "idempotency_key": key,
                    "side_effecting": call.side_effecting,
                },
            )

        if call.side_effecting and self._idempotency is not None and self._idempotency.seen(key):
            recorded = self._idempotency.recorded_result(key)
            if recorded is not None:
                normalized = _normalize_result(recorded)
                outcome = self._build_outcome(
                    normalized,
                    status="replayed",
                    approved=approved,
                    key=key,
                    started=started,
                    start_monotonic=start_monotonic,
                    replayed=True,
                )
                self._emit_outcome(outcome, approved)
                return outcome

        async def run_handler() -> Any:
            return await self._handler.execute(call)

        try:
            raw = (
                await asyncio.wait_for(run_handler(), timeout=self._timeout_seconds)
                if self._timeout_enabled
                else await run_handler()
            )
            normalized = _normalize_result(raw)
            timed_out = False
        except asyncio.TimeoutError:
            normalized = ToolResult(
                success=False,
                error=(
                    f"Tool '{call.action}' exceeded the execution timeout "
                    f"of {self._timeout_seconds:.2f}s"
                ),
            )
            timed_out = True
        except Exception as exc:
            self._logger.exception("Approved tool handler raised", action=call.action)
            normalized = ToolResult(success=False, error=f"Tool '{call.action}' failed: {exc}")
            timed_out = False

        if call.side_effecting and self._idempotency is not None and normalized.success:
            # Remember only successful outcomes. A transient failure or timeout
            # remains retryable rather than becoming a poisoned idempotency key.
            self._idempotency.remember(key, normalized.model_dump(mode="json"))

        outcome = self._build_outcome(
            normalized,
            status="timed_out" if timed_out else "completed" if normalized.success else "failed",
            approved=approved,
            key=key,
            started=started,
            start_monotonic=start_monotonic,
            replayed=False,
            timed_out=timed_out,
        )
        self._emit_outcome(outcome, approved)
        return outcome

    def _build_outcome(
        self,
        result: ToolResult,
        *,
        status: str,
        approved: ApprovedToolCall,
        key: str,
        started: Any,
        start_monotonic: float,
        replayed: bool,
        timed_out: bool = False,
    ) -> ExecutionOutcome:
        ended = utc_now()
        return ExecutionOutcome(
            result=result,
            status=status,  # type: ignore[arg-type]
            replayed=replayed,
            timed_out=timed_out,
            idempotency_key=key,
            duration_ms=(time.monotonic() - start_monotonic) * 1000,
            started_at=started,
            ended_at=ended,
        )

    def _idempotency_key(self, approved: ApprovedToolCall) -> str:
        """Stable semantic key for side effects; read-only calls never use it."""
        if not approved.call.side_effecting:
            return ""
        payload = {
            "agent": self._agent_id,
            "actor": approved.authority.actor,
            "action": approved.call.action,
            "params": approved.call.params,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        return f"exec:{approved.call.action}:{digest}"

    def _emit(self, event: ExecutionAuditEvent) -> None:
        if self._audit_hook is not None:
            self._audit_hook.record(event)
        self._logger.info(
            "Tool execution lifecycle",
            phase=event.phase,
            action=event.action,
            actor=event.actor,
            status=event.status or "started",
            idempotency_key=event.idempotency_key,
            replayed=event.replayed,
            timed_out=event.timed_out,
            duration_ms=round(event.duration_ms, 2),
            error=event.error or "",
            result_success=event.result.success,
            result=event.result.output or event.result.error or "",
        )

    def _emit_outcome(self, outcome: ExecutionOutcome, approved: ApprovedToolCall) -> None:
        event = ExecutionAuditEvent(
            action=approved.call.action,
            actor=approved.authority.actor,
            phase="finished",
            status=outcome.status,
            idempotency_key=outcome.idempotency_key,
            replayed=outcome.replayed,
            timed_out=outcome.timed_out,
            duration_ms=outcome.duration_ms,
            result=outcome.result,
            error=outcome.result.error,
        )
        self._emit(event)
        if self._reporter is not None:
            method = self._reporter.completed if outcome.result.success else self._reporter.failed
            method(
                "executor",
                event_kind="executor",
                summary=(
                    f"Tool execution {'replayed' if outcome.replayed else 'completed'}: {approved.call.action}"
                    if outcome.result.success
                    else f"Tool execution failed: {approved.call.action}"
                ),
                fields={
                    "action": approved.call.action,
                    "actor": approved.authority.actor,
                    "execution_status": outcome.status,
                    "replayed": outcome.replayed,
                    "timed_out": outcome.timed_out,
                    "duration_ms": round(outcome.duration_ms, 2),
                    "idempotency_key": outcome.idempotency_key,
                    "result": "success" if outcome.result.success else "failed",
                    "error": outcome.result.error or "",
                },
            )


def _normalize_result(value: Any) -> ToolResult:
    """Normalize any handler return into the one result contract."""
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, BaseModel):
        return ToolResult(
            success=True,
            output=str(value),
            data=value.model_dump(mode="json"),
        )
    if isinstance(value, dict):
        if "success" in value or "output" in value or "error" in value:
            return ToolResult.model_validate(value)
        return ToolResult(success=True, data=value, output=json.dumps(value, default=str))
    if value is None:
        return ToolResult(success=True, output="")
    return ToolResult(success=True, output=str(value))
