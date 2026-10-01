"""One call per role: route to the role's model, validate, account.

Every model call in this agent goes through `StructuredCaller`, so three
things hold everywhere rather than wherever someone remembered them: the
role is routed to its configured model; the answer is validated against
the role's Pydantic schema by the platform guardrail (repair is mechanical
and bounded, failure is loud); and the tokens spent are charged to the
run's budget and reported to the live feed.
"""

import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from agentic_commons.control_loop import BudgetGuard
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest, CompletionResult, Message
from agentic_commons.logging import get_logger
from agentic_commons.observability import StageReporter
from agentic_commons.resilience import ResilientCaller
from research_agent.domain.budget import TOKENS
from research_agent.narration import say, title_of, toned
from research_agent.prompts.base import RolePrompt

T = TypeVar("T", bound=BaseModel)

LLM_STAGE = "llm_call"
# The dependency name of a role's resilience stack (retry with backoff, circuit breaker, bulkhead).
RESILIENCE_PREFIX = "research_llm_"
# The provider's finish reason when the output hit its token cap.
TRUNCATED = "length"
# ...and when it ended normally.
FINISHED = "stop"


class ModelCallTimeoutError(RuntimeError):
    """A model call did not return within its own time limit."""

    def __init__(self, role: str, seconds: float) -> None:
        super().__init__(f"'{role}' did not respond within {seconds:.0f}s")
        self.role = role
        self.seconds = seconds


class OutputTruncatedError(RuntimeError):
    """A model's answer stopped at its token limit, so it cannot be valid structured output."""


@runtime_checkable
class RoleClients(Protocol):
    """Resolves a role name to the client that plays it (`RoleModelRouter`)."""

    def client_for(self, role: str) -> LLMClient: ...

    def model_key(self, role: str) -> str: ...


class StructuredCaller:
    """Validated, accounted model calls, one per role."""

    def __init__(
        self,
        clients: RoleClients,
        validator: PydanticValidator,
        reporter: StageReporter | None = None,
        role_max_tokens: Mapping[str, int] | None = None,
        truncation_retries: int = 0,
        truncation_retry_factor: float = 2.0,
        role_timeout_seconds: Mapping[str, float] | None = None,
        timeout_retries: int = 0,
    ) -> None:
        self._clients = clients
        self._role_timeouts = dict(role_timeout_seconds or {})
        self._timeout_retries = timeout_retries
        self._resilient: dict[str, ResilientCaller] = {}
        self._role_max_tokens = dict(role_max_tokens or {})
        self._retries = truncation_retries
        self._retry_factor = truncation_retry_factor
        self._validator = validator
        self._reporter = reporter
        self._budget: BudgetGuard | None = None
        self._usage: dict[str, dict[str, float]] = defaultdict(
            lambda: {"tokens": 0.0, "seconds": 0.0, "calls": 0.0}
        )
        self._logger = get_logger("research_agent")

    def usage(self) -> dict[str, dict[str, float]]:
        """Tokens, model seconds and calls per role since this caller was built."""
        return {role: dict(values) for role, values in self._usage.items()}

    def bind_budget(self, budget: BudgetGuard | None) -> None:
        """Charge subsequent calls to this run's budget (None: scoping, no run yet)."""
        self._budget = budget

    def model_key(self, role: str) -> str:
        return self._clients.model_key(role)

    async def call(self, prompt: RolePrompt, schema: type[T], max_tokens: int | None = None) -> T:
        """Ask the role's model, return the validated answer, charge its tokens.

        `max_tokens` caps this call's output; unset, the role's configured cap applies, and
        failing that the provider's default. A call cut off at its cap is retried with a larger
        one (the reasoning model may have spent the cap thinking) before it is given up on.
        """
        client = self._clients.client_for(prompt.role)
        cap = max_tokens if max_tokens is not None else self._role_max_tokens.get(prompt.role)
        result = await self._attempt(client, prompt, schema, cap)
        for _ in range(self._retries):
            if result.finish_reason != TRUNCATED or cap is None:
                break
            cap = int(cap * self._retry_factor)
            result = await self._attempt(client, prompt, schema, cap)
        if result.finish_reason == TRUNCATED:
            raise OutputTruncatedError(
                f"'{prompt.role}' output was cut off at its token limit "
                f"({result.usage.completion_tokens} tokens), so the JSON is incomplete"
            )
        return schema.model_validate(self._validator.validate_or_raise(result.content, schema))

    async def chat(
        self,
        role: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> CompletionResult:
        """One free-form model call (tool calling allowed), reported and metered like any role.

        Not charged to the research run's budget: the caller keeps its own. A call that does not
        return within `timeout` seconds raises `ModelCallTimeoutError`.
        """
        cap = max_tokens if max_tokens is not None else self._role_max_tokens.get(role)
        request = CompletionRequest(
            messages=messages, tools=tools, temperature=temperature, max_tokens=cap
        )
        try:
            return await self._send(self._clients.client_for(role), role, request, timeout, False)
        except ModelCallTimeoutError as exc:
            self._stalled(role, exc.seconds, final=True)
            raise

    async def _attempt(
        self, client: LLMClient, prompt: RolePrompt, schema: type[T], cap: int | None
    ) -> CompletionResult:
        """One structured model call, charged to the budget and the role's meter.

        A call that does not return in its role's time is cancelled and tried again (it is a
        pure read), up to `timeout_retries` times; each stall is a card on the feed.
        """
        request = CompletionRequest(
            messages=prompt.messages(schema),
            json_schema=schema.model_json_schema(),
            max_tokens=cap,
        )
        timeout = self._role_timeouts.get(prompt.role, self._role_timeouts.get("default"))
        for attempt in range(self._timeout_retries + 1):
            try:
                return await self._send(client, prompt.role, request, timeout, True)
            except ModelCallTimeoutError as exc:
                final = attempt == self._timeout_retries
                self._stalled(prompt.role, exc.seconds, final)
                if final:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _complete(
        self, client: LLMClient, role: str, request: CompletionRequest, timeout: float | None
    ) -> CompletionResult:
        """The call itself: through the role's retry and breaker when it has a time limit.

        A provider error (a 500, a dropped connection) is retried with backoff inside the
        role's time limit; the limit still bounds everything, so a stall ends where it always did.
        """
        if timeout is None:
            return await client.complete(request)
        stack = self._resilient.setdefault(role, ResilientCaller(f"{RESILIENCE_PREFIX}{role}"))
        return await stack.call(
            lambda: client.complete(request),
            idempotency_key=f"{role}:{uuid.uuid4().hex}",
            timeout_seconds=timeout,
        )

    def _stalled(self, role: str, seconds: float, final: bool) -> None:
        key = "llm.timeout_final" if final else "llm.timeout_retry"
        summary = say(key, role=title_of(role), seconds=f"{seconds:.0f}")
        self._logger.warning("Model call stalled", role=role, seconds=seconds, final=final)
        if self._reporter is not None:
            self._reporter.emit(LLM_STAGE, "failed", event_kind="narrative", summary=summary)

    async def _send(
        self,
        client: LLMClient,
        role: str,
        request: CompletionRequest,
        timeout: float | None,
        charge: bool,
    ) -> CompletionResult:
        started = time.perf_counter()
        if self._reporter is not None:
            # The live status line names what is being waited for, per role.
            self._reporter.started(f"{LLM_STAGE}_{role}")
        try:
            async with asyncio.timeout(timeout):
                result = await self._complete(client, role, request, timeout)
        except TimeoutError as exc:
            raise ModelCallTimeoutError(role, timeout or 0.0) from exc
        except Exception as exc:
            self._failed(role, exc)
            raise
        tokens = result.usage.total_tokens
        if charge and self._budget is not None:
            self._budget.charge(TOKENS, tokens)
        elapsed = time.perf_counter() - started
        meter = self._usage[role]
        meter["tokens"] += tokens
        meter["seconds"] += elapsed
        meter["calls"] += 1
        self._report(
            role,
            tokens,
            result.model or self._clients.model_key(role),
            elapsed,
            _shown(result),
            result.finish_reason or "",
        )
        return result

    def _failed(self, role: str, exc: Exception) -> None:
        """A provider error is a card too, so the feed never just stops."""
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        self._logger.error("Model call failed", role=role, error=error)
        if self._reporter is not None:
            self._reporter.emit(
                LLM_STAGE,
                "failed",
                event_kind="narrative",
                summary=say("llm.error", role=title_of(role), error=error),
            )

    def _report(
        self,
        role: str,
        tokens: int,
        model: str,
        elapsed: float = 0.0,
        response: str = "",
        finish_reason: str = "",
    ) -> None:
        self._logger.debug("Role call", role=role, tokens=tokens, model=model)
        if self._reporter is not None:
            self._reporter.emit(
                LLM_STAGE,
                "completed",
                tokens_delta=tokens,
                event_kind="narrative",
                summary=say(
                    "llm.heading",
                    role=title_of(role),
                    tokens=f"{tokens:,}",
                    seconds=f"{elapsed:.1f}",
                ),
                fields=self._card_sections(model, response, finish_reason),
            )

    @staticmethod
    def _card_sections(model: str, response: str, finish_reason: str) -> dict[str, object]:
        sections: dict[str, object] = {
            say("llm.section.model"): [model],
            say("llm.section.response"): response.splitlines() or [say("llm.empty")],
        }
        if finish_reason == TRUNCATED:
            sections[say("llm.section.finish")] = [toned(say("llm.cut_off"), "fail")]
        elif finish_reason and finish_reason != FINISHED:
            sections[say("llm.section.finish")] = [finish_reason]
        return sections


def _shown(result: CompletionResult) -> str:
    """What a model call's card shows: its text, then each tool call it asked for."""
    calls = [f"-> {call.name}({call.arguments})" for call in result.tool_calls or []]
    return "\n".join(part for part in (result.content, *calls) if part)
