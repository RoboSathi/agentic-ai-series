"""One call per role: route to the role's model, validate, account.

Every model call in this agent goes through `StructuredCaller`, so three
things hold everywhere rather than wherever someone remembered them: the
role is routed to its configured model; the answer is validated against
the role's Pydantic schema by the platform guardrail (repair is mechanical
and bounded, failure is loud); and the tokens spent are charged to the
run's budget and reported to the live feed.

Resilience is not here. Each role's client is already protected by the platform
(`RoleModelRouter` → `ModelChainBuilder`): every model behind its own retry, breaker and
timeout, and backup models when one fails or is slow. This class only asks, validates and
accounts.
"""

import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message
from agent_platform.logging import describe_error, get_logger
from agent_platform.loop_control import BudgetGuard
from agent_platform.observability import StageReporter
from agent_platform.observability.step_card import ModelCall
from agent_platform.presentation.text_tree import value_lines
from agent_platform.resilience import AsyncTimeoutGuard
from agent_platform.skills import AgentInstructions
from research_agent.config.settings import InstructionSettings
from research_agent.domain.budget import TOKENS
from research_agent.presentation.narration import role_title, say, toned
from research_agent.prompts.base import RolePrompt

T = TypeVar("T", bound=BaseModel)

LLM_STAGE = "llm_call"
# The provider's finish reason when the output hit its token cap.
TRUNCATED = "length"
# ...and when it ended normally.
FINISHED = "stop"


class ModelCallTimeoutError(RuntimeError):
    """A model call did not return before its caller's deadline (the direct answer's clock)."""

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
        instructions: AgentInstructions | None = None,
        profile: InstructionSettings | None = None,
    ) -> None:
        self._clients = clients
        # The profile and the role's procedures, for the roles configured to carry them.
        self._instructions = instructions
        self._profile = profile
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

    @property
    def instructions(self) -> AgentInstructions | None:
        """The profile and procedures the roles are prompted with, if any."""
        return self._instructions

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
        result = await self._complete(prompt, schema, max_tokens, quiet=False)
        return self._validated(result, schema)

    async def call_traced(
        self, prompt: RolePrompt, schema: type[T], max_tokens: int | None = None
    ) -> tuple[T, ModelCall]:
        """`call`, for a step that draws the call inside its own card (`StepCard`).

        The call is logged and counted as ever, but draws no LLM CALL card of its own: one
        step, one card. What it contributed — the model that answered, its reasoning, its
        answer — comes back for the step to show.
        """
        started = time.perf_counter()
        result = await self._complete(prompt, schema, max_tokens, quiet=True)
        primary = self._clients.model_key(prompt.role).split(":", maxsplit=1)[-1]
        call = ModelCall.of(prompt.role, result, time.perf_counter() - started, primary=primary)
        return self._validated(result, schema), call

    async def _complete(
        self, prompt: RolePrompt, schema: type[T], max_tokens: int | None, *, quiet: bool
    ) -> CompletionResult:
        client = self._clients.client_for(prompt.role)
        cap = max_tokens if max_tokens is not None else self._role_max_tokens.get(prompt.role)
        result = await self._attempt(client, prompt, schema, cap, quiet)
        for _ in range(self._retries):
            if result.finish_reason != TRUNCATED or cap is None:
                break
            cap = int(cap * self._retry_factor)
            result = await self._attempt(client, prompt, schema, cap, quiet)
        if result.finish_reason == TRUNCATED:
            raise OutputTruncatedError(
                f"'{prompt.role}' output was cut off at its token limit "
                f"({result.usage.completion_tokens} tokens), so the JSON is incomplete"
            )
        return result

    def _validated(self, result: CompletionResult, schema: type[T]) -> T:
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
        quiet: bool = False,
    ) -> CompletionResult:
        """One free-form model call (tool calling allowed), reported and metered like any role.

        Not charged to the research run's budget: the caller keeps its own. `timeout` is the
        caller's deadline (what is left of its own clock), not the model's resilience — a call
        still running when it passes raises `ModelCallTimeoutError`. A `quiet` call is logged
        and metered but draws no card: its caller shows the call on a card of its own.
        """
        cap = max_tokens if max_tokens is not None else self._role_max_tokens.get(role)
        request = CompletionRequest(
            messages=messages, tools=tools, temperature=temperature, max_tokens=cap
        )
        client = self._clients.client_for(role)
        if timeout is None:
            return await self._send(client, role, request, False, quiet=quiet)
        try:
            return await AsyncTimeoutGuard().run(
                lambda: self._send(client, role, request, False, quiet=quiet), timeout
            )
        except TimeoutError as exc:
            raise ModelCallTimeoutError(role, timeout or 0.0) from exc

    async def _attempt(
        self,
        client: LLMClient,
        prompt: RolePrompt,
        schema: type[T],
        cap: int | None,
        quiet: bool = False,
    ) -> CompletionResult:
        """One structured model call, charged to the budget and the role's meter."""
        request = CompletionRequest(
            messages=prompt.messages(schema, self._preamble(prompt.role)),
            json_schema=schema.model_json_schema(),
            max_tokens=cap,
        )
        return await self._send(client, prompt.role, request, True, quiet=quiet)

    def _preamble(self, role: str) -> str:
        """The profile and this role's procedures — or nothing, for a role kept narrow."""
        if self._instructions is None or self._profile is None:
            return ""
        if role not in self._profile.profile_roles:
            return ""
        signal = self._profile.role_signal.format(role=role)
        return self._instructions.compose(frozenset({signal}), "").text

    async def _send(
        self,
        client: LLMClient,
        role: str,
        request: CompletionRequest,
        charge: bool,
        quiet: bool = False,
    ) -> CompletionResult:
        started = time.perf_counter()
        if self._reporter is not None:
            # The live status line names what is being waited for, per role.
            self._reporter.started(f"{LLM_STAGE}_{role}")
        try:
            result = await client.complete(request)
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
            quiet=quiet,
        )
        return result

    def _failed(self, role: str, exc: Exception) -> None:
        """A provider error is a card too, so the feed never just stops."""
        error = describe_error(exc)
        self._logger.error("Model call failed", role=role, error=error)
        if self._reporter is not None:
            self._reporter.emit(
                LLM_STAGE,
                "failed",
                event_kind="narrative",
                summary=say("llm.error", role=role_title(role), error=error),
            )

    def _report(
        self,
        role: str,
        tokens: int,
        model: str,
        elapsed: float = 0.0,
        response: str = "",
        finish_reason: str = "",
        quiet: bool = False,
    ) -> None:
        self._logger.debug("Role call", role=role, tokens=tokens, model=model)
        if self._reporter is not None:
            self._reporter.emit(
                LLM_STAGE,
                "completed",
                quiet=quiet,
                tokens_delta=None if quiet else tokens,
                event_kind="narrative",
                summary=say(
                    "llm.heading",
                    role=role_title(role),
                    tokens=f"{tokens:,}",
                    seconds=f"{elapsed:.1f}",
                ),
                fields=self._card_sections(model, response, finish_reason),
            )

    @staticmethod
    def _card_sections(model: str, response: str, finish_reason: str) -> dict[str, object]:
        sections: dict[str, object] = {
            say("llm.section.model"): [model],
            # A structured answer reads as a tree of its own fields, never as JSON.
            say("llm.section.response"): value_lines(response) if response else [say("llm.empty")],
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
