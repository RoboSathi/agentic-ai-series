"""One call per role: route to the role's model, validate, account.

Every model call in this agent goes through `StructuredCaller`, so three
things hold everywhere rather than wherever someone remembered them: the
role is routed to its configured model; the answer is validated against
the role's Pydantic schema by the platform guardrail (repair is mechanical
and bounded, failure is loud); and the tokens spent are charged to the
run's budget and reported to the live feed.
"""

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from agentic_commons.control_loop import BudgetGuard
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest
from agentic_commons.logging import get_logger
from agentic_commons.observability import StageReporter
from research_agent.domain.budget import TOKENS
from research_agent.prompts.base import RolePrompt

T = TypeVar("T", bound=BaseModel)

LLM_STAGE = "llm_call"


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
    ) -> None:
        self._clients = clients
        self._validator = validator
        self._reporter = reporter
        self._budget: BudgetGuard | None = None
        self._logger = get_logger("research_agent")

    def bind_budget(self, budget: BudgetGuard | None) -> None:
        """Charge subsequent calls to this run's budget (None: scoping, no run yet)."""
        self._budget = budget

    def model_key(self, role: str) -> str:
        return self._clients.model_key(role)

    async def call(self, prompt: RolePrompt, schema: type[T]) -> T:
        """Ask the role's model, return the validated answer, charge its tokens."""
        client = self._clients.client_for(prompt.role)
        result = await client.complete(
            CompletionRequest(
                messages=prompt.messages(schema), json_schema=schema.model_json_schema()
            )
        )
        tokens = result.usage.total_tokens
        if self._budget is not None:
            self._budget.charge(TOKENS, tokens)
        self._report(prompt.role, tokens, result.model or self._clients.model_key(prompt.role))
        return schema.model_validate(self._validator.validate_or_raise(result.content, schema))

    def _report(self, role: str, tokens: int, model: str) -> None:
        self._logger.debug("Role call", role=role, tokens=tokens, model=model)
        if self._reporter is not None:
            self._reporter.completed(
                LLM_STAGE,
                tokens_delta=tokens,
                summary=f"{role} · {tokens} tokens",
                fields={"role": role, "model": model, "total_tokens": tokens},
            )
