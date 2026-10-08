"""One tool call, the governed way: validated, authorized, executed, audited.

The MCP server's tools are ordinary `ToolDefinition`s, so they take the same
path as any agent's: the registry validates the arguments; the Gatekeeper rules
on the call against this agent's policy (and writes the audit entry); the
Executor runs it under its deadline and records the fact. With the Gatekeeper
withheld the call still goes through the Executor, and the audit entry says
UNGOVERNED — an execution nobody ruled on is still on the record.
"""

from pydantic import BaseModel, ValidationError

from agent_platform.audit import STANDING_FIELD, AuditEntry, AuditLedger
from agent_platform.executor import ApprovedToolCall, ToolExecutor
from agent_platform.gatekeeper import Gatekeeper
from agent_platform.gatekeeper.types import AuthorityContext
from agent_platform.llm.types import ModelToolCall
from agent_platform.logging import get_current_trace_id, get_logger
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.types import ToolCall, ToolResult
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord


class CallOutcome(BaseModel):
    """What one call told the model, and the query it answered, if it was one."""

    text: str
    succeeded: bool
    refused: bool = False
    query: QueryRecord | None = None


class GovernedCalls:
    """Runs the model's tool calls through registry, Gatekeeper and Executor."""

    def __init__(
        self,
        registry: NamespacedToolRegistry,
        executor: ToolExecutor,
        audit: AuditLedger,
        settings: OpsAgentSettings,
        actor: str,
        gatekeeper: Gatekeeper | None = None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._audit = audit
        self._settings = settings
        self._actor = actor
        self._gatekeeper = gatekeeper
        self._logger = get_logger("ops_agent")

    async def run(self, call: ModelToolCall, session_id: str) -> CallOutcome:
        built = self._registry.build_call(call.name, call.arguments)
        if built is None:
            text = self._settings.words.invalid_call.format(tool=call.name)
            return CallOutcome(text=text, succeeded=False)
        authority = AuthorityContext(
            actor=self._actor,
            session_id=session_id,
            offered_actions=[tool.qualified_name for tool in self._registry.all_tools()],
        )
        approved = await self._approve(built, authority)
        if isinstance(approved, CallOutcome):
            return approved
        outcome = await self._executor.execute(approved)
        return _outcome(outcome.result)

    async def _approve(
        self, call: ToolCall, authority: AuthorityContext
    ) -> ApprovedToolCall | CallOutcome:
        if self._gatekeeper is None:
            return self._ungoverned(call, authority)
        decision = await self._gatekeeper.evaluate(call, authority)
        if not decision.allowed:
            text = self._settings.words.refused_tool.format(
                action=call.action, reason=decision.reason
            )
            return CallOutcome(text=text, succeeded=False, refused=True)
        return ApprovedToolCall.from_decision(call, decision, authority)

    def _ungoverned(self, call: ToolCall, authority: AuthorityContext) -> ApprovedToolCall:
        approved = ApprovedToolCall.ungoverned(call, authority)
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or "",
                actor=authority.actor,
                action=call.action,
                authority_context={
                    **authority.model_dump(mode="json"),
                    STANDING_FIELD: approved.decision.standing,
                },
                decision="allow",
                rationale=approved.decision.reason,
            )
        )
        return approved


def _outcome(result: ToolResult) -> CallOutcome:
    if not result.success:
        return CallOutcome(text=result.error or result.output, succeeded=False)
    return CallOutcome(text=result.output, succeeded=True, query=_query(result))


def _query(result: ToolResult) -> QueryRecord | None:
    """The result as a query record, when it is one (a tool that returned rows)."""
    try:
        return QueryRecord.model_validate(result.data)
    except ValidationError:
        return None


__all__ = ["CallOutcome", "GovernedCalls"]
