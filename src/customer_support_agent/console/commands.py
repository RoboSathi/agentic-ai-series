"""Customer Support's own console command (LLD §20, §21): `/void`.

An issued refund settles on its own when its void window closes (the
console's housekeeping sweep); until then a reviewer can void it.

Review decisions are the platform's `/review` (ADR 0012), with this agent's
refund and reply kinds added by `console/review_handler.py`.

Scoped like the agent's tools: in a console session these show and act on
only the customer verified in that session (`ConsoleScope`). The agent has
no route to any of this: commands are dispatched before a message ever
becomes a conversation turn.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.interface.commands import CommandError, CommandResult
from agent_platform.interface.commands._pool import PoolBackedCommand
from agent_platform.interface.commands.scope import ConsoleScope
from customer_support_agent.config import get_agent_settings


class _SettlementCommand(PoolBackedCommand):
    """The settlement service and the console's scope, for `/void`.

    Scoped like `/review`: in a console session only the verified customer's
    refunds are listed or voided, and someone else's order gets the
    answer an order with nothing issued gets.
    """

    group = "Agent"
    name: str
    summary: str
    usage: str

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        super().__init__(pool)
        self._scope: ConsoleScope | None = None

    def bind_scope(self, scope: ConsoleScope) -> None:
        """Act only on the console session's verified customer's refunds."""
        self._scope = scope

    def argument_completions(self, args: list[str]) -> list[str]:
        """The orders with an issued refund: a closed set, so it completes."""
        return [] if args else [str(order.order_id) for order in self._issued()]

    def _unverified(self) -> CommandResult | None:
        if self._scope is not None and self._scope.scoped and self._scope.subject() is None:
            return CommandResult(title=self.summary, ok=False).line(
                get_settings().commands.verify_first
            )
        return None

    def _customer(self) -> int | None:
        subject = self._scope.subject() if self._scope is not None else None
        return int(subject) if subject is not None else None

    def _service(self) -> Any:
        from customer_support_agent.graph.builder import build_settlement_service

        return build_settlement_service(self.pool)

    def _issued(self) -> list[Any]:
        return list(self._service().issued(self._customer()))

    def _is_theirs(self, order_id: int) -> bool:
        customer = self._customer()
        return customer is None or any(order.order_id == order_id for order in self._issued())

    def _listing(self) -> CommandResult:
        words = get_agent_settings().settlement
        issued = self._issued()
        result = CommandResult(title=words.titles["issued"].format(count=len(issued)))
        if not issued:
            return result.line(words.lines["none_issued"])
        result.headers = list(words.headers)
        for order in issued:
            result.row(str(order.order_id), str(order.amount), str(order.refund_settles_at)[:19])
        return result


class VoidCommand(_SettlementCommand):
    """Reverse an issued refund before it settles (a VOIDABLE action, undone)."""

    name = "void"
    summary = "Issued refunds: list them, or void one before it settles"
    usage = "/void [<order_id>]"

    async def run(self, args: list[str]) -> CommandResult:
        refused = self._unverified()
        if refused is not None:
            return refused
        if not args:
            return self._listing()
        if not args[0].isdigit():
            raise CommandError(f"Usage: {self.usage}")
        order_id = int(args[0])
        words = get_agent_settings().settlement
        if not self._is_theirs(order_id):
            return CommandResult(
                title=words.titles["refused"].format(order_id=order_id), ok=False
            ).line(words.lines["not_issued"].format(order_id=order_id))
        outcome = self._service().void(order_id, get_agent_settings().console_reviewer)
        if outcome.state == "voided":
            return CommandResult(title=words.titles["voided"].format(order_id=order_id)).line(
                words.lines["voided"].format(order_id=order_id, amount=outcome.amount)
            )
        key = "already_settled" if outcome.state == "already_settled" else "not_issued"
        return CommandResult(
            title=words.titles["refused"].format(order_id=order_id), ok=False
        ).line(words.lines[key].format(order_id=order_id))
