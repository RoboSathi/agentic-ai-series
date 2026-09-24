"""Customer Support's own console commands (LLD §20, §21).

`/users`, `/orders` and `/queue` read this agent's tables and speak this
agent's vocabulary, so they belong to this agent. They used to live in
`agentic_commons/commands/`, which meant the platform's console could
not be opened without a refund database behind it.

These are operator views, deliberately *not* scoped to an authenticated
customer the way the agent's tools are. An operator inspecting the
database is a different principal from a customer asking about their own
account, and conflating the two would either cripple the console or
widen what the agent can reach. The agent has no route to any of them:
they are dispatched before a message ever becomes a conversation turn.
"""

from typing import Any, get_args

from agentic_commons.commands import CommandError, CommandResult
from agentic_commons.commands._pool import PoolBackedCommand
from agentic_commons.config import get_settings
from agentic_commons.observability.queue import LittlesLawQueueModel
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.domain.models import OrderStatus
from customer_support_agent.persistence.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    UsersRepository,
)


class UsersCommand(PoolBackedCommand):
    """List customers, or look one up by code."""

    name = "users"
    summary = "List customers, or look one up by code"
    usage = "/users [code|limit]   e.g. /users 2HEDZTV"

    async def run(self, args: list[str]) -> CommandResult:
        pool = self.pool
        users = UsersRepository(pool)
        orders = OrdersRepository(pool)

        if args and not args[0].isdigit():
            user = users.get_by_code(args[0])
            if user is None:
                raise CommandError(f"No customer with code '{args[0]}'")
            result = CommandResult(title=f"Customer #{user.user_id}")
            result.headers = ["field", "value"]
            result.row("name", user.name)
            result.row("code", user.user_code)
            result.row("email", user.email)
            result.row("phone", user.phone or "—")
            result.row("since", user.created_at[:10])
            result.row("orders", str(len(orders.for_user(user.user_id))))
            return result

        limit = (
            int(args[0]) if args and args[0].isdigit() else get_settings().commands.default_limit
        )
        rows = pool.execute(
            "SELECT u.user_id, u.user_code, u.name, u.email, COUNT(o.order_id) AS orders "
            "FROM users u LEFT JOIN orders o ON o.user_id = u.user_id "
            "GROUP BY u.user_id ORDER BY u.user_id LIMIT ?",
            (limit,),
        ).rows
        total = pool.query_one("SELECT COUNT(*) AS n FROM users")
        result = CommandResult(
            title=f"Customers — showing {len(rows)} of {total['n'] if total else 0}"
        )
        result.headers = ["id", "code", "name", "email", "orders"]
        for row in rows:
            result.row(
                str(row["user_id"]),
                row["user_code"],
                row["name"],
                row["email"],
                str(row["orders"]),
            )
        result.line("Use /users <code> for one customer, /orders <code> for their orders.")
        return result


class OrdersCommand(PoolBackedCommand):
    """List orders, filtered by customer code or status."""

    name = "orders"
    summary = "List orders, by customer code or status"
    usage = "/orders [code|status] [limit]   e.g. /orders delivered 10"

    async def run(self, args: list[str]) -> CommandResult:
        pool = self.pool
        settings = get_settings().commands
        limit = next((int(arg) for arg in args if arg.isdigit()), settings.default_limit)
        selector = next((arg for arg in args if not arg.isdigit()), "")

        if selector and selector.casefold() in _order_statuses():
            return self._by_status(pool, selector.casefold(), limit)
        if selector:
            user = UsersRepository(pool).get_by_code(selector)
            if user is None:
                raise CommandError(
                    f"'{selector}' is neither a customer code nor a status "
                    f"({', '.join(sorted(_order_statuses()))})"
                )
            found = OrdersRepository(pool).for_user(user.user_id)
            result = CommandResult(title=f"{len(found)} orders for {user.name}")
            return self._table(result, [order.model_dump() for order in found][:limit])

        rows = pool.execute(
            "SELECT order_id, user_id, timestamp, order_status, refund_status, amount "
            "FROM orders ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).rows
        total = pool.query_one("SELECT COUNT(*) AS n FROM orders")
        result = CommandResult(
            title=f"Orders — showing {len(rows)} of {total['n'] if total else 0}, newest first"
        )
        return self._table(result, rows)

    def _by_status(self, pool: SqliteConnectionPool, status: str, limit: int) -> CommandResult:
        rows = pool.execute(
            "SELECT order_id, user_id, timestamp, order_status, refund_status, amount "
            "FROM orders WHERE order_status = ? ORDER BY timestamp DESC LIMIT ?",
            (status, limit),
        ).rows
        count = pool.query_one("SELECT COUNT(*) AS n FROM orders WHERE order_status = ?", (status,))
        result = CommandResult(
            title=f"Orders — {count['n'] if count else 0} {status}, showing {len(rows)}"
        )
        return self._table(result, rows)

    @staticmethod
    def _table(result: CommandResult, rows: list[dict[str, Any]]) -> CommandResult:
        result.headers = ["order", "customer", "placed", "status", "refund", "amount"]
        for row in rows:
            refund = str(row["refund_status"])
            result.row(
                str(row["order_id"]),
                str(row["user_id"]),
                str(row["timestamp"])[:10],
                str(row["order_status"]),
                "—" if refund == "none" else refund,
                f"${float(row['amount']):,.2f}",
                style="yellow" if refund == "pending_review" else "",
            )
        if not rows:
            result.line("No orders matched.")
        return result


def _order_statuses() -> frozenset[str]:
    """The statuses an order can hold, from the domain type itself.

    Taken from the `OrderStatus` literal rather than restated here: a
    second list of statuses is a list that goes stale the first time the
    domain gains one.
    """
    return frozenset(get_args(OrderStatus))


class QueueCommand(PoolBackedCommand):
    """Show the human review queue and the staffing it implies.

    The queue is this agent's — a refund waiting on a person. The
    staffing arithmetic is not: Little's Law is the platform's, and is
    composed here rather than restated.
    """

    name = "queue"
    summary = "Human review queue and its staffing maths"
    usage = "/queue [arrivals_per_hour] [minutes_each]"

    async def run(self, args: list[str]) -> CommandResult:
        pending = HitlQueueRepository(self.pool).pending()
        result = CommandResult(title=f"Review queue — {len(pending)} pending")
        if pending:
            result.headers = ["id", "order", "customer", "amount", "raised"]
            for request in pending[: get_settings().commands.recent_window]:
                result.row(
                    str(request.id),
                    str(request.order_id),
                    str(request.user_id),
                    f"${request.requested_amount:,.2f}",
                    request.created_at[:16],
                )
        else:
            result.line("Nothing awaiting review.")

        review = get_settings().observability.review_queue
        arrivals = float(args[0]) if args else max(len(pending), 1) * 1.0
        # The configured review time is the default, not a literal here.
        # A hardcoded 4.0 meant `minutes_per_review` was config nothing read.
        minutes = float(args[1]) if len(args) > 1 else review.minutes_per_review
        capacity = LittlesLawQueueModel().capacity_for(arrivals, minutes)
        result.line("")
        result.line(
            f"Little's Law at {arrivals:g}/hr × {minutes:g} min: "
            f"{capacity.reviewer_hours_per_hour:.2f} reviewer-hours per hour "
            f"({capacity.reviewers_required} people)"
        )
        return result
