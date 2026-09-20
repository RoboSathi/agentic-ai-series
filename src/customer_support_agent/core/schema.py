"""Domain models and graph state (LLD §21).

The domain vocabulary is deliberately narrow: an Order, a User, a
HITL request, and the state the graph carries between nodes. Everything
else — authority, memory, context, resilience — belongs to
`agentic_commons` and arrives here by injection.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from agentic_commons.llm.types import Message
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.config import get_agent_settings

OrderStatus = Literal["pending", "shipped", "delivered", "cancelled"]
RefundStatus = Literal["none", "pending_review", "approved", "denied", "completed"]
HitlStatus = Literal["pending", "approved", "denied"]

# An order is refundable only once it is settled. Domain invariant, not a
# tunable: it is what "refund" means for this business, and the dummy
# data generator applies the same rule so the fixtures cannot contradict
# the policy the agent is asked to enforce.
REFUNDABLE_ORDER_STATUSES: frozenset[str] = frozenset({"delivered", "cancelled"})

_SCHEMA_FILE = Path(__file__).resolve().parents[1] / "dummy_data" / "schema.sql"


class Order(BaseModel):
    """One customer order."""

    order_id: int
    user_id: int
    timestamp: str
    order_status: OrderStatus
    refund_status: RefundStatus
    amount: float

    @property
    def already_refunded(self) -> bool:
        """Whether this order has been refunded and cannot be refunded again."""
        return self.refund_status == "completed"

    @property
    def is_refund_eligible(self) -> bool:
        """Whether this order has reached a state a refund can apply to.

        An order becomes refundable once it is settled: delivered, so the
        customer has it and wants their money back, or cancelled before
        delivery, so they never received it. A pending or shipped order
        is still in flight and there is nothing to refund yet.
        """
        return self.order_status in REFUNDABLE_ORDER_STATUSES

    @property
    def refund_ineligibility_reason(self) -> str:
        """Why this order cannot be refunded, in words a customer can read."""
        if self.already_refunded:
            return f"Order {self.order_id} has already been refunded in full"
        if not self.is_refund_eligible:
            return (
                f"Order {self.order_id} is still {self.order_status}, so there is "
                f"nothing to refund yet — a refund applies once an order has been "
                f"delivered, or if it is cancelled before delivery"
            )
        return ""


class User(BaseModel):
    """One customer.

    `user_code` is the credential the customer types; `user_id` is the
    internal key that appears in joins and logs. Keeping them separate
    means an enumerable primary key is never the thing that grants
    access.
    """

    user_id: int
    user_code: str
    created_at: str
    name: str
    email: str
    phone: str | None = None

    @property
    def masked_email(self) -> str:
        """Email with the local part partly hidden, for read-back."""
        local, _, domain = self.email.partition("@")
        if not domain:
            return self.email
        shown = local[:2] if len(local) > 2 else local[:1]
        return f"{shown}{'*' * max(3, len(local) - len(shown))}@{domain}"


class HitlRequest(BaseModel):
    """A refund waiting on a human decision."""

    id: int | None = None
    order_id: int
    user_id: int
    requested_amount: float
    status: HitlStatus = "pending"
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


def _merge_messages(existing: list[Message], incoming: list[Message]) -> list[Message]:
    """Reducer: graph nodes append to the transcript rather than replace it."""
    return [*existing, *incoming]


class AgentState(BaseModel):
    """What flows between graph nodes.

    A pydantic model rather than a loose dict so that a node returning
    the wrong shape fails where it happened, not three nodes later where
    the value is finally read.
    """

    query: str = ""
    session_id: str = ""
    actor: str = ""
    customer_id: int | None = None
    messages: Annotated[list[Message], _merge_messages] = Field(default_factory=list)
    context: str = ""
    tool_rounds: int = 0
    tool_results: list[str] = Field(default_factory=list)
    authoritative_facts: dict[str, Any] = Field(default_factory=dict)
    decisions: list[str] = Field(default_factory=list)
    escalated: bool = False
    response: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def ensure_schema(pool: SqliteConnectionPool) -> None:
    """Create the domain tables from the single source of truth.

    The DDL lives in `dummy_data/schema.sql` and is read from disk rather
    than duplicated here: two copies of a schema drift, and the one in
    the test fixture is always the one that drifts.
    """
    pool.execute_script(_SCHEMA_FILE.read_text())


def policy_text() -> str:
    """The refund policy, resolved package-relative and never from the CWD."""
    return get_agent_settings().resolved_policy_path.read_text()
