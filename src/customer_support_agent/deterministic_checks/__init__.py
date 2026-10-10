"""This agent's own pre-flight checks on a request.

`deterministic_checks` is the seam where an agent's business logic meets the
platform: `agent_platform` owns only the contract, and this is the
implementation Customer Support injects into it.
"""

from customer_support_agent.deterministic_checks.order_validator import (
    OrderChecks,
    checked_order,
    owned_order_id,
)

__all__ = ["OrderChecks", "checked_order", "owned_order_id"]
