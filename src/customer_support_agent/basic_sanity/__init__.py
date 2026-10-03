"""This agent's own pre-flight checks on a request.

`basic_sanity` is the seam where an agent's business logic meets the
platform: `agent_platform` owns only the contract, and this is the
implementation Customer Support injects into it.
"""

from customer_support_agent.basic_sanity.order_validator import OrderSanityCheck, owned_order_id

__all__ = ["OrderSanityCheck", "owned_order_id"]
