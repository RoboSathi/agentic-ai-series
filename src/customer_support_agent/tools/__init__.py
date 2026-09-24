"""The actions this agent can take, and their governance envelope.

Tools belong to the agent. The platform provides the registry they are
declared into, the selector that narrows them and the executor that runs
them — but never the tools themselves.
"""

from customer_support_agent.tools.definitions import (
    NAMESPACE,
    ApprovalArgs,
    CustomerSupportToolHandler,
    OrderLookupArgs,
    RefundArgs,
    SupportTools,
)

__all__ = [
    "NAMESPACE",
    "ApprovalArgs",
    "CustomerSupportToolHandler",
    "OrderLookupArgs",
    "RefundArgs",
    "SupportTools",
]
