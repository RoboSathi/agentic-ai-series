"""This agent's own analytics (ADR 0009 D7, Tier 2): its fact tables and semantic layer.

The platform records every Tier-1 metric for this agent without any code here.
What only this agent has — a person reviewing a queued refund — is recorded by
the facts in this package, through the platform's fact store, and described in
`semantic_layer.yaml`, which the source, the analytics MCP server and the
dashboards discover by convention.
"""

from customer_support_agent.analytics.facts import ReviewFact, record_review

__all__ = ["ReviewFact", "record_review"]
