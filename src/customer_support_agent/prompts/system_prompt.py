"""Builds the customer support agent's system prompt."""

from pathlib import Path

_POLICY_FILE = Path(__file__).resolve().parent / "customer_policy.md"

_INSTRUCTIONS = """You are a customer support agent handling refund requests.

Use get_order_status / get_refund_status to look up an order before acting.
Follow the refund policy below exactly when deciding whether to call
process_refund or queue_for_approval — do not guess or apply your own
judgment about the dollar threshold.

# Refund Policy
{policy}
"""


def build_system_prompt() -> str:
    """Build the system prompt, embedding customer_policy.md verbatim."""
    policy = _POLICY_FILE.read_text()
    return _INSTRUCTIONS.format(policy=policy)
