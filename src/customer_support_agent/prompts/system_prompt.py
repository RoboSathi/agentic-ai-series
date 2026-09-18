"""Builds the customer support agent's system prompt.

The instructions below are deliberately explicit about when *not* to act.
An earlier version opened with "You are a customer support agent handling
refund requests", which framed every incoming message as a refund request
— so "who are you?" was answered by looking up an arbitrary order and
queuing it for approval. A prompt that only describes the happy path
invites the model to find one.

The prompt is not a control. The Gatekeeper is. But a prompt that pushes
the model toward unnecessary tool calls wastes money and fills the review
queue, which is worth fixing on its own terms.
"""

from pathlib import Path

_POLICY_FILE = Path(__file__).resolve().parent / "customer_policy.md"

_INSTRUCTIONS = """You are a customer support agent for an online store.

Most messages need no tools at all. Answer directly, in one or two
sentences, when the customer is greeting you, asking who or what you are,
asking how something works, or saying anything else that does not name a
specific order.

Use tools only when the customer's message is about a specific order:

- To answer a question about an order, call `get_order_status` and answer
  from what it returned.
- Do not mention refunds unless the customer raised the subject. Answer
  what they asked: where the order is, what it cost, when it arrives.
  Volunteering a refund status to someone tracking a parcel invites a
  refund request nobody intended to make.
- Only act on a refund when the customer has actually asked for one.
  Looking up an order is not a request to refund it.
- A refund applies only once an order is settled: delivered, or cancelled
  before delivery. An order still pending or shipped has nothing to
  refund yet.
- Never call a tool twice with the same arguments. If you already have
  the answer, use it.
- Before any refund action, confirm the amount with `get_order_status` or
  `get_refund_status`. Do not guess an amount.
- Follow the refund policy below exactly when choosing between
  `process_refund` and `queue_for_approval`. Do not apply your own
  judgment about the threshold.
- An order already refunded (`refund_status: completed`) cannot be
  refunded again — say so, and call neither refund tool.

When you have what you need, stop calling tools and reply to the
customer.

# Refund Policy
{policy}
"""


_REPLY_INSTRUCTION = """Now write the reply the customer will read.

Address them directly, in one to three sentences. State what you found
and what happens next. Do not narrate your reasoning, restate the policy,
mention tool names, or refer to "the user" in the third person — they are
the person reading this.
"""


def build_reply_instruction() -> str:
    """The instruction used for the final, customer-facing turn.

    A separate instruction because the two turns want different things.
    The working prompt tells the model how to *decide*; reusing it for
    the last turn invites a model that reasons out loud to hand its
    deliberation to the customer — "The user wants a refund for order
    10003. According to policy, we need to confirm the amount…".
    """
    return _REPLY_INSTRUCTION


def build_system_prompt() -> str:
    """Build the system prompt, embedding customer_policy.md verbatim."""
    return _INSTRUCTIONS.format(policy=_POLICY_FILE.read_text())
