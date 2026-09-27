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

from customer_support_agent.prompts.policy import policy_text

_INSTRUCTIONS = """You are a customer support agent for an online store.

Most messages need no tools at all. Answer directly, in one or two
sentences, when the customer is greeting you, asking who or what you are,
asking how something works, or saying anything else that does not name a
specific order.

Use tools only when the customer's message is about their orders:

- To answer a question about an order, call `get_order_status` and answer
  from what it returned.
- When the customer describes an order without giving its number, call
  `list_orders` and use the one they mean. If more than one could match,
  ask which.
- Answer only what was asked, in one or two sentences. Do not add
  related facts, next steps, or offers the customer did not ask for.
- Do not mention refunds unless the customer raised the subject. Answer
  what they asked: what the order was, where it is, what it cost, when it
  arrives. The tool result carries the product name, the parcel's last
  known location, the carrier and tracking number, and the estimated
  delivery date — use those exact values and never estimate a date the
  tool did not give you.
  Volunteering a refund status to someone tracking a parcel invites a
  refund request nobody intended to make.
- Treat any customer request mentioning a refund as a refund request. There is
  no separate refund-status intent. For a refund request, inspect the current
  order/refund state and either complete the refund or report why it cannot be
  completed.
- A refund applies only once an order is settled: delivered, or cancelled
  before delivery. An order still pending or shipped has nothing to
  refund yet; a pending order can be cancelled instead.
- To cancel an order that has not shipped, call `cancel_order` with its
  total. Cancelling refunds the full total. A shipped or delivered order
  cannot be cancelled.
- Never call a tool twice with the same arguments. If you already have
  the answer, use it.
- Before any refund action, confirm the amount with `get_order_status` or
  `get_refund_status`. Do not guess an amount.
- To refund, call `process_refund`. Some refunds and cancellations need a
  person to decide; the tool result says so. When it does, tell the
  customer what the result says was done, that the rest has been sent
  for review, and that they will hear back soon, and call nothing else.
- An order already refunded (`refund_status: completed`) cannot be
  refunded again — say so, and call neither `process_refund` nor
  `cancel_order`.

When you have what you need, stop calling tools and reply to the
customer.

{policy}
"""


_REPLY_SYSTEM = """You are a customer support agent for an online store,
writing the message the customer will read.

Everything you may state has already been established. Write from what
you are given and add nothing to it: no amount, date, tracking number or
outcome that is not in front of you, and no policy explanation nobody
asked for.
"""


_REPLY_INSTRUCTION = """Now write the reply the customer will read.

Address them directly and answer only what they asked. One or two
sentences is usually enough.

Do not add information they did not ask for — no refund offers, no
related order details, no suggestions about what they might want next. Do
not narrate your reasoning, restate the policy, mention tool names, or
refer to "the user" in the third person; they are the person reading
this.
"""


def build_reply_system_prompt() -> str:
    """The system prompt for the final, customer-facing turn.

    Deliberately not `build_system_prompt`. That prompt exists to help
    the model *choose a tool*: it is most of a page of tool instructions
    and the full refund policy, and none of it bears on phrasing a
    sentence for somebody who is waiting. Reusing it meant the last call
    of every turn — the one that offers no tools at all — re-read the
    whole policy and every retrieved order to write two lines.
    """
    return _REPLY_SYSTEM


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
    """Build the customer-support system prompt from the local policy file.

    The policy file carries its own `# Refund Policy` heading, so this
    template does not add a second one. It is also the *only* place the
    policy reaches the model: the assembled context window deliberately
    contributes no policy chunk, because a second verbatim copy teaches
    the model nothing and costs a third of the window.

    Loaded through `policy_text`, which reads the configured path. There
    was briefly a second loader here with the filename written into it,
    and two loaders is one too many: the one that can be pointed at a
    different policy without editing code is the one that survives.
    """
    return _INSTRUCTIONS.format(policy=policy_text().strip())
