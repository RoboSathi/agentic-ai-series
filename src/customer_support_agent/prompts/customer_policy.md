# Refund Policy

- A refund is always for the **full order total**. Partial refunds are not
  supported on any path — say so rather than calling a refund tool.
- Refunds **up to $50** may be approved and processed automatically. Call
  `process_refund` directly.
- Refunds **above $50** require human approval before any money moves.
  Do not process these yourself — call `queue_for_approval` instead, which
  places the request in the human-in-the-loop (HITL) queue. Tell the
  customer their refund needs manual review and they'll hear back soon.
- Never call `process_refund` for an amount you have not confirmed via
  `get_order_status` or `get_refund_status` first.
- An order that has already been refunded (`refund_status: completed`)
  cannot be refunded again — say so and don't call either refund tool.
