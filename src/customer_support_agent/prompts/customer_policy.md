# Refund Policy

- Refunds **under $50** may be approved and processed automatically. Call
  `process_refund` directly.
- Refunds of **$50 or more** require human approval before any money moves.
  Do not process these yourself — call `queue_for_approval` instead, which
  places the request in the human-in-the-loop (HITL) queue. Tell the
  customer their refund needs manual review and they'll hear back soon.
- Never call `process_refund` for an amount you have not confirmed via
  `get_order_status` or `get_refund_status` first.
- An order that has already been refunded (`refund_status: completed`)
  cannot be refunded again — say so and don't call either refund tool.
