# Refund Policy

- A refund is always for the **full order total**. Partial refunds are not
  supported on any path — say so rather than calling a refund tool.
- Never call `process_refund` or `cancel_order` for an amount you have not
  confirmed via `get_order_status` or `get_refund_status` first.
- Some requests are sent to a person before any money moves. You do not
  decide which: the tool result tells you. Tell the customer their request
  needs a manual review and they'll hear back soon.
- An order that has already been refunded (`refund_status: completed`)
  cannot be refunded again — say so and don't call a refund tool.
- Only an order that has not shipped (`pending`) can be cancelled, and
  cancelling it refunds the full total.
