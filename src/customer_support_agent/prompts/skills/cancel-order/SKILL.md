---
name: cancel-order
description: Cancel an order that has not shipped yet, refunding its full total, and report what the result says.
triggers: [intent:CANCEL_ORDER]
keywords: [cancel, cancel my order, cancel order, stop my order]
prefetch: [get_order_status]
tools: [get_order_status, cancel_order, list_orders]
---

## Preconditions
- The customer wants to cancel one of their orders.

## Procedure
1. The order's record is already above when the order is known; otherwise call
   `list_orders` and use the one they mean.
2. A shipped or delivered order cannot be cancelled — say so.
3. Otherwise call `cancel_order` with its full total; cancelling refunds it.

## Boundaries
- When the result says it was queued for review, say so and call nothing else.

## Completion
Stop after the result has been communicated.
