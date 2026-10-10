---
name: cancel-order
description: Cancel an order that has not shipped yet, refunding its full total, and report what the result says.
triggers: [intent:CANCEL_ORDER, route:llm_fallback]
keywords: [cancel, cancel my order, cancel order, stop my order]
prefetch: [look_up_order]
tools: [look_up_order, cancel_order, list_my_orders]
---

## Preconditions
- The customer wants to cancel one of their orders.

## Procedure
1. The order's record is already above when the order is known; otherwise call
   `list_my_orders` and use the one they mean.
2. A shipped or delivered order cannot be cancelled — say so.
3. Otherwise call `cancel_order` with its full total; cancelling refunds it.

## Boundaries
- When the result says it was queued for review, say so and call nothing else.

## Completion
Stop after the result has been communicated.
