---
name: order-details
description: Answer what a specific order contains — the product, what it cost, and when it was placed.
triggers: [intent:ORDER_DETAILS]
keywords: [order details, what did the order contain, what was in, cost, price, product]
prefetch: [get_order_status]
tools: [get_order_status]
---

## Preconditions
- The customer is asking about one specific order of their own.

## Procedure
1. The order's record is already above when the order is known; otherwise ask which
   order they mean.
2. Answer the detail they asked for, and only that.

## Boundaries
- Do not recite the whole record.

## Completion
Stop once the detail they asked for has been given.
