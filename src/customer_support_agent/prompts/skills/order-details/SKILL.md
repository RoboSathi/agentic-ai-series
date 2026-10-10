---
name: order-details
description: Answer what a specific order contains — the product, what it cost, and when it was placed.
triggers: [intent:ORDER_DETAILS]
keywords: [order details, what did the order contain, what was in, cost, price, product]
prefetch: [look_up_order]
tools: [look_up_order, list_my_orders]
---

## Preconditions
- The customer is asking about one specific order of their own.

## Procedure
1. The order's record is already above when the order is known; otherwise call
   `list_my_orders` and use the one they mean.
2. Answer the detail they asked for, and only that.

## Boundaries
- Do not recite the whole record.

## Completion
Stop once the detail they asked for has been given.
