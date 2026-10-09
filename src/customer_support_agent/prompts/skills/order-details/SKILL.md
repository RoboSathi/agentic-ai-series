---
name: order-details
description: Answer what a specific order contains — the product, what it cost, and when it was placed.
triggers: [intent:ORDER_DETAILS]
keywords: [order details, what did the order contain, what was in, cost, price, product]
---

## Preconditions
- The customer is asking about one specific order of their own.

## Procedure
1. If the customer did not give an order number, call `list_orders` and
   use the one they mean; if more than one could match, ask which.
2. Call `get_order_status` for that order.
3. Answer the detail they asked for, using the exact values returned.

## Boundaries
- Answer only the detail asked for; do not recite the whole record.
- Do not mention refunds unless the customer raised the subject.

## Completion
Stop once the detail they asked for has been given.
