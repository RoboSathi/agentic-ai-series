---
name: cancel-order
description: Cancel an order that has not shipped yet, refunding its full total, and report what the result says.
triggers: [intent:CANCEL_ORDER]
keywords: [cancel, cancel my order, cancel order, stop my order]
---

## Preconditions
- Establish which order: if the customer did not give its number, call
  `list_orders` and use the one they mean; if more than one could match,
  ask which.

## Procedure
1. Call `get_order_status` to read the order's state and full total.
2. A shipped or delivered order cannot be cancelled — say so.
3. Otherwise call `cancel_order` with its full total. Cancelling refunds
   the full total.
4. Report what the result says was done.

## Boundaries
- When the result says the cancellation was queued for review, tell the
  customer that and call nothing else.
- Do not claim the order was cancelled unless the result confirms it.

## Completion
Stop after the result has been accurately communicated.
