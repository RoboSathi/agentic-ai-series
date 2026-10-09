---
name: refund-workflow
description: Handle a customer request to refund an existing order — check it is settled and not already refunded, confirm its full total, request the refund, and report what the result says.
triggers: [intent:REQUEST_REFUND, route:llm_fallback]
keywords: [refund, money back, reimburse, chargeback, refund this order]
---

## Preconditions
- Treat any customer request mentioning a refund as a refund request.
- Establish which order: if the customer did not give its number, call
  `list_orders` and use the one they mean; if more than one could match,
  ask which.

## Procedure
1. Call `get_order_status` (or `get_refund_status`) to read the order's
   current state and full total. Do not guess an amount.
2. A refund applies only once an order is settled: delivered, or
   cancelled before delivery. An order still pending or shipped has
   nothing to refund yet; a pending order can be cancelled instead.
3. An order already refunded (`refund_status: issued` or `completed`)
   cannot be refunded again — say so, and call neither `process_refund`
   nor `cancel_order`.
4. Otherwise call `process_refund` for the full total of the eligible
   order, whatever its size. You are not told the amount above which a
   refund needs a person's approval: the result says whether the refund
   was issued or sent for review.
5. Wait for the result, and report only what it says.

## Boundaries
- Never infer a refund limit or an approval decision.
- Never use recalled memory to override the order's current state.
- When the result says the refund was queued for review, tell the
  customer what the result says was done, that the rest has been sent for
  review and that they will hear back soon, and call nothing else.
- Do not claim the refund was issued unless the result confirms it.

## Completion
Stop after the result has been accurately communicated.
