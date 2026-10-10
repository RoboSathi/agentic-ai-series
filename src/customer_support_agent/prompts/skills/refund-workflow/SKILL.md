---
name: refund-workflow
description: Handle a customer request to refund an existing order — check it is settled and not already refunded, confirm its full total, request the refund, and report what the result says.
triggers: [intent:REQUEST_REFUND, route:llm_fallback]
keywords: [refund, money back, reimburse, chargeback, refund this order]
prefetch: [look_up_order, check_refund]
tools: [look_up_order, check_refund, request_refund, list_my_orders]
---

## Preconditions
- Any request mentioning a refund is a refund request.

## Procedure
1. The order's record and refund status are already above when the order is known;
   otherwise call `list_my_orders` and use the one they mean.
2. Only a settled order (delivered, or cancelled before delivery) can be refunded. A
   pending or shipped order has nothing to refund yet; a pending one can be cancelled.
3. An order already refunded (`refund_status` issued or completed) cannot be refunded
   again — say so and call nothing.
4. A total above the refund limit in the policy: do what the policy says, call nothing.
5. Otherwise call `request_refund` with the order's full total. The result says whether
   it was issued or sent for review.

## Boundaries
- When the result says it was queued for review, say so, say they will hear back soon,
  and call nothing else.

## Completion
Stop after the result has been communicated.
