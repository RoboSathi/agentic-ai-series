---
name: refund-workflow
description: Handle a customer request to refund an existing order — check it is delivered or cancelled and not already refunded, confirm its full total, request the refund, and report what the result says.
triggers: [intent:REQUEST_REFUND]
keywords: [refund, money back, reimburse, chargeback, refund this order]
prefetch: [get_order_status, get_refund_status]
tools: [get_order_status, get_refund_status, process_refund, list_orders]
---

## Preconditions
- Any request mentioning a refund is a refund request.

## Procedure
1. The order's record and refund status are already above when the order is known;
   otherwise call `list_orders` and use the one they mean.
2. Only a delivered or cancelled order can be refunded. A pending or shipped order has
   nothing to refund yet; a pending one can be cancelled.
3. An order already refunded (`refund_status` issued or completed) cannot be refunded
   again — say so and call nothing. A refund a reviewer declined (`denied`) is final —
   say so, give the escalation email the record states, and call nothing.
4. Otherwise call `process_refund` with the order's full total, whatever it is. The
   result says whether it was issued or sent to a person for review.

## Boundaries
- When the result says it was queued for review, say so, say they will hear back soon,
  and call nothing else.

## Completion
Stop after the result has been communicated.
