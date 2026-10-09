---
name: order-status
description: Answer where an order is — whether it has shipped, where the parcel was last seen, and when it is expected to arrive.
triggers: [intent:ORDER_STATUS, route:llm_fallback]
keywords: [where is my order, track, tracking, parcel, shipped, arrived, delivery, arrive]
---

## Preconditions
- The customer is asking about one of their own orders.
- If they give no order number, establish which order they mean first
  (see "Procedure", step 1).

## Procedure
1. When the customer describes an order without giving its number, call
   `list_orders` and use the one they mean. If more than one could match,
   ask which.
2. Call `get_order_status` for that order.
3. Answer from what it returned: the status, the parcel's last known
   location, the carrier and tracking number, and the estimated delivery
   date.

## Boundaries
- Never estimate a delivery date the tool did not give you.
- Do not volunteer a refund status to someone tracking a parcel: it
  invites a refund request nobody intended to make.

## Completion
Stop once the customer has the status they asked for.
