---
name: order-status
description: Answer where an order is — whether it has shipped, where the parcel was last seen, and when it is expected to arrive.
triggers: [intent:ORDER_STATUS]
keywords: [where is my order, track, tracking, parcel, shipped, arrived, delivery, arrive]
prefetch: [get_order_status]
tools: [get_order_status]
---

## Preconditions
- The customer is asking about one of their own orders.

## Procedure
1. The order's record is already above when the order is known; otherwise ask which
   order they mean.
2. Answer from the record: status, last known location, carrier and tracking number,
   estimated delivery date.

## Boundaries
- Do not volunteer a refund status to someone tracking a parcel.

## Completion
Stop once the customer has the status they asked for.
