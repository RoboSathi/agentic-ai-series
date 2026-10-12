---
name: refund-status
description: Answer where a refund stands or when it will arrive, from the refund record.
triggers: [intent:REFUND_STATUS]
keywords: [refund status, where is my refund, when will i get my refund, track refund]
prefetch: [get_refund_status]
tools: [get_refund_status]
---

## Preconditions
- The customer is asking about a refund already requested.

## Procedure
1. The refund record is already above when the order is known; otherwise ask which
   order they mean.
2. Answer from it: issued, settled or under review, and the amount.

## Boundaries
- Do not start a new refund from a status question.

## Completion
Stop once the customer knows where their refund stands.
