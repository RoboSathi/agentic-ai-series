---
name: refund-status
description: Answer where a refund stands, when it arrives, or how refunds work, from the refund record and policy.
triggers: [intent:REFUND_STATUS]
keywords: [refund status, where is my refund, when will i get my refund, track refund, how long, show on my card, refund policy]
prefetch: [get_refund_status]
tools: [get_refund_status]
---

## Preconditions
- The customer asks where their refund stands, when it arrives, or how refunds work.

## Procedure
1. The refund record is already above when the order is known; otherwise ask which
   order they mean. Answer from it: issued, settled or under review, and the amount.
2. When it arrives or how refunds work: answer from the refund policy; no order needed.

## Boundaries
- Do not start a new refund from a status question.

## Completion
Stop once the customer knows where their refund stands.
