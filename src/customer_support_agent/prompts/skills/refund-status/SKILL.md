---
name: refund-status
description: Answer where a refund stands or when it will arrive, from the refund record.
triggers: [intent:REFUND_STATUS]
keywords: [refund status, where is my refund, when will i get my refund, track refund]
---

## Preconditions
- The customer is asking about a refund already requested, not asking
  for a new one.

## Procedure
1. Establish the order the refund belongs to (ask if it is ambiguous).
2. Call `get_refund_status` for it.
3. Answer from the result: whether it is issued, settled, or under
   review, and the amount it records.

## Boundaries
- An *issued* refund settles shortly; do not call it complete.
- Do not start a new refund from a status question.

## Completion
Stop once the customer knows where their refund stands.
