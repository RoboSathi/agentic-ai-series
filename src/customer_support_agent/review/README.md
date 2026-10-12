# review — the human half of escalation

`ReviewQueueService`: a reviewer approves or denies a queued refund; approving resumes the waiting durable run so the refund executes through the same governed path. Items live in the platform's one review queue.

- **Entry points:** `ReviewQueueService` (`queue.py`), referral of escalated refunds
- **Configuration:** `customer_support_agent.console_reviewer`
- **Design:** [ADR P-14](../../../design_docs/ADR.md#p-14--one-human-review-queue-a-comment-on-every-decision)
