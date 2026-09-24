"""The human half of the escalation path.

Built from its own composition root, never from the agent's: the agent
must have no route to the queue that reviews it.
"""

from customer_support_agent.review.queue import ReviewOutcome, ReviewQueueService

__all__ = ["ReviewOutcome", "ReviewQueueService"]
