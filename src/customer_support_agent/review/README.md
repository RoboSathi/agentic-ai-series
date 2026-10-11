# review — the human half of escalation

`ReviewQueueService`: a reviewer approves or denies a refund the agent
queued. Approving resumes the waiting durable run with the decision, so the
refund executes through the same governed path it would have taken.

Since ADR 0012 these are items in the platform's one review queue:
`SupportReviewHandler` (`console/review_handler.py`) is this agent's handler for its kinds
(`refund`, `cancel_and_refund`, `reply`), decided through the platform's
`/queue` and Review screen with a required comment.

## Vocabulary

- **review request** — a queued refund above the direct limit, waiting on a person

## Public interface

- `ReviewQueueService.approve(request_id, reviewer)` / `deny(...)`, `ReviewOutcome` (`queue.py`)

## Configuration

`customer_support_agent.console_reviewer` — who the console acts as.

## Seams

- Reached from the console through `/queue approve|deny <id> <comment>` and the Review screen (`ctrl+r`); `support_queue_command()` registers it with this agent's kinds.

## Collaborators

- `graph` (resume), `persistence.HitlQueueRepository`, `audit`.

## Episodes

E38.

## Tests

Run with `./venv/bin/pytest tests/unit/test_production_hardening.py -q` (or `make test` for everything).
