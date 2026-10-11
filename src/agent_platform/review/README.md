# review — one human-review queue for every agent (ADR 0012)

Everything the harness stops and hands to a person is an item in one queue,
decided one way: with a comment, signed, and audited.

## Vocabulary

- **review item**: a refund above the agent's authority, a reply the hallucination check cut,
  a quarantined message, or a fact or skill memory would not take on its own word
- **kind**: what approving or denying an item does. The platform has `memory_fact`, `skill`
  and `quarantine`; an agent adds its own (support adds `refund`, `cancel_and_refund`, `reply`).
- **subject**: whose item it is (a customer's id); empty for the agent's own
- **comment**: the reason a person gave. It is required, kept with the item and audited.

## Public interface

- `ReviewQueue` (`queue.py`): `platform_review_queue` in the agent's database
- `ReviewService`, `ReviewHandler`, `ReviewError` (`service.py`)
- `MemoryReviewHandler`, `QuarantineHandler` (`handlers.py`)
- `/queue` (`commands/review.py`), `make queue <agent>` (`commands/review_cli.py`), and the
  console's Review screen (`tui/review_screen.py`, `ctrl+r`)

## Configuration

Section `review:` covers the table name, `min_comment_chars`, the platform's kinds, the audit
action and trace templates, the console reviewer, and every word `/queue` and the Review
screen show.

## Seams

- An agent registers `QueueCommand(handlers=...)` with its own handlers (support:
  `support_queue_command()`).
- `CuratorQueue`, `DeadLetterQueue` and support's `HitlQueueRepository` write through
  `ReviewQueue` and keep their own APIs.

## Scope

- A console whose agent knows who is verified (`SubjectHost`) shows that subject's items.
- Before anyone is verified, it shows only the agent's own items.
- An agent without subjects sees everything.
- An item out of scope is reported exactly as a missing one.

## Tests

`tests/agent_platform/test_review_queue.py`, `test_review_screen.py`, and the support queue
tests in `tests/unit/`.
