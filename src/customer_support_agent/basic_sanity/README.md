# basic_sanity — this agent's pre-flight checks

`OrderSanityCheck`: before any model is asked, when a message names an
order explicitly, check that it exists, belongs to the authenticated
customer, is not already refunded and is in a refundable state. A refusal
here costs no model call and cannot be talked around.

## Vocabulary

- **order reference** — an explicit `order 123` / `#123` / `refund 10001` in the message
- **finding** — one check with its evidence (`order_exists_and_is_owned`, `not_already_refunded`, `refund_eligible`)

## Public interface

- `OrderSanityCheck(orders, refund_policy=..., refund_markers=...)` — implements the platform's `BasicSanityChecker`

## Configuration

No section of its own; the stage exists when `features.basic_sanity` is on.

## Seams

- Injected by `graph/builder.py` through `build_agent(basic_sanity=...)`.

## Collaborators

- Uses `domain.RefundPolicy` and `persistence.OrdersRepository`.

## Episodes

The seam that lets business logic join the harness (E01).

## Tests

Run with `./venv/bin/pytest tests/unit/test_basic_sanity.py -q` (or `make test` for everything).
