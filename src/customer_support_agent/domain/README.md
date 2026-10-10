# domain — orders, customers, and who is asking

The business vocabulary and its rules: orders and their states, users,
review requests, the authenticated customer (a context variable the model
can never set), resolving which order a message means, and the refund
policy — the single source of truth every other component asks.

## Vocabulary

- **refundable** — delivered, or cancelled before delivery; pending and shipped orders have nothing to refund yet
- **already refunded** — `refund_status == completed` — never refunded again
- **direct limit** — the most this agent may refund without a person (from its persisted policy, $50 in the seed)
- **full refund only** — a refund is always the whole order total
- **order under discussion** — the order the conversation established, which mandates are scoped to

## Public interface

- `Order`, `User`, `HitlRequest`, `OrderStatus`, `RefundStatus`, `REFUNDABLE_ORDER_STATUSES` (`models.py`)
- `Customer`, `current_customer`, `set_current_customer` (`identity.py`)
- `OrderReferenceResolver`, `ORDER_UNDER_DISCUSSION` (`order_reference.py`)
- `RefundPolicy.validate(order_id, customer_id, amount, action=...)` → `RefundValidationResult` with a reason code (`refund_policy.py`)

## Configuration

No section of its own; the direct limit comes from the database policy, not config.

## Seams

- Identity is bound upstream (authentication, an SSO session, a test fixture), never by a tool argument.

## Collaborators

- Used by `deterministic_checks`, `gatekeeper`, `tools`, `resilience` (the saga). The eval oracle deliberately does *not* use `RefundPolicy` — it checks side effects against the fixture, so the policy code is what is under test.

## Episodes

E01, E35, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_refund_eligibility.py tests/unit/test_authentication_and_scope.py -q` (or `make test` for everything).
