# verification — reading back what a refund did

`RefundEffectSource`: after a refund, reads the refunds ledger and the
order back so the platform's `ReadBackVerifier` can confirm the money
actually moved before any reply claims it did.

## Public interface

- `RefundEffectSource` (`refund_effects.py`)

## Configuration

No section of its own.

## Seams

- Supplied to the platform verifier by the composition root.

## Collaborators

- `agentic_commons.verification`, `persistence.RefundLedgerRepository`.

## Episodes

Effect verification (LLD §24).

## Tests

Covered through the refund flow tests; dedicated tests are an open item.
