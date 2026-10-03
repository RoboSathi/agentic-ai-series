# resilience — a refund that survives failing halfway

`RefundSaga`: mark the order refunded → record the payment → notify, each
step with a compensation, undone in reverse if a later step fails, and
idempotent so a retried refund never pays twice.

## Vocabulary

- **compensation** — the undo for one step

## Public interface

- `RefundSaga` (`refund_saga.py`)

## Configuration

The platform's `resilience.saga` and `idempotency`.

## Collaborators

- `agent_platform.resilience`, `persistence.RefundLedgerRepository`.

## Episodes

E35, E38, E48b, E48c.

## Tests

Run with `./venv/bin/pytest tests/unit/test_refund_eligibility.py tests/unit/test_production_hardening.py -q` (or `make test` for everything).
