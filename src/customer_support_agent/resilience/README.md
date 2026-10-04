# resilience — a refund that survives failing halfway, and settles when it should

`RefundSaga` issues a refund: mark it **issued** (one conditional update that
carries the refund state the caller saw, the order's full total and a refundable
order state) → record the payment. Each step has a compensation, undone in
reverse if a later step fails, and the saga is idempotent so a retried refund
never pays twice. `CancellationSaga` cancels a pending order first, then runs
the same refund steps.

`RefundSettlement` is what happens after: an issued refund is **voidable** until
its window closes (`/void` reverses the payment, puts the order back to no
refund, and releases the idempotency records it was made under, so asking again
pays again); when the window closes, or on `/settle`, it **settles** — the
refund completes and only then is the customer notified.

## Vocabulary

- **compensation** — the undo for one step
- **issued / settled** — paid and still voidable / completed and final (payments' void-before-settlement)
- **void** — reversing an issued refund before it settles

## Public interface

- `RefundSaga`, `refund_key` (`refund_saga.py`); `CancellationSaga`, `cancellation_key` (`cancellation_saga.py`)
- `RefundSettlement`, `SettlementOutcome` (`refund_settlement.py`)

## Configuration

`customer_support_agent.settlement` (`window_seconds`, `sweep_interval_seconds`, the
`/void` and `/settle` wording); the platform's `resilience.saga` and `idempotency`.

## Collaborators

- `agent_platform.resilience`, `agent_platform.executor.current_execution_key`,
  `persistence.OrdersRepository` / `RefundLedgerRepository`; the console's `/void`,
  `/settle` and its settlement sweep (`HousekeepingHost`).

## Episodes

E35, E38, E48b, E48c.

## Tests

Run with `./venv/bin/pytest tests/unit/test_settlement_and_standings.py tests/unit/test_gatekeeper_demo.py tests/unit/test_customer_support_agent.py -q` (or `make test` for everything).
