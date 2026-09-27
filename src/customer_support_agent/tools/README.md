# tools — what this agent can do

Four typed tools under the `support` namespace: `get_order_status`,
`get_refund_status` (read), `process_refund` (write, through the saga) and
`queue_for_approval` (write, to the review queue). Each has a permission and
an authority rank, and every call passes the Gatekeeper first.

## Vocabulary

- **namespace** — `support` — the model calls `support.process_refund`

## Public interface

- `SupportTools`, `CustomerSupportToolHandler`, `OrderLookupArgs`, `RefundArgs`, `ApprovalArgs`, `NAMESPACE` (`definitions.py`)

## Configuration

`customer_support_agent.intents[*].tools` decides which intent unlocks which tool.

## Collaborators

- `agentic_commons.tools`, `gatekeeper`, `executor`; `domain.RefundPolicy`; `resilience.RefundSaga`.

## Episodes

E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_customer_support_agent.py tests/unit/test_refund_eligibility.py -q` (or `make test` for everything).
