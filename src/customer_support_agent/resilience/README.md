# resilience — a refund that survives failure and settles when it should

`RefundSaga` marks a refund issued (one conditional update) and records the payment, each step compensated, idempotent; `CancellationSaga` cancels a pending order first; `RefundSettlement` voids (`/void`) or settles an issued refund when its window closes.

- **Entry points:** `RefundSaga`, `CancellationSaga`, `RefundSettlement`
- **Configuration:** `customer_support_agent.settlement`
- **Design:** [ADR C-03](../../../design_docs/ADR.md#c-03--a-refund-is-issued-then-settles)
