# domain — orders, customers, and the refund policy

The business vocabulary and its rules: orders and their states, the authenticated customer (a context variable the model never sets), the order under discussion, and `RefundPolicy` / `CancellationPolicy` — the single source of business truth.

- **Entry points:** `Order`, `OrderStatus`, `RefundStatus`, `Customer`, `OrderReferenceResolver`, `RefundPolicy`, `CancellationPolicy`, `ports.py`
- **Configuration:** none (the direct limit is a database row)
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent) · [ADR C-01](../../../design_docs/ADR.md#c-01--business-rules-live-in-code-not-in-the-prompt)
