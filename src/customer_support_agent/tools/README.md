# tools — what this agent can do

Five typed tools under the `support` namespace: `get_order_status`, `list_orders`, `get_refund_status` (reads), `process_refund` and `cancel_order` (writes, through the sagas). Each has a permission and an authority rank; every call takes the platform tool path.

- **Entry points:** `SupportTools`, `CustomerSupportToolHandler` (`definitions.py`)
- **Configuration:** `customer_support_agent.intents[*].tools`
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent)
