# presentation — orders, for a person to read

Deterministic rendering: the order menu a customer chooses from, one
order's details, and the fields each stage card shows in the console. Where
a reply can be built from the record, it is — a model is not asked to
restate a delivery date.

## Vocabulary

- **order menu** — the numbered list a customer picks an order from

## Public interface

- `build_menu`, `OrderMenu`, `resolve_selection`, `render_order_list`, `render_order_detail`, `customer_orders`, `describe_customer` (`orders.py`)
- Stage-card field builders: `tool_request_fields`, `tool_result_fields`, `gatekeeper_fields`, … (`reporting.py`)

## Configuration

No section of its own.

## Collaborators

- Used by the graph nodes; the eval harness rebuilds tool calls from the fields `reporting.py` emits.

## Tests

Run with `./venv/bin/pytest tests/unit/test_order_menu.py tests/test_order_reference_conversation.py -q` (or `make test` for everything).
