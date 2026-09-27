# gatekeeper — this agent's rules for the platform Gatekeeper

The deterministic business rules the platform Gatekeeper calls before any
tool runs: is the customer authenticated, does the order exist and belong to
them, does the refund policy allow it, is the amount over this agent's direct
limit (escalate), is a review already waiting. And the mandate policy: an
action may only target an order the conversation established.

## Vocabulary

- **action class** — `refund` — the budget class refund tools are charged against
- **mandate** — permission scoped to the order under discussion, with a maximum amount

## Public interface

- `CustomerSupportBusinessRules.evaluate(call, authority)` → `BusinessRuleResult` (`business_rules.py`)
- `EstablishedOrderScope` — mandates from the conversation (`mandates.py`)
- `REFUND_ACTION_CLASS`

## Configuration

`customer_support_agent.action_classes`; `gatekeeper.mandate.target_param: order_id` in this agent's overlay.

## Seams

- Injected into the platform `GatekeeperImpl` by `graph/builder.py`.

## Collaborators

- `domain.RefundPolicy`, `persistence` repositories.

## Episodes

E02, E03, E12, E33.

## Tests

Run with `./venv/bin/pytest tests/unit/test_gatekeeper.py -q` (or `make test` for everything).
