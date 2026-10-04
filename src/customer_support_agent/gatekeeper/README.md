# gatekeeper — this agent's rules for the platform Gatekeeper

Two things, both about *authority*, neither about the order's own correctness:

- the refund limit — a refund or cancellation above this agent's per-action limit
  (the persisted policy row; the model never sees it) escalates to a person
  (RATIFIABLE). It reads no database: the limit is loaded with the policy, the
  amount is the call's;
- the mandate policy — a write may only target the order the conversation
  established (else VOID), however valid a refund of that other order would be.

Whether the order exists, is the customer's, can be refunded and is being
refunded in full are facts about the record: the tool checks them, once, and the
refund's write is a conditional update. They used to be re-checked here too.

## Vocabulary

- **action class** — `refund` — the budget class refund tools are charged against
- **mandate** — permission scoped to the order under discussion, with a maximum amount

## Public interface

- `CustomerSupportBusinessRules(policy).evaluate(call, authority)` → `BusinessRuleResult` (`business_rules.py`)
- `EstablishedOrderScope` — mandates from the conversation (`mandates.py`)
- `REFUND_ACTION_CLASS`

## Configuration

`customer_support_agent.action_classes`; `gatekeeper.mandate.target_param: order_id` in this agent's overlay.

## Seams

- Injected into the platform `GatekeeperImpl` by `graph/builder.py`.

## Collaborators

- `security.auth.AgentPolicy` (the limit); `persistence.OrdersRepository` (mandates only).

## Episodes

E02, E03, E12, E33.

## Tests

Run with `./venv/bin/pytest tests/unit/test_gatekeeper.py tests/unit/test_gatekeeper_demo.py tests/unit/test_settlement_and_standings.py -q` (or `make test` for everything).
