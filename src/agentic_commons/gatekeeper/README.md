# gatekeeper — deterministic mediation

Decides whether a proposed action may run: authority (who is asking and
at what rank), the agent's permissions, cumulative budgets over a window,
surface policy, mandates established in the conversation, and the agent's
own business rules. Every decision is `allow`, `deny` or `escalate`, is
explained check by check, and is audited. No model is consulted for the verdict.

## Vocabulary

- **authority rank** — Sheridan-style levels of autonomy; a proposal needs the rank its action requires
- **cumulative ledger** — the structuring defence: individually-permitted actions that breach a window budget in aggregate
- **mandate** — what the conversation actually established (e.g. which order) — an action on anything else is refused
- **tier 2** — an optional semantic advisor that can flag, never allow

## Public interface

- `Gatekeeper` (Protocol), `GatekeeperImpl` (`gatekeeper.py`)
- `AuthorityHierarchyResolver`, `RoleAuthorizer` (`hierarchy.py`), `CumulativeLedger` (`ledger.py`)
- `DeterministicBusinessRules` (Protocol) with `AllowAllBusinessRules`; `BusinessRuleResult`
- `Tier2Advisor`, `SemanticTier2Advisor`, `NullTier2Advisor` (`tier2.py`); mandates (`mandate.py`); automation escalation (`automation.py`)
- `CheckReport`, `AuthorityContext`, `AuthorityDecision`, `Verdict` (`types.py`)

## Configuration

Section `gatekeeper:` — `authority_grant_max_rank`, `authority_hierarchy`, `ledger`, `mandate`,
`tier2`, `surface_policy`, `automation`. Whether it is built at all: `features.gatekeeper` (PRD G1
is retired — when withheld, a tool call executes with no authorization, but the audit entry is
still written and no budget is committed).

## Seams

- An agent supplies its business rules and mandate policy.

## Collaborators

- Reads `security.auth` policy and `persistence` budgets; writes `audit`; hands approved calls to `executor`.

## Episodes

E02, E03, E12, E31, E33, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_gatekeeper.py tests/integration/test_e12_attacks.py tests/integration/test_cross_subsystem_invariants.py -q` (or `make test` for everything).
