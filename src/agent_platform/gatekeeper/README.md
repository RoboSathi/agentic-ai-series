# gatekeeper — deterministic mediation

Decides one thing: whether the agent has the **authority** to run the exact call
the model proposed. Not whether the action is *correct* for the record — the tool
checks that, once, and its write is conditional — and not whether the *input* is
safe — screening does that. It reads trusted policy and tool metadata, calls no
model, touches no database per call, and audits every decision.

It asks, in order: is the agent enabled; was this tool offered this turn; does the
call target the record the customer established (mandate); does the agent's role
hold the tool's permission (RBAC); does the instruction's source rank allow it; is
it within the agent's own bounds (the agent's rules, e.g. a refund limit); and,
where a deployment has one, the cumulative budget. The first failure decides.

Every decision carries a **standing** — what the principal may do about it:

| Standing | When | Effect |
|---|---|---|
| VOID | denied (not offered, outside the mandate, no permission, …) | nothing runs, nothing is queued; nobody can adopt it |
| RATIFIABLE | escalated (beyond the agent's bounds) | queued for a person; approval is recorded RATIFIED |
| VOIDABLE | allowed, and the tool declares `voidable` | runs; undoable until it settles |
| VALID | allowed | runs |
| UNGOVERNED | the Gatekeeper is withheld | runs through the Executor; the audit row says so |

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
- `CheckReport`, `AuthorityContext`, `AuthorityDecision`, `Verdict`, `Standing` (`types.py`)

## Configuration

Section `gatekeeper:` — `authority_grant_max_rank`, `authority_hierarchy`, `ledger`, `mandate`,
`tier2`, `surface_policy`, `automation`. Whether it is built at all: `features.gatekeeper` (PRD G1
is retired — when withheld, a tool call executes with no authorization, still through the
Executor; the audit entry is written with standing UNGOVERNED and no budget is committed). A
deployment whose `ledger.enabled` is false has no cumulative check at all, rather than one
reported SKIPPED on every call.

## Seams

- An agent supplies its business rules and mandate policy.

## Collaborators

- Reads `security.auth` policy and `persistence` budgets; writes `audit`; hands approved calls to `executor`.

## Episodes

E02, E03, E12, E31, E33, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_gatekeeper.py tests/integration/test_e12_attacks.py tests/integration/test_cross_subsystem_invariants.py -q` (or `make test` for everything).
