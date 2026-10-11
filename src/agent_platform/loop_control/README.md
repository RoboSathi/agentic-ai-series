# loop_control — one authority over every agent's loop

`LoopGovernor` is the single runtime that governs a tool loop's lifecycle for every agent:
the step cap, the budgets (tool calls, tokens, cost, wall clock), a deadline that cancels a
call still running, repeat detection (exact, canonical, semantic), cycle detection,
convergence (progress per step) and a historical baseline (how long this agent's turns
usually are). It answers with one contract, `LoopVerdict`; routers read only its decision
(`route_on_verdict`), and every agent draws the same LOOP CONTROL card from it
(`StepEvents.loop_verdict`).

## Vocabulary

- **LoopVerdict** — `continue | suppress | finalize | abort | pause`, the mechanism (`layer`)
  that decided, its reason, every budget line and every check
- **TerminalOutcome** — how a turn ended: `answered | finalized_on_cap | aborted | cancelled |
  escalated | failed`; `RUN_STATUS` and `TOOL_STAGE` map it once onto runs and tool stages
- **LoopState** — the governor's whole memory (phase, counters, seconds used, actions and
  their outcomes, evidence gains, last verdict); agents keep it in their checkpointed state
- **Stagnation tier** — `exact` → `canonical` (normalised arguments) → `semantic`
  (similar to a failed step); `cycle` catches a repeating block of attempts
- **Convergence** — the new evidence each step adds; a plateau finalizes

## Lifecycle

`begin_turn` → per step `begin_step` (verdict) → `call_limits` / `deadline` →
`after_model` → per action `before_action` (verdict) → `after_action` → `finish(outcome)`.
`restore(state)` continues a checkpointed run, seconds included.

## Public interface

- `governor.py` — `LoopGovernor`, `LoopBudget`, `CallLimits`, `route_on_verdict`
- `factory.py` — `build_loop_governor(budget, facts=, agent_id=)` (composition root only)
- `types.py` — `LoopVerdict`, `ControlCheck`, `BudgetLine`, `TerminalOutcome`, `LoopPhase`
- `state.py` — `LoopState`, `LoopStateMachine` (legal phase transitions)
- `stagnation.py` — fingerprints, `SemanticTier`, `CycleDetector`, `HistoricalBaseline`
- `convergence.py` — `ConvergenceMonitor`, `ConvergenceSignal`
- `budget.py` — `BudgetGuard` (reserve → commit/release for parallel workers)

## Configuration

`loop_control.*` (platform): call timeout, stagnation, convergence, historical baseline,
suppression messages, reason sentences, card wording. Caps are each agent's own config,
mapped onto `LoopBudget` in its composition root.

## Tests

`tests/agent_platform/test_loop_governor.py`, `test_budget.py`,
`tests/unit/test_unrequested_action_regression.py`, the agents' loop tests.

## Status and limits

Always on. A long-running loop of parallel workers still budgets
through `BudgetGuard` and its own stop rules; it reports in the same `LoopVerdict` card.
