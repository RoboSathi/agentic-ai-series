# loop_control — deterministic loop control

The watchdogs that keep an agent's tool rounds bounded and honest: the step
cap, duplicate-action detection, run budgets (`BudgetGuard`), and a report of
every check (the LOOP CONTROL card). It is an import surface; the loop itself
is the agent's graph.

It is *loop control*, not a control loop: nothing here drives the agent's whole
flow. That control loop (an explicit FSM over the run, LLD §15) is **not
built**; `AgentState`, `Transition`, `Trajectory`, `Checkpoint` and the
`ReactStep`/`LoopController`/`StateMachine`/`CheckpointStore` protocols are its
parked contracts, with no implementation and no caller.

## Vocabulary

- **ToolLoopControl** — the per-turn watchdog over tool rounds
- **action fingerprint** — a stable hash of a proposed action, used to spot repeats
- **LoopVerdict** — continue, stop, or escalate, with the checks that decided it

## Public interface

- Types: `AgentState`, `Checkpoint`, `ControlCheck`, `LoopControlReport`, `LoopDecision`, `LoopVerdict`, `ReactStep`, `StepResult`, `Trajectory`, `Transition` (`types.py`)
- Watchdogs: `DuplicateActionDetector`, `WatchdogLayer`, `ToolLoopControl`, `action_fingerprint` (`watchdogs.py`)
- Protocols: `LoopController`, `StateMachine`, `CheckpointStore` (`protocols.py`)

## Configuration

No section of its own; the iteration ceiling belongs to the agent's own config.

## Seams

- An agent composes these into its own loop.

## Collaborators

- Used by the agent's `act` and `reason` nodes; its report surfaces in the TUI feed.

## Episodes

E20, E23.

## Tests

Covered through the agent's loop tests (`tests/unit/test_unrequested_action_regression.py`, `tests/unit/test_design_gaps.py`).

## Status and limits

Always on: removing loop control produces a broken program, not a riskier agent.
