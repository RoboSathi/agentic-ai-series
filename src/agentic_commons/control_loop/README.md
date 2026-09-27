# control_loop — deterministic loop control

The data contracts and watchdogs that keep an agent's reason → act loop
bounded and honest: iteration ceilings, duplicate-action detection, legal
state transitions, and a report of every check. It is an import surface;
the loop itself is the agent's graph.

## Vocabulary

- **ToolControlLoop** — the per-turn watchdog over tool rounds
- **action fingerprint** — a stable hash of a proposed action, used to spot repeats
- **LoopVerdict** — continue, stop, or escalate, with the checks that decided it

## Public interface

- Types: `AgentState`, `Checkpoint`, `ControlCheck`, `ControlLoopReport`, `LoopDecision`, `LoopVerdict`, `ReactStep`, `StepResult`, `Trajectory`, `Transition` (`types.py`)
- Watchdogs: `DuplicateActionDetector`, `WatchdogLayer`, `ToolControlLoop`, `action_fingerprint` (`watchdogs.py`)
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
