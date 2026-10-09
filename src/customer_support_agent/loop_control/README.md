# loop_control — this agent's two loops

The **tool-use loop** is the graph's `propose ⇄ execute` cycle, bounded by
`max_tool_iterations` and watched by the platform's `ToolLoopControl`. It is
deliberately not ReAct. The **output quality gate** (`quality_gate.py`)
looks a plan up in config, applies one deterministic critique, allows at
most one repair, and scores the result independently of the model that wrote it.

## Vocabulary

- **plan** — configured steps per intent (`customer_support_agent.plans`)
- **critique** — one deterministic check of a draft against its plan

## Public interface

- `OutputQualityGate`, `Planner`, `PlanLookup`, `GroundedCritic`, `IndependentScorer`, `Reflector`, `Evaluator` (`quality_gate.py`)
- Re-exports the platform's `ToolLoopControl`, `DuplicateActionDetector`, `action_fingerprint`

## Configuration

`customer_support_agent.max_tool_iterations`, `plans`, `deliberation_markers`.

## Collaborators

- `agent_platform.loop_control`, `evals` scoring cascade.

## Episodes

E20.

## Tests

Run with `./venv/bin/pytest tests/unit/test_unrequested_action_regression.py tests/unit/test_design_gaps.py -q` (or `make test` for everything).
