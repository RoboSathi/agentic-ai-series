# loop_control — one authority over every agent's loop

`LoopGovernor` governs the step cap, budgets, deadline, repeat and cycle detection, convergence and a historical baseline, and answers with one `LoopVerdict` every agent routes on and draws.

- **Entry points:** `LoopGovernor`, `build_loop_governor`, `build_shared_loop_governor` (a budget its workers share), `LoopVerdict`, `TerminalOutcome`, `LoopState`, `BudgetGuard` (`budget.py`)
- **Configuration:** `loop_control:`
- **Design:** [LLD §A8](../../../design_docs/LLD.md#a8-loop_control)
