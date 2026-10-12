# durability — runs that survive failure

Checkpointed LangGraph runs (encrypted SQLite), resume without repeating committed side effects, HITL interrupts, isolated sub-graphs, and the Stop / Resume / Abort run controls every durable agent gets.

- **Entry points:** `create_checkpointer`, `DurableGraphRunner`, `RunRegistry`, `await_decision` (`hitl.py`), `IsolatedGraphRunner` (`isolated.py`), `run_actions` (`controls.py`)
- **Configuration:** `durability:`
- **Design:** [LLD §A11](../../../design_docs/LLD.md#a11-durability)
