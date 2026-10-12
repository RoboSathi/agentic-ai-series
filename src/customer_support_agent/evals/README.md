# evals — this agent's golden set, oracle, metrics and red team

Everything the platform pipeline needs behind `CsaEvalPlugin`: 50 golden cases (30 core) over a time-stable fixture, oracle rules graded from side effects, DeepEval metrics, a 27-attack Promptfoo suite (20 core), and the intent bake-off cases. Run with `make eval csa [oracle|deepeval|promptfoo] [full]`.

- **Entry points:** `CsaEvalPlugin` (`plugin.py`), `agent_under_test.py`, `fixture.py`, `dataset/`, `oracle/`, `deepeval/`, `promptfoo/`, `intent/`
- **Configuration:** `customer_support_agent.evals`, `evals:`
- **Design:** [evals.md §7](../../../design_docs/evals.md#7-customer-support-suite)
