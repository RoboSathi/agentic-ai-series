# evals — three graders over an agent's suites

Oracle (code, outcomes vs ground truth), DeepEval (the platform judge, conversation quality) and Promptfoo (red team), plus the intent bake-off. An agent joins through one seam, `EvalPlugin`.

- **Entry points:** `EvalPlugin`, `EvalPipeline`, `InvariantOracle`, `DeepEvalGrader`, `PromptfooSuite`, `DuckDbEvalStore`, `python -m agent_platform.evals`
- **Configuration:** `evals:`, `judge:`
- **Design:** [LLD §A25](../../../design_docs/LLD.md#a25-evals) · [evals.md](../../../design_docs/evals.md)
