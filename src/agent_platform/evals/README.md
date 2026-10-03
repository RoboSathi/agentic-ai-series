# evals — the eval framework

Two things live here.

1. **The eval pyramid and the NOLLM scoring cascade** (E09–E11): the six-tier
   cascade (exact → rule → embedding → NLI → LLM judge → human) that every
   subsystem resolves claims through, the checked-in eval manifest, judge
   calibration (Cohen's κ) and golden-label provenance.
2. **The eval pipeline** that runs an agent over its golden set and scores it:
   generate → oracle → security → judge → deepeval, persisted in DuckDB,
   with a detailed Markdown report per stage.

The platform supplies every mechanism and names no agent. An agent takes
part through one seam, `EvalPlugin`, named in its config (`evals.plugin`).

```
make eval             # every stage, in order; stops after the oracle on a broken invariant
make eval-generate    # run the agent over the golden set (reuses unchanged transcripts)
make eval-oracle      # deterministic business invariants — zero model calls
make eval-security    # Promptfoo attack suite
make eval-judge       # rubric quality + claim-level hallucination, one call per case
make eval-deepeval    # DeepEval metrics (optional extra: pip install -e '.[evals]')
make eval-report RUN=<id>   make eval-diff A=<id> B=<id>   make eval-runs   make eval-sql
```
Options: `RPM=<n>` (provider budget), `FRESH=1`, `LIMIT=<n>`, `CASE=<id>`, `TAG=<tag>`,
`RUN=<id>`, `EVAL_CONTINUE_ON_INVARIANT_FAILURE=1`.

## Vocabulary

- **suite** — a named set of cases (`golden`, `security`)
- **GoldenCase** — turns plus an agent-owned `payload` (fixture refs, expectations); hashed into `case_hash`
- **run** — one suite generated against one agent fingerprint and one dataset version
- **transcript** — everything the agent did for a case: replies, tool calls, evidence, side effects, usage
- **grader** — turns a case and transcript into `Score`s; kinds `oracle`, `judge`, `deepeval`, `promptfoo`
- **hard / soft** — hard metrics gate at 100% (one failure fails the run); soft ones at a pass rate
- **agent fingerprint** — content hash of the source tree, the effective config and the agent's inputs — uncommitted edits count, git does not decide
- **grader fingerprint** — names + versions of a stage's graders; scores are reused only while it and the transcript are unchanged

## Public interface

- Contracts (`contracts.py`): `GoldenCase`, `Transcript`, `ToolCallRecord`, `UsageRecord`, `Score`, `Rubric`, `QualityMetricSpec`; Protocols `EvalPlugin`, `AgentUnderTest`, `Grader`
- `resolve_eval_plugin()` (`plugin.py`), `compute_fingerprint()` (`fingerprint.py`)
- Pipeline (`pipeline/`): `GenerationStage`, `GradingStage`, `evaluate_gate`, `EvalPipeline` (one method per make target)
- Store (`store/`): `DuckDbEvalStore` — tables `eval_runs`, `eval_cases`, `eval_transcripts`, `eval_stages`, `eval_scores`; views `v_run_summary`, `v_metric_pass_rates`, `v_category_breakdown`, `v_failures`, `v_hard_failures`, `v_cost_latency`, `v_stage_cost`, `v_metric_trend`
- Graders (`graders/`): `RubricJudge` — rubric criteria + claim-level hallucination in one call
- Adapters (`adapters/`): DeepEval (`deepeval_graders`), Promptfoo (`PromptfooSuite`, the agent provider and the judge grader provider)
- Reports (`reporting/`): `EvalReporter` → `reports/evals/<run_id>/00_summary.md … 05_deepeval.md`, `diff_*.md`; Wilson intervals on every rate
- Cascade and pyramid (unchanged): `create_scoring_cascade`, `NollmScoringCascadeImpl`, `LlmJudge`, `JudgeCalibrator`, `load_manifest`, `GoldenDatasetProvenanceLedger`
- CLI: `python -m agent_platform.evals {generate,oracle,judge,deepeval,security,run,report,diff,runs,sql}`

## Configuration

Section `evals:` — cascade thresholds and toggles; `judge` (its own `model`, a different vendor
from the agent's, and `max_tokens` for a reasoning model); `plugin`; `suites`; `store.db_path`
(`data/evals.duckdb`, its own file because DuckDB admits one process per file); `generation.max_concurrency`;
`fingerprint` (source roots, excluded paths and config sections); `gate`
(`stop_on_invariant_failure`, `soft_pass_rate`); `deepeval`; `promptfoo` (pinned package, runner, env);
`reporting` (output dir, `confidence_z`, file names). The suite's soft threshold comes from its
manifest entry (`suite:` + `regression_threshold`) when declared.

## Seams

- An agent implements `EvalPlugin`: cases, a `prepare()` step, its `AgentUnderTest`, its oracle, its judge rubric, its quality metrics, its Promptfoo config, and fingerprint inputs.
- DeepEval and Promptfoo are never runtime dependencies: DeepEval is the optional `evals` extra, imported lazily by one adapter; Promptfoo is a Node tool run through `npx` at a pinned version. `tests/unit/test_platform_boundary.py` enforces both.

## Collaborators

- Every model call (agent, judge, DeepEval, Promptfoo's grader) goes through `llm` and the shared `resilience.rate_limiter` budget.
- `/eval` in the console reads the same store and views as the reports.

## Episodes

E09 (the pyramid and manifest), E10 (the NOLLM cascade), E11 (judge calibration and label provenance).

## Tests

Run with `./venv/bin/pytest tests/unit/test_evals.py tests/unit/test_eval_framework.py tests/unit/test_eval_adapters.py tests/integration/test_eval_pipeline.py -q` (or `make test` for everything).

## Status and limits

Generation is cached by (case hash, agent fingerprint) and resumable; grading by
(transcript hash, grader fingerprint). A grader that raises records an error score that is
never reused. The golden-task and production-shadow pyramid runners remain for E09's component
view; the golden suite itself runs through the pipeline above.
