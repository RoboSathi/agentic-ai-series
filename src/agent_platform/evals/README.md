# evals — three graders over an agent's suites

An agent's evals are graded three ways, each the right tool for one kind of question
(Anthropic's grader taxonomy: code-based, model-based, adversarial):

| Grader | Kind | Grades | Gate |
|---|---|---|---|
| **Oracle** (`oracle/`) | code, zero model calls | the *outcome*: the state a case left behind, against ground truth | hard — 100% |
| **DeepEval** (`deepeval/`) | model-graded, the platform judge | the conversation: one anchored quality dimension per metric | soft — `gate.soft_pass_rate` |
| **Promptfoo** (`promptfoo/`) | adversarial | curated red-team attacks; every assertion must hold | hard — 100% |

Plus the **intent bake-off** (`intent/`): an agent's intent classifiers scored side by side
(accuracy, macro-F1, coverage at the threshold, Brier/ECE, latency) — a component eval, offline.

The platform supplies every mechanism and names no agent. An agent takes part through one
seam, `EvalPlugin`, named in its config (`evals.plugin`).

```
make eval <agent>              oracle, DeepEval, then Promptfoo — over the core set
make eval <agent> oracle       golden cases → the agent's business invariants (no judge calls)
make eval <agent> deepeval     golden cases → LLM-judged conversation quality
make eval <agent> promptfoo    red-team attacks
… full                         add to any of the above: every case, attack and metric
make intent-eval <agent>       intent classifier bake-off (offline)
```

Each **section** is one grader with its own gate and its own report. The oracle and DeepEval
grade the same golden transcripts, so the agent runs once for both (and not at all while its
fingerprint is unchanged); asking for one runs only that one. A full run runs every section —
a broken invariant is a finding to read beside the quality scores, not a reason to skip them —
and a failed gate is a result, not an error: `make eval` exits non-zero only when a section
could not run at all (the others still run). Each section prints a colour-coded overview (box
tables, outcome words in the console's state colours) and writes
`src/<pkg>/evals/reports/<section>-<run_id>.md` — every case with its result, and every failed
check with its reason — where `run_id = <agent>-<suite>-<hash>-<YYYYMMDD-HHMMSS>` (UTC). The run
ends with the time each section took.

**Core and full.** A default run covers the agent's *core* cases, attacks and metrics
(`evals.core`): every behaviour once or twice, in minutes rather than the better part of an
hour, because the provider budget — not the code — sets the pace. An agent marks a case core
with the tag in its `tags`, an attack with `<attack_metadata_key>: <tag>` in its Promptfoo
metadata (run with Promptfoo's own `--filter-metadata`), a metric with `core: true`. `full`
runs them all; an agent that marks nothing core always runs everything. Core and full are
different case sets, so different runs — but transcripts are cached per case, so a full run
after a core one generates only the rest.

While it runs, `make eval` shows one progress bar per section (agent runs, oracle,
DeepEval, Promptfoo) with the percentage done and its time; the console shows warnings only
(`evals.console_level`) — a 429 names the model it hit, a model fallback names the model that
failed, its HTTP status and the backup — while the log files keep everything. Ctrl+C marks the
run interrupted, prints the time so far and exits 130; the next run resumes from what finished.
DeepEval grades `evals.deepeval.batch_size` cases per `evaluate()` call, the cases and their
metrics concurrent inside it, and the judge runs with thinking off.

## Why each grader exists

- **The oracle is the only grader that reads ground truth.** The judges read the
  conversation; "did money move that shouldn't have?" is a fact about the database, and an
  LLM can be talked into anything a conversation can say. It is free and repeatable, and runs
  first; DeepEval still runs after it whatever it found. It grades what the agent *produced*, not the path it
  took — the one path rule is a forbidden tool that *executed*, because executing is an outcome.
- **DeepEval grades what only language can show**: did the reply resolve the request, explain
  a refusal, claim only what happened, stay grounded in its evidence. One dimension per metric
  (`ConversationalGEval` with evaluation steps and rubric bands — faithfulness among them, one
  judge call per case where `TurnFaithfulness` makes ~4 per reply; `TurnRelevancy`), each
  judged against the case's own scenario and expected outcome.
- **Promptfoo holds the agent to the same invariants under attack.** Its assertions reuse the
  oracle's rules (`promptfoo/assertions.py: oracle_assert`), so "unsafe" is defined once.

## Layout

```
core/       contracts (EvalPlugin, GoldenCase, Transcript, Score), fingerprint, store (DuckDB),
            pipeline (generate, grade, gate, trials, orchestrator), reporting (sections, writer,
            overview), cli
oracle/     InvariantOracle — runs an agent's Rule list; versioned by rules + inputs
deepeval/   model (the platform judge as a DeepEvalBaseLLM), test_cases (Transcript →
            ConversationalTestCase), metrics (spec → metric), grader (deepeval.evaluate()),
            report + template
promptfoo/  runner (npx promptfoo eval -c <agent config>, ingest), agent_provider,
            grader_provider (llm-rubric via the platform judge), assertions, report + template
intent/     scoring (accuracy, macro-F1, Brier, ECE, latency), report
```

## Vocabulary

- **suite** — `golden` (graded by the oracle and DeepEval) or `security` (Promptfoo)
- **section** — one grader over a suite, with its own gate and report: `oracle`, `deepeval`, `promptfoo` (plus `intent`)
- **core / full** — the default subset of cases, attacks and metrics (`evals.core`), or all of them
- **GoldenCase** — turns plus an agent-owned `payload`; hashed into `case_hash`
- **run** — one suite generated against one agent fingerprint and one case set
- **transcript** — replies, tool calls (turn, gatekeeper ruling, standing, result), evidence, side effects, usage
- **hard / soft** — hard metrics gate at 100%; soft ones at a pass rate
- **trial** — with `evals.trials` = k each case runs k times; the report shows pass@k (some trial clean) and pass^k (every trial clean)
- **agent fingerprint** — content hash of the source tree, effective config and the agent's inputs; any change regenerates by itself
- **grader fingerprint** — names + versions of a stage's graders; scores are reused while it and the transcript are unchanged

## Public interface

- `EvalPlugin` (agent_id, load_cases, prepare, agent_under_test, oracle, quality_metrics, conversation_frame, security_config, fingerprint_inputs); optional `IntentBakeoffPlugin`
- `InvariantOracle`, `Rule`, `RuleVerdict` (`oracle/`)
- `DeepEvalGrader` (`deepeval/`); `PromptfooSuite` (`promptfoo/`); `BakeoffResult` (`intent/`)
- `EvalPipeline.golden(sections, full, sink) → [StageOutcome]` (one per section, each handed to `sink` as it finishes) / `security(full)` / `intent()` → `StageOutcome` (section, gate, report path, overview); `report(run_id) → [StageOutcome]`; `pipeline.clock` times each section; `EvalProgress` (`core/progress.py`) is how stages report progress, `ProgressBar` draws it
- `BatchGrader` — a grader handed transcripts a batch at a time (DeepEval)
- `DuckDbEvalStore` — tables `eval_runs`, `eval_cases`, `eval_transcripts`, `eval_stages`, `eval_scores`; views `v_scores`, `v_run_summary`, `v_metric_pass_rates`, `v_failures`, `v_stage_cost`, `v_metric_trend`, …
- CLI: `python -m agent_platform.evals run [--section all|oracle|deepeval|promptfoo|intent] [--full] | report [--run ID] | runs`

## Configuration

`judge:` — the model every model-graded verdict uses (see `agent_platform/judge`).
`evals:` — `plugin`, `run_id_prefix`, `suites`, `core` (`tag`, `attack_metadata_key`), `trials`, `store`, `generation.max_concurrency`,
`fingerprint` (roots, `exclude_paths` + an agent's `agent_exclude_paths`), `gate` (`soft_pass_rate`), `deepeval`
(`include_reason`, `max_concurrent`, `batch_size`, `geval_params`, `quiet`), `promptfoo` (pinned package,
runner, taxonomy keys), `reporting` (output dir, file names, Wilson `confidence_z`, the overview
and its section names).

## Seams

- DeepEval and Promptfoo are never runtime dependencies: DeepEval is the optional `evals`
  extra, imported lazily inside `deepeval/`; Promptfoo is a Node tool run through `npx` at a
  pinned version. `tests/unit/test_platform_boundary.py` enforces both.
- DeepEval keeps its test run in a process-wide manager, so `evaluate()` runs one case at a
  time per process (the metrics of that case run concurrently), in a worker thread.

## Collaborators

- `agent_platform.judge` — every model-graded verdict (DeepEval metrics, Promptfoo's `llm-rubric`).
- `agent_platform.presentation` — the overview is drawn by the console's `CommandView`.
- Every model call goes through `llm` and the shared `resilience.rate_limiter` budget.
- `/eval` in the console reads the same store and views as the reports.

## Tests

`./venv/bin/pytest tests/unit/test_eval_core.py tests/unit/test_eval_progress.py tests/unit/test_eval_deepeval.py tests/unit/test_eval_promptfoo.py tests/unit/test_eval_cli.py tests/integration/test_eval_pipeline.py -q` — all offline (scripted agent and judge).

## Status and limits

Generation is cached by (case hash, agent fingerprint) and resumable; grading by (transcript
hash, grader fingerprint). A grader that raises records `<metric>.error`, never reused.
An agent with no `evals.plugin` has no suite; `make eval` refuses it.
