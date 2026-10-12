# Evals

How each agent is evaluated: the platform's three graders, the shared judge, and each agent's
suite. Decisions: ADR P-28 (graders), P-29 (offline tests), C-06 (intent bake-off), R-10 (the
research agent has no suite yet). Component design: [LLD §A25](LLD.md#a25-evals).

## 1. Three graders, each the right tool for one question

| Grader | Kind | Grades | Gate |
|---|---|---|---|
| **Oracle** | code, zero model calls | the *outcome*: the state a case left behind, against ground truth | hard — 100% |
| **DeepEval** | model-graded, the platform judge | the conversation: one anchored quality dimension per metric | soft — `evals.gate.soft_pass_rate` |
| **Promptfoo** | adversarial | curated red-team attacks; every assertion must hold | hard — 100% |

Plus the **intent bake-off**: an agent's intent classifiers scored side by side (accuracy,
macro-F1, coverage at threshold, Brier, ECE, latency) — a component eval, offline.

- **The oracle is the only grader that reads ground truth.** "Did money move that shouldn't
  have?" is a fact about the database; an LLM can be talked into anything a conversation can
  say. It grades what the agent *produced*, not the path it took — the one path rule is a
  forbidden tool that *executed*.
- **DeepEval grades what only language can show**: did the reply resolve the request, explain a
  refusal, claim only what happened, stay grounded. `ConversationalGEval` with evaluation steps
  and rubric bands, judged against each case's own scenario and expected outcome.
- **Promptfoo holds the agent to the same invariants under attack**: its assertions reuse the
  oracle's rules (`oracle_assert`), so "unsafe" is defined once. Each attack names its plugin,
  strategy and severity in Promptfoo's red-team taxonomy; the set includes one benign control
  that must be answered, so refusing everything does not pass.

## 2. Running

```
make eval <agent>              oracle, DeepEval, then Promptfoo — over the core set
make eval <agent> oracle       golden cases → business invariants (no judge calls)
make eval <agent> deepeval     golden cases → LLM-judged conversation quality
make eval <agent> promptfoo    red-team attacks
… full                         add to any of the above: every case, attack and metric
make intent-eval <agent>       intent classifier bake-off (offline)
RPM=38 make eval csa           override the provider budget for this run
```

- Each section prints a colour-coded overview and writes
  `reports/<package>/evals/<section>-<run_id>.md` — every case with its result, every failed
  check with its reason — where `run_id = <agent>-<suite>-<hash>-<YYYYMMDD-HHMMSS>` (UTC).
- Every section always runs; a failed gate is a result. `make eval` exits non-zero only when a
  section could not run at all. The run ends with the time each section took.
- One progress bar per section; warnings only on the console (a 429 names the model it hit, a
  fallback names the model that failed and its backup); the log files keep everything.
- Ctrl+C marks the run interrupted and exits 130; the next run resumes from what finished.
- The oracle and DeepEval grade the same golden transcripts, so the agent runs once for both,
  and not at all while its fingerprint is unchanged.

## 3. Core and full

A default run covers the agent's **core** cases, attacks and metrics (`evals.core`): every
behaviour once or twice, in minutes, because the provider budget — not the code — sets the pace.
A case is core when the core tag is in its `tags`; an attack when its metadata says
`tier: core`; a metric when it says `core: true`. `full` runs everything; transcripts are cached
per case, so a full run after a core run generates only the rest.

## 4. Caching and trials

- **Agent fingerprint** — a content hash of the source tree, effective config and the agent's
  inputs (minus `agent_exclude_paths`); any change regenerates by itself. Generation is cached by
  (case hash, agent fingerprint).
- **Grader fingerprint** — names and versions of a stage's graders; scores are reused while it
  and the transcript are unchanged. A grader that raises records `<metric>.error`, never reused.
- **Trials** — `evals.trials = k` runs each case k times; the report shows pass@k (some trial
  clean) and pass^k (every trial clean).

## 5. The shared judge

Every model-graded verdict asks the `judge:` model — the strongest model on the account, backed
by one that is never the agents' primary, so a judge never grades its own model's output. The
judge runs with thinking off (the same verdict in a fraction of the time on an explicit rubric).
`LlmJudge` abstains when it cannot answer; an abstention is never a silent pass or fail.
`JudgeCalibrator` reports Cohen's κ between the judge's faithfulness verdicts and the runtime
hallucination guard.

## 6. Store

`db/agent_platform/evals.duckdb` (its own file: a running console holds the observability
DuckDB). Tables `eval_runs`, `eval_cases`, `eval_transcripts`, `eval_stages`, `eval_scores`;
views `v_scores`, `v_run_summary`, `v_metric_pass_rates`, `v_failures`, `v_stage_cost`,
`v_metric_trend`. `/eval` in the console and `python -m agent_platform.evals report | runs` read
the same store.

## 7. Customer Support suite

Behind `CsaEvalPlugin` (`src/customer_support_agent/evals/`).

| | core (default, about 12 min) | full (about 35–40 min) |
|---|---|---|
| Golden cases | 30 — every category, money-moving ones 2–3× | 50 in 16 categories |
| Attacks | 20 — every critical one, every plugin and strategy, the control | 27 |
| DeepEval metrics | task_resolution, honesty, faithfulness | + policy_communication, tone, relevancy |

- **Golden set** (`dataset/golden_cases.yaml`) — balanced between cases where the agent must act
  (refund, review, cancel) and must not (already refunded, issued but unsettled, declined,
  ineligible, not the customer's, partial, ambiguous). Each is labelled from the written policy
  against a hand-built, time-stable fixture (`dataset/fixtures/eval_seed.sql`: customers
  `EVAL001`–`EVAL005`, orders 90001+), and says in prose — for the judge — the scenario and the
  outcome a good reply delivers. `test_csa_evals.py` cross-checks every money-moving label against
  `RefundPolicy` and `CancellationPolicy`, so a mislabelled case fails before any run.
- **Agent under test** — the real `build_agent`, per case on a private copy of the fixture
  (`db/customer_support_agent/eval_fixture.db`, workspaces under
  `data/customer_support_agent/evals/`). The transcript records replies, every tool call with its
  turn, Gatekeeper ruling and standing, the evidence the model was shown, and the refunds, reviews
  and order-state changes left in the books.
- **Oracle rules** (`oracle/rules.py`) — the outcome matches; no unexpected, partial, over-limit,
  foreign or double refund; no second review; an order's state moves only as required; the
  money-moving call stands as required (VOIDABLE / VALID / RATIFIABLE); a forbidden tool never
  executes; ids and amounts in the reply were shown to the agent; no other customer named; the
  refund limit never disclosed. Soft: reason code surfaced, the asked-for fact mentioned.
- **Outcomes** — `refunded`, `queued_for_review`, `refused`, `answered`, `clarification`,
  `out_of_scope`, decided by side effects.
- **Red team** (`promptfoo/`) — Promptfoo's taxonomy: plugins rbac, bfla, bola, excessive-agency,
  pii, prompt-extraction, indirect-prompt-injection, harmful; strategies prompt-injection,
  jailbreak, base64, crescendo; plus one benign control.
- **Intent bake-off** (`intent/cases.yaml`) — 50 labelled messages; Laya and BERT, each at its
  own threshold, with and without phrase rules (results in ADR C-06).

## 8. Research agent

No suite. The fixture-world suite (golden cases, planted flaws, ablations, reference judge) was
removed with the world (ADR R-10). `make eval ra` says so. The research agent's machinery is
covered by unit and flow tests with scripted models (`tests/research_agent/`); its independent
audit is a runtime evaluator-optimizer, not an offline eval. A suite over live or recorded web
data is owed.

## 9. Ops agent

No suite. Owed: an NL-to-SQL golden set scored by execution accuracy, run through the platform
pipeline rather than a parallel runner (ADR P-25).

## 10. Tests are not evals

`make test` runs the unit and integration suite, offline (ADR P-29): models are scripted,
databases are in memory, NLI is the lexical model, provider keys and hosted guards are removed by
`tests/conftest.py`. Evals are the only place an agent meets a real model on purpose.

| Level | Scope | Where |
|---|---|---|
| Unit | one module, I/O boundaries scripted | `tests/agent_platform/`, `tests/<agent>/` |
| Integration | several real modules composed; cross-subsystem invariants | `tests/integration/` |
| Guards | layout, platform boundary, agent layering, config surface | `tests/agent_platform/test_package_layout.py`, `test_platform_boundary.py`, `test_agent_layering.py`; `tests/customer_support_agent/test_config_surface.py` |
| Eval | golden outcomes, conversation quality, attacks | `make eval <agent>` |
