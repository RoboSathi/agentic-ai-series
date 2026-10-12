# ADR 0013 — Evals as three graders, one shared judge, one make target

- Status: accepted; implemented 2026-10-12
- Date: 2026-10-12
- Related: the 2026-09-26 evals framework plan (superseded in layout, kept in its store and
  caching), ADR 0011 (the intent bake-off), ra-0019 (the research agent has no suite yet),
  ra-0024 (the research agent's model line-up)

## Context

- The eval framework was built on 2026-09-27 and had not run since. The agent had moved on:
  standings, `issued` refunds that settle, `cancel_order`, one review queue. The golden set
  still expected a reason code no code path produced (`REVIEW_ALREADY_PRESENT`), never touched
  `issued`, `denied`, cancellations or listing, and the transcript dropped the standing.
- Four paths graded with a model, overlapping: a custom `RubricJudge` (every criterion and
  every claim in one call, 1–5 with no anchors), DeepEval (two metrics, called metric by
  metric), Promptfoo's `llm-rubric`, and a never-constructed `LlmJudge`. The research agent's
  Independent Audit had a fifth judge of its own.
- The eval judge's chain shared models with the agents it graded.
- Runtime grounding code (the scoring cascade, entailment) lived in `evals/`, imported by
  about twenty runtime modules; `pyramid.py`, the manifest and the provenance ledger had no
  production caller.
- Eleven make targets and seven knobs; a report per stage in a folder per run.

## Decision

### Three graders, one per kind of question (Anthropic's grader taxonomy)

| Grader | Kind | Grades | Gate |
|---|---|---|---|
| Oracle | code | the outcome in the database, against ground truth | hard, 100% |
| DeepEval | model (platform judge) | conversation quality, one anchored dimension per metric | soft |
| Promptfoo | adversarial | curated red-team attacks; assertions reuse the oracle's rules | hard, 100% |

- **The oracle grades outcomes, not paths.** `required_tools_requested` and
  `tool_rounds_within_budget` are gone; the one tool rule left is a forbidden tool that
  *executed*. New rules: `standing_matches`, `order_state_matches`, `no_duplicate_review`,
  `reply_omits` (the refund limit is never disclosed). It is generic in the platform
  (`InvariantOracle`); an agent supplies rules and their context.
- **Every section always runs** (amended 2026-10-12, Karan): a broken invariant no longer skips
  DeepEval, and a failed gate is a result, not a process error — `make eval` exits non-zero
  only when a section could not run (the others still run). The run ends with each section's
  time (agent runs, oracle, DeepEval, Promptfoo) and the total.
- **DeepEval the standard way**: a `ConversationalTestCase` per transcript (turns carry the
  tools called in them and the evidence), `deepeval.evaluate()`, `ConversationalGEval` with
  evaluation steps and rubric bands, `TurnFaithfulness`, `TurnRelevancy`. The custom
  `RubricJudge` is deleted.
- **Promptfoo the standard way**: the agent's `promptfooconfig.yaml` declares its provider,
  its grader and its tests; the platform runs `promptfoo eval -c` and ingests the JSON. Each
  attack names its plugin, strategy and severity in Promptfoo's red-team taxonomy; the set
  stays curated (a fixed set is a regression signal) and includes one benign control that
  must be answered, so refusing everything does not pass.
- **Balanced golden set**: about as many cases where the agent must act as where it must not.
- **Trials**: `evals.trials` = k runs each case k times; the report shows pass@k and pass^k.

### One shared judge (`agent_platform/judge`)

- `judge:` is a top-level config section: Nemotron Ultra 550B, backed by glm-5.3-flash then
  gpt-oss-20b (the only other models this account serves that answered a probe, 2026-10-12).
- **Every agent runs on lightning; ultra is the judge's** (Karan, 2026-10-12). Ultra is removed
  from `llm.chain`; lightning is not a judge backup. A judge never grades its own model.
- `LlmJudge` is a Template Method — `_answer` (hook) → `verdict`; any failure → `abstain`, never
  a silent zero. `ModelJudge` adds rendered messages on the platform judge with the shared
  grading rules (`prompts/judge_rules.md`) and the guardrails' bounded repair.
- Users: DeepEval's model adapter, Promptfoo's grader, and the research agent's Independent
  Audit judge, which subclasses `LlmJudge` and answers through its own `StructuredCaller`.

### Layout and surface

- `agent_platform/evals/{core,oracle,deepeval,promptfoo,intent}`; the scoring cascade moves to
  `agent_platform/scoring` (a leaf package — `hallucination_check` sits in an import cycle).
- `make eval <agent> [STAGE=deepeval|promptfoo]`, and `make intent-eval <agent>` for the
  intent bake-off, are the only targets. No `CASE`/`FRESH`/`LIMIT`/`RUN` knobs: the content fingerprint
  regenerates whatever changed.
- One report per framework per run in `src/<pkg>/evals/reports/`:
  `{deepeval,promptfoo,intent}-<run_id>.md`, `run_id = <agent>-<suite>-<hash>-<YYYYMMDD-HHMMSS>`.
  Each stage also prints a colour-coded overview through the console's `CommandView`.

## Consequences

- Deleted: `pyramid.py`, `manifest.py` and both eval manifests, `provenance.py`'s ledger,
  `graders/`, `judge.py`'s `LlmJudge`, the `diff` and `sql` CLI commands, `/eval manifest`,
  every `eval-*` target but `eval`, `intent-bakeoff`.
- The research agent's writer roles move from ultra to lightning (thinking off); report
  quality must be watched (ra-0024).
- Cascade tier 5 still declines (no judge is wired into it); the seam (`scoring.CalibratedJudge`)
  is unchanged.
- Every hosted model has fallbacks (amended 2026-10-12): the agents' chain lightning →
  gpt-oss-20b → glm-5.3-flash; the jailbreak guard NemoGuard → glm-5.3-flash → gpt-oss-20b and
  the content-safety guard nemotron-3.5-content-safety → gpt-oss-20b → glm-5.3-flash, each
  fallback a chat model prompted as the classifier (`security.guard_chain.GuardChain`); the
  fail-closed policy applies only when every one is unavailable. Embeddings have no
  cross-model fallback: another model's vectors do not match the stored index.
- DeepEval's test-run manager is process-wide, so `evaluate()` runs one case at a time per
  process; the metrics of a case still run concurrently.
