# judge — the LLM judge every agent and every eval grader builds on

A judge is a model asked to decide something about someone else's output. What it is asked
differs per judge; *how* it is asked must not, or each judge grows its own JSON parsing,
its own error handling and its own idea of "could not decide". The skeleton lives here.

## Vocabulary

- **judge model** — `judge:` in config: the strongest model on the account (ultra 550B), which
  no agent runs on (agents run on lightning), so a judge never grades its own model's output.
  When it does not answer, or has not after `hedge_after_seconds`, super-120b starts beside it;
  lightning is never a judge backup
- **verdict** — the caller's own type: what the judge's validated answer means to it
- **abstain** — the judge could not answer (every model down, or an answer that failed its
  schema after the guardrails' bounded repair); never a silent pass or fail
- **grading rules** — `prompts/judge_rules.md`, prepended to every platform judge call:
  evidence only, reason before verdict, quote what earns a pass, say so when it cannot decide

## Public interface

- `create_judge_client()` — the judge model and its backups, through `ModelChainBuilder`
- `LlmJudge[Request, Answer, Verdict]` — Template Method: `judge(request)` calls the
  `_answer` hook, then `verdict`; any failure goes to `abstain`; `models_used()` says who judged
- `ModelJudge` — the common case: implement `render(request) -> messages`; the platform judge
  model, the grading rules and validation are supplied
- `ask_validated(client, messages, schema)` / `ask_text(client, prompt)` — one judge call,
  for a framework with its own loop (DeepEval's model adapter)
- `JudgeCalibrator` — Cohen's κ against a second rater (a person, or the runtime guard)

## Who builds on it

- `evals/deepeval/model.py` — DeepEval's metrics ask through `ask_validated` / `ask_text`
- `evals/promptfoo/grader_provider.py` — `PromptfooRubricJudge(ModelJudge)` for `llm-rubric`
- An agent's own judge subclasses `LlmJudge` and answers through its own model router
  (role routing, budget, trace card) by implementing `_answer`
- `evals/deepeval/report.py` — κ between the judge's faithfulness verdicts and the runtime guard
- The scoring cascade's tier-5 seam (`scoring.CalibratedJudge`) is owned by its consumer;
  a judge built here that answers that shape can be wired in

## Configuration

`judge:` — `provider`, `model`, `fallbacks`, `timeout_seconds`, `hedge_after_seconds`,
`temperature`, `max_tokens`, `max_evidence_chars`, `validation_mode`, `kappa_acceptable`.

## Tests

`./venv/bin/pytest tests/unit/test_judge.py -q`
