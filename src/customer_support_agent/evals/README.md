# evals — this agent's golden set, oracle, metrics and red team

Everything agent-specific the platform eval pipeline needs, behind `CsaEvalPlugin` (named in
this agent's config as `evals.plugin`). Run with `make eval csa [STAGE=deepeval|promptfoo]`
(the intent bake-off: `make intent-eval csa`);
reports land in `reports/` as `{deepeval,promptfoo,intent}-<run_id>.md`.

- **Golden set** (`dataset/golden_cases.yaml`) — 50 cases in 16 categories, balanced between
  cases where the agent must act (refund, review, cancel) and must not (already refunded,
  issued but unsettled, declined, ineligible, not the customer's, partial, ambiguous). Each
  is labelled from the written policy against a hand-built, time-stable fixture
  (`dataset/fixtures/eval_seed.sql`, orders 90001+, customers `EVAL001`–`EVAL005`), and says
  in prose — for the DeepEval judge — the scenario and the outcome a good reply delivers.
- **Agent under test** — the real `build_agent`, per case on a private copy of the fixture;
  the transcript records replies, every tool call with its turn, gatekeeper ruling and
  **standing**, the evidence the model was shown, and the refunds, reviews and order-state
  changes left in the books.
- **Oracle** (`oracle/rules.py`, on the platform's `InvariantOracle`) — outcome rules, graded
  from side effects against the fixture: the outcome matches; no unexpected, partial,
  over-limit, foreign or double refund; no second review; an order's state moves only as
  required; the money-moving call stands as required (VOIDABLE / VALID / RATIFIABLE); a
  forbidden tool never executes; ids and amounts in the reply were shown to the agent; no
  other customer named; the refund limit never disclosed. Soft: reason code surfaced, the
  asked-for fact mentioned. No path rules.
- **DeepEval metrics** (`deepeval/metrics.yaml`) — task resolution, policy communication,
  honesty, tone (anchored `ConversationalGEval`), faithfulness, relevancy.
- **Red team** (`promptfoo/`) — 26 curated attacks in Promptfoo's taxonomy (rbac, bfla, bola,
  excessive-agency, pii, prompt-extraction, indirect-prompt-injection, harmful; strategies
  prompt-injection, jailbreak, base64, crescendo) plus one benign control that must be
  answered. `promptfooconfig.yaml` declares the provider (`provider.py`) and grader
  (`grader.py`); assertions reuse the oracle's rules.
- **Intent bake-off** (`intent/`) — the classifier contenders over 50 labelled messages.

## Vocabulary

- **outcome** — `refunded`, `queued_for_review`, `refused`, `answered`, `clarification`, `out_of_scope` — decided by side effects
- **standing** — how the money-moving call stood: VOIDABLE (refund), VALID (cancellation), RATIFIABLE (a person decides)
- **side effects** — refunds paid, reviews requested, order and refund states changed, relative to the fixture

## Public interface

- `CsaEvalPlugin` (`plugin.py`)
- `CsaAgentUnderTest`, `tool_calls_from`, `side_effects`, `record_evidence` (`agent_under_test.py`)
- `EvalFixture`, `FixtureFacts`, `policy_seed_sql` (`fixture.py`)
- `CsaCasePayload`, `CsaExpectation`, `load_golden_cases` (`dataset/schema.py`)
- `csa_oracle`, `RULES`, `OracleContext` (`oracle/`)

## Configuration

`customer_support_agent.evals` (paths, `chatbot_role`, grounding roles, reply patterns,
`money_tools`, prompt-leak n-gram, the bake-off) and the agent's `evals:` overlay (`plugin`,
`run_id_prefix: csa`, report folder, fingerprint roots and `agent_exclude_paths`).

## Tests

`./venv/bin/pytest tests/unit/test_csa_evals.py tests/unit/test_eval_promptfoo.py tests/integration/test_eval_pipeline.py -q` — offline.
`test_csa_evals.py` cross-checks every money-moving label against `RefundPolicy` and
`CancellationPolicy`, so a mislabelled case fails before any run.
