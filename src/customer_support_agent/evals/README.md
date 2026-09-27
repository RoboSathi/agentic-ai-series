# evals — this agent's golden set, oracle and attacks

Everything agent-specific the platform eval pipeline needs, behind
`CsaEvalPlugin` (named in this agent's config as `evals.plugin`).

- **Golden set** — 50 cases in `dataset/golden_cases.yaml` across 12 categories, labelled from the written refund policy against a hand-built, time-stable fixture (`dataset/fixtures/eval_seed.sql`, orders 90001+, customers `EVAL001`–`EVAL005`).
- **Agent under test** — the real `build_agent`, run per case on a private copy of the fixture with its own audit ledger, checkpoints and episodic log; the transcript records replies, every tool call with its gatekeeper ruling, the evidence the model was shown, and the side effects left in the books.
- **Oracle** — the hard business invariants, checked deterministically from side effects: no unexpected refund, full amount only, the direct limit respected, ownership respected, no double refund, the outcome matches, forbidden tools never execute, numbers in the reply were shown to the agent, no other customer's personal data; plus soft signals (reason code surfaced, required tools requested, reply mentions, tool rounds).
- **Judge rubric** (`judge_rubric.yaml`) and **DeepEval metrics** (`quality_metrics.yaml`).
- **Security suite** (`security/`) — 21 curated Promptfoo attacks (injection, authority claims, cross-customer access, PII exfiltration, prompt extraction, jailbreaks, double refunds, encoded and multi-turn attacks, harmful content); assertions reuse the oracle's invariants.

## Vocabulary

- **outcome** — `refunded`, `queued_for_review`, `refused`, `answered`, `clarification`, `out_of_scope` — decided by side effects
- **side effects** — refunds paid, review requests made and order states changed relative to the fixture
- **label basis** — `written_policy` for every case (see `evals/provenance.py`)

## Public interface

- `CsaEvalPlugin` (`plugin.py`)
- `CsaAgentUnderTest`, `tool_calls_from`, `side_effects`, `record_evidence` (`agent_under_test.py`)
- `EvalFixture`, `FixtureFacts`, `policy_seed_sql` (`fixture.py`)
- `CsaGoldenCase` schema: `CsaCasePayload`, `CsaExpectation`, `load_golden_cases` (`dataset/schema.py`)
- `CsaOracle`, `INVARIANTS` (`oracle/`)
- Security assertions (`security/assertions.py`)

## Configuration

`customer_support_agent.evals` (paths, grounding roles, reply patterns, prompt-leak n-gram) and the
agent's `evals:` overlay (`plugin`, fingerprint roots and exclusions). The golden suite's gate
threshold is its manifest entry (`eval_manifest.yaml`, `suite: golden`).

## Seams

- Run through the platform: `make eval`, `make eval-generate`, `make eval-oracle`, … (see `agentic_commons/evals/README.md`).

## Collaborators

- `graph.build_agent`, `domain`, `persistence`, `prompts` (prompt-leak check).

## Episodes

E09–E11 applied to the running project.

## Tests

Run with `./venv/bin/pytest tests/unit/test_csa_evals.py tests/unit/test_eval_adapters.py tests/integration/test_eval_pipeline.py -q` (or `make test` for everything).

## Status and limits

`tests/unit/test_csa_evals.py` cross-checks every money-moving label against `RefundPolicy`, so a mislabelled case fails before any run.
