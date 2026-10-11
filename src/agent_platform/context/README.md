# context — what the model is handed

Assembles the context window for a turn: which candidates enter, how they
are ranked, what is reduced, offloaded or isolated to stay inside the token
budget, and whether the assembled context shows a known failure mode
(poisoning, distraction, confusion, clash).

## Vocabulary

- **Chunk** — one candidate piece of context with a kind and provenance
- **strategy** — consolidate, isolate, cache (always), then retrieve, offload, reduce (while over
  budget) — run by `StrategySelector` inside the pipeline's fit stage, before anything is cut
- **five-stage pipeline** — gather → score → fit (strategies, then cut) → format → order
- **trust** — a chunk's provenance class (`system`, `verified`, `retrieved`, `user`, `model`): who wins a clash
- **offload handle** — where an offloaded chunk's text waits (session working memory), fetched
  back by the `fetch_offloaded_context` tool
- **context failure** — poisoning (injection, or an unverified claim a verified source
  contradicts), clash (`key: value` and prose, by local NLI), distraction (similarity to the
  task), confusion (near-copies; too many tools), rot (answer quality falling as the window
  grows), diagnosed and mitigated before the model call

## Public interface

- `ContextAssembler` / `FiveStagePipeline` (`pipeline.py`)
- Retrievers: `LexicalRetriever` (BM25), `DenseRetriever`, `HybridRetriever`, `GraphRetriever`; `CrossEncoderReranker`
- Strategies (`strategies.py`) and `StrategySelector`; `build_context_management` (`factory.py`)
- `ContextFailureDetector`, `RotMonitor` (`failure/`: one module per mode)
- `TextSimilarity` (`similarity.py`), trust (`trust.py`), offload store and tool (`offload.py`)
- The request a model is sent is built from named sections by `llm/prompt.py`
  (`PromptBuilder`), and drawn as the CONTEXT card (`observability/context_card.py`)
- `CandidateSource` (Protocol) — anything that contributes chunks, e.g. `WorkingMemoryContextSource`

## Configuration

Section `context:` — `token_budget`, `chars_per_token`, `retrieval_mode`, `top_k`,
`hybrid_lexical_weight`, `rerank`, `bm25_k1`/`bm25_b`, `strategies`, `strategy_notes`, `similarity`, `consolidation`, `isolation`, `offload`, `reduce`, `trust`, `findings`,
`failure_detection`. Whether the stage exists: `features.context` (+ `rerank`, `failure_detection`).

## Seams

- Candidate sources are injected by the composition root (working memory, knowledge base, memory recall).

## Collaborators

- Uses `memory`, `llm.embeddings`/encoders and the `evals` cascade for ranking; its output feeds the agent's `reason` node and the hallucination guard's evidence.

## Episodes

E07, E17, E18, E19, E25, E26.

## Tests

Run with `./venv/bin/pytest tests/unit/test_context.py -q` (or `make test` for everything).
