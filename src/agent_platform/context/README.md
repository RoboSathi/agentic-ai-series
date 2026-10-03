# context — what the model is handed

Assembles the context window for a turn: which candidates enter, how they
are ranked, what is reduced, offloaded or isolated to stay inside the token
budget, and whether the assembled context shows a known failure mode
(poisoning, distraction, confusion, clash).

## Vocabulary

- **Chunk** — one candidate piece of context with a kind and provenance
- **strategy** — reduce, offload, isolate, retrieve, cache or consolidate — chosen by `StrategySelector`
- **five-stage pipeline** — gather → rank → select → assemble → check
- **context failure** — a detected poisoning / clash / distraction / confusion, diagnosed and mitigated

## Public interface

- `ContextAssembler` / `FiveStagePipeline` (`pipeline.py`)
- Retrievers: `LexicalRetriever` (BM25), `DenseRetriever`, `HybridRetriever`, `GraphRetriever`; `CrossEncoderReranker`
- Strategies (`strategies.py`) and `StrategySelector`
- `ContextFailureDetector`, `ContextHealth` (`failure.py`)
- `CandidateSource` (Protocol) — anything that contributes chunks, e.g. `WorkingMemoryContextSource`

## Configuration

Section `context:` — `token_budget`, `chars_per_token`, `retrieval_mode`, `top_k`,
`hybrid_lexical_weight`, `rerank`, `bm25_k1`/`bm25_b`, `strategies`, `reduce_keep_ratio`,
`failure_detection`. Whether the stage exists: `features.context` (+ `rerank`, `failure_detection`).

## Seams

- Candidate sources are injected by the composition root (working memory, knowledge base, memory recall).

## Collaborators

- Uses `memory`, `llm.embeddings`/encoders and the `evals` cascade for ranking; its output feeds the agent's `reason` node and the hallucination guard's evidence.

## Episodes

E07, E17, E18, E19, E25, E26.

## Tests

Run with `./venv/bin/pytest tests/unit/test_context.py -q` (or `make test` for everything).
