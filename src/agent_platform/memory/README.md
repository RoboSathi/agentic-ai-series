# memory — the four CoALA stores

Working, episodic, semantic and procedural memory — all real, none
stubbed — plus the governed write pipeline that decides what is worth
keeping, a deterministic recall tier, and document RAG for the knowledge
base. Every write is authorized, attributed and audited.

## Vocabulary

- **working** — this conversation's turns and current-turn scratch
- **episodic** — what happened — append-only session logs
- **semantic** — what is true — facts and the knowledge index (RAG)
- **procedural** — how to do things — versioned know-how
- **provenance** — where a memory came from and how much to trust it
- **curator** — the queue that reviews contested or sensitive writes

## Public interface

- `MemoryFacade` — the governed long-term entry point (`facade.py`)
- Stores: `WorkingMemoryStore`, `EpisodicStore`, `SemanticStore`, `SkillStore`
- `MemoryWritePipelineImpl` (`write_pipeline.py`); governance: `MemoryCommand`, `MemoryPrincipal`, `MemoryAuthorizer` (`governance.py`)
- RAG (`semantic/rag/`): chunking, index (sqlite-vec + FTS), fusion (RRF), retriever, knowledge-base context source
- Deterministic recall tier (`deterministic/`)
- Files (`files/`): `MemoryDirectory`, frontmatter records, `MEMORY.md` index, derived FTS index (ADR 0012)
- Dreams (`dream/`): `DreamEvent`, `FileDreamInbox`, `TurnRecorder`, consolidator protocols with defaults, `Dream` — how events are created and consolidated: `dream/README.md`
- `MemoryProfile` (`profile.py`): an agent's only customisation; `MemoryProfile.default()` works out of the box
- Skills (`procedural/skills/`): the SKILL.md catalog, selector and instructions (ADR 0010)

## Configuration

Section `memory:` — `files` (root and layout of `src/<agent>/data/memory/`), `dream`, `subject_label`, `semantic.rag` (db, corpus, chunking, retrieval), `write_pipeline`,
`provenance_weights`, `retrieval_weights`, `recency_half_life_days`, `governance`, `deterministic`,
`importance`. Which stores exist: `features.memory` and its four children.

## Seams

- The composition root may inject stores — `build_agent(working_memory=..., episodic_store=...)`; the eval harness gives each case its own episodic log.
- `host/memory.py` composes governed long-term memory and dreams for any agent; a host offers `memory_profile()` (`MemoryHost`) to customise.
- `make dream|show-memory|clean-memory|memory-reindex <agent>` (`commands/memory_cli.py`).

## Collaborators

- Feeds `context`; writes `audit`; uses `security.pii` for redaction and `evals` scoring for relevance.

## Episodes

E08, E13–E16, E21, E31, E39, E48a, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_memory.py -q` (or `make test` for everything).

## Status and limits

RAG, governance and the deterministic tier were built without dedicated tests; that pass is outstanding.
