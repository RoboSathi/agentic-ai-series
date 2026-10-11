# ADR 0012 — File memory, dreams for every agent, one review queue

- Status: accepted; implemented 2026-10-11
- Date: 2026-10-11
- Related: ADR 0010 (profile and skills), ra-0023 (the research agent's run memory),
  supersedes the storage parts of the 2026-09-27 dream and the 2026-10-03 one-HITL-queue
  decisions

## Context

- Long-term memory was three storage engines. Episodic memory was JSONL under `logs/`,
  semantic and procedural memory were SQLite tables in `agent.db`, and the dream inbox was a
  fourth table. Only the support agent could dream, and its dream lived in its own `admin/`.
- The research agent kept "lessons" in episodic memory and showed them on a LESSONS card. That
  was a separate mechanism from the dream, and nothing consolidated it.
- Skills (`agent_platform/skills/`) were procedural memory but lived outside `memory/`. People
  saw them as "procedures".
- A person decided things in four places: `/queue` (refunds and cut replies, support only),
  `/dlq` (quarantines), `/skills approve` (learned skills) and nothing at all for facts the
  curator held. A deny comment was optional, and an approval took no comment at all.
- Tool results reached the model provider verbatim, names and addresses included.
- The audit trail had no terminal view, and entries written offline (dreams) went to a
  different ledger than live turns.

## Decision

### Memory is files, one directory per agent (`agent_platform/memory/files/`)

`src/<agent>/data/memory/{dream_events,episodic,semantic,procedural}/`, gitignored.

- **Records are the truth.** Every version of a fact or skill is a numbered markdown record
  with YAML frontmatter under `.history/`. The numbering is a store-wide sequence, under a
  cross-process lock. Writes are atomic.
- **Views are regenerated**: `semantic/<subject>/<key>.md`, `procedural/<name>/SKILL.md` (the
  active version, verbatim) and a `MEMORY.md` index per store.
- **Search is derived**: `.index.sqlite` (FTS5 BM25) is rebuilt whenever its fingerprint no
  longer matches the records. Deleting it loses nothing.
- **Paths cannot escape** (Anthropic memory-tool rule): an absolute path, `..` or a symlink out
  is refused. A user-influenced name gets a digest suffix whenever it had to be cleaned.
- The store contracts are unchanged, so the facade and the governed pipeline did not change.

### Dreams for every agent (`agent_platform/memory/dream/`)

- A completed turn writes a **dream event** (`TurnRecorder`). An agent may write others; a
  research run does. An event carries **signals** (typed findings) and a **scope** (where they
  happened, written as a skill trigger).
- `make dream <agent>` and `/dream` run a `Dream` with the agent's **`MemoryProfile`**: three
  consolidators, one per store. `MemoryProfile.default()` works out of the box:
  - a template session summary
  - deterministic facts
  - `RecurringSignalSkills`, which turns a signal recurring in `skill_threshold` sessions into a
    valid learned SKILL.md and proposes it to the curator
- An agent customises by offering `memory_profile()` on its host (`MemoryHost`).
- Composition is in `host/memory.py` (`long_term_memory`, `governed_memory`, `build_dream`),
  once for all agents. `memory` still never imports `security`.

### Skills, not procedures

- `agent_platform/skills` moved to `agent_platform/memory/procedural/skills`.
- `ProceduralStore` was renamed `SkillStore`.
- Every user-visible "procedure" now says "skill". A skill is **authored** (a file in the repo)
  or **learned** (approved by a curator).
- The CoALA type stays `procedural`, and a SKILL.md's `## Procedure` section keeps its name.
- A procedural key is no longer normalised, so a skill's name is the name its file declares.

### One review queue (`agent_platform/review/`)

- `platform_review_queue` in the agent's database holds every kind of item:
  - the platform's: `memory_fact`, `skill`, `quarantine`
  - the agent's own: support's `refund`, `cancel_and_refund`, `reply`
- `CuratorQueue`, `DeadLetterQueue` and CSA's `HitlQueueRepository` keep their APIs and write
  through it.
- `ReviewService`:
  - refuses a decision without a comment (`review.min_comment_chars`)
  - reports an item outside the reviewer's scope as missing
  - runs the kind's handler, then settles, signs and audits the item
- `/queue` (every agent), `make queue <agent>` and the console's Review screen (`ctrl+r`, its
  buttons disabled until a comment is typed) are three doors to that one service.
- **Scope**:
  - A support console shows its verified customer's items.
  - Before anyone is verified, it shows only the agent's own (subject-less) items.
  - Agents without subjects see everything.

### The provider boundary (`security/pii/boundary.py`)

- With `pii` declared, `PiiBoundary` wraps the agent's LLM clients. Every message's text and
  tool-call arguments are tokenised on the way out, with the vault's keyed tokens, or a
  per-process key when there is no vault.
- The answer's tokens are restored from that request's own map. A token the request did not
  send stays a token.
- Grounding, guardrails and tools all work on real values. Only the provider sees tokens.

### Reading the audit trail (`audit/view.py`)

- `AuditTrail` merges the shared DuckDB ledger and the agent's own SQLite ledger.
- One renderer serves `/audit` and `make audit <agent>`:
  - a box table coloured by standing, then by decision
  - filters `DECISION`, `STANDING`, `KIND`, `SINCE` and `N`
  - `TRACE=<id>` shows one turn as a tree
- The trail holds tokens only. `make pii-reveal` stays the only way back to a value, and it is
  audited.

## mem0 and Letta: patterns, not libraries

- **mem0** makes a remote LLM call to decide every write (ADD/UPDATE/DELETE/NOOP). It stores
  to a vector DB, bypasses our governance and curator, and has no provenance or bi-temporality.
  We keep its reconcile vocabulary only.
- **Letta** owns the agent loop and its state, which conflicts with LangGraph. It needs a
  server, and its memory writes are model-directed. We keep its sleep-time consolidation (the
  dream).
- **Anthropic** offers a file directory a client owns, frontmatter memories, an index and
  SKILL.md progressive disclosure. That is what we adopted. It adds no dependency.
- The model is never given a memory-write tool. Only the dream writes long-term memory,
  through governance.

## What each agent customises

| Memory | Customer Support | Research | Ops |
|---|---|---|---|
| Dream events | turns; scope `intent:…`; refusals by `refusal_markers` | turns + one `run` event per run (`criterion_failed`, `source_rejected`, `budget_stop`; scope `archetype:…`) | none (no episodic) |
| Episodic | BART + NLI session summary (`memory/profile.py`) | `RunSummary` from the run's record | — |
| Semantic | default deterministic facts, per customer | default + `SourceReliability` (rejected domains, system-of-record) | — |
| Procedural | authored skills + learned from recurring refusals | learned skills only, selected at plan by `archetype:…` | authored skills only |
| Review kinds | `refund`, `cancel_and_refund`, `reply` (`console/review_handler.py`) | platform kinds only | platform kinds only |

## Consequences

- There is no data migration for memory. Demo memory is rebuilt by `make clean-memory` and
  `make dummy-data csa`, which now removes the memory directory. An older `hitl_queue` is
  moved into the review queue on first open.
- A learned skill counts toward an agent's `skills.max_prompt_chars`, like any skill.
- `/dlq`, `dlq_cli`, `make dlq`, `/skills approve`, CSA's `admin/{dream,show_memory,clean_memory}`
  and `research_agent/reflexion/` are deleted.
