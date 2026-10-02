# ADR ra-0017 — Two modes: Q&A and Research (amends ra-0011)

## Context

With four specialised archetypes and a keyword gate, most ordinary
questions ("important AI news in the last month", "what is the cycle life
of X?") were classified as open-ended and sent back with a reframe
question — and a reply in the menu's own words could loop. A search API
answers "which pages mention this?"; the archetypes answer "what does a
finished, checkable answer look like?". Both are needed, but most
questions do not need an interview.

## Decision

The user chooses between two **modes**, and the gate routes by default:

- **Q&A** (`qa`): a direct question goes straight to a small, automatic
  run — no interview and no approval. The gate builds the Scope Brief from
  the message (topic; a time window read by the configured parser, so
  "last 1 month" is 30 days), the rubric comes from the `qa` template
  (every sentence cited, entailed, ≥1 finding, recency if stated), the
  contract is frozen and hashed as usual (`auto_freeze`), and the existing
  run graph runs it on the Q&A budget (3 searches, one follow-up round, no
  gap-fill). Defences, the verbatim rule and the audit are unchanged.
- **Research**: the full flow. The four specialised kinds stay and are
  picked automatically; `general` covers research questions none fits.

`/qa <question>` and `/research [question]` override the gate (a bare
`/research` goes deeper on the last question). `needs_reframe` is kept
only for messages with no question at all; its numbered menu offers the
two modes with examples and resolves a reply by number, label or alias.
There is no separate "news" category: news is any question with a time
window, and a window of ≤31 days searches the providers' news vertical.

## Consequences

- Most questions are answered in one turn; research runs keep every guarantee.
- A stated time window is never asked about again (the interview item is pre-covered).
- Two new stop-rule users: `QuestionAnswered` (every coverage check passes).
- A zero budget cap (Q&A's gap-fill) is "not allowed", not "exhausted" —
  the commons `BudgetGuard` no longer reports it as a stop.

## Alternatives rejected

- **A `news_digest` archetype** — a category per topic does not scale; a
  time window is a property of any question.
- **No categories at all** — loses the per-kind stop rules and checks
  that make decision and fact-check reports auditable.
- **Always interview** — slow and costly for questions a few searches settle.
