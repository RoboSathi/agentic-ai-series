# ADR ra-0012 — The verbatim-span rule

## Context

The most common research-agent failure is a quote that is not in the source. Paraphrase checking by model is itself fallible.

## Decision

A ledger entry is admitted only if its span is an **exact substring of the page text this run fetched**, and the candidate's page hash matches that text (commons `VerbatimSpanValidator`). Rejections are recorded, never retried silently. One span of one page is **one** entry, whichever sub-question found it — a second sighting is not corroboration.

## Consequences

- `spans_verbatim` is a hard invariant.
- Withholding `citation_verification` swaps in the non-empty validator: the rule is a toggle like every other control.

## Alternatives rejected

- **NLI between claim and page** — expensive and probabilistic at the point where a string check is exact.
