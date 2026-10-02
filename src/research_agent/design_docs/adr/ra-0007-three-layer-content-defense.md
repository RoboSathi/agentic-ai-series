# ADR ra-0007 — Three-layer content defence

## Context

Every fetched page is attacker-controlled text read by a model that holds the user's private context. One control is not enough: each layer has a known failure mode.

## Decision

Three independently toggleable layers (details in `../threat_model.md`): **structural** (`injection_structural`: outbound guard over every query/URL, source policy over every hit), **classifier** (`injection_classifier`: the platform's `InstructionDataSeparator` over fetched text; a flagged page is rejected whole) and **quarantine** (`quarantined_extractor`: only the extractor reads raw text, in the platform's data envelope, and can return only verbatim spans).

## Consequences

- The ablations measure each layer alone: the classifier alone keeps every hostile page out; the structural layer alone stops exfiltration but not the endorsement; the quarantine alone stops page-directed actions but an endorsement is still a verbatim span.
- Rejected pages are listed in Coverage & Limits with the phrases that tripped the classifier.

## Alternatives rejected

- **Classifier only** — phrasing changes defeat it.
- **Quarantine only** — a verbatim span can still carry an attacker's claim.
