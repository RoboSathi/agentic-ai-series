# Rubric

- R1 [must] The map names at least three distinct players. (deterministic:distinct_subjects ≥ 1)
- R2 [must] Every factual sentence in the report cites collected evidence. (deterministic:citations_per_claim ≥ 1)
- R3 [must] Cited claims are supported by their quoted spans. (nli:claims_entailed ≥ 1)
- R4 [soft] Sources span at least three independent domains. (deterministic:distinct_domains ≥ 1)
- R5 [must] Each player's 'good' is described with a citation. (deterministic:criterion_evidence ≥ 1)
- R6 [must] Each player's 'bad' is described with a citation. (deterministic:criterion_evidence ≥ 1)
- R7 [must] Each player's 'ugly' is described with a citation. (deterministic:criterion_evidence ≥ 1)
- R8 [must] Each player's 'interesting' is described with a citation. (deterministic:criterion_evidence ≥ 1)
- R9 [must] Cited sources fall within the 90-day recency window. (deterministic:recency_window ≥ 1)
- R10 [must] Every cited link was fetched and resolved. (deterministic:links_resolved ≥ 1)
- R11 [must] The report establishes at least 1 cited finding(s). (deterministic:findings_present ≥ 1)
