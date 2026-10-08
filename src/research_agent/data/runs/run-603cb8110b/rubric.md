# Rubric

- R1 [must] The map names at least three distinct players. (deterministic:distinct_subjects ≥ 1)
- R2 [must] Every factual sentence in the report cites collected evidence. (deterministic:citations_per_claim ≥ 1)
- R3 [must] Cited claims are supported by their quoted spans. (nli:claims_entailed ≥ 1)
- R4 [soft] Sources span at least three independent domains. (deterministic:distinct_domains ≥ 1)
- R5 [must] Each player's 'medal tally' is described with a citation. (deterministic:criterion_evidence ≥ 1)
- R6 [must] The known unknown 'what the answer will be used for' is acknowledged and bounded in the report. (deterministic:known_unknown_acknowledged ≥ 1)
- R7 [must] Every cited link was fetched and resolved. (deterministic:links_resolved ≥ 1)
- R8 [must] The report establishes at least 1 cited finding(s). (deterministic:findings_present ≥ 1)
