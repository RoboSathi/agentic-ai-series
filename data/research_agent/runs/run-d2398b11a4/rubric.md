# Rubric

- R1 [must] Every factual sentence in the answer cites the ledger. (deterministic:citations_per_claim ≥ 1)
- R2 [must] Cited claims are supported by their quoted spans. (nli:claims_entailed ≥ 1)
- R3 [soft] The answer draws on at least two independent domains. (deterministic:distinct_domains ≥ 1)
- R4 [must] Cited sources fall within the 30-day recency window. (deterministic:recency_window ≥ 1)
- R5 [must] Every cited link was fetched and resolved. (deterministic:links_resolved ≥ 1)
- R6 [must] The report establishes at least 1 cited finding(s). (deterministic:findings_present ≥ 1)
