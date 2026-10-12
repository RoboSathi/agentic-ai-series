# Rubric

- R1 [must] Every factual sentence in the report cites collected evidence. (deterministic:citations_per_claim ≥ 1)
- R2 [must] Cited claims are supported by their quoted spans. (nli:claims_entailed ≥ 1)
- R3 [soft] Findings come from at least two independent domains. (deterministic:distinct_domains ≥ 1)
- R4 [soft] The findings answer the question as asked. (judge:user_item ≥ 0.7)
- R5 [must] Every cited link was fetched and resolved. (deterministic:links_resolved ≥ 1)
- R6 [must] The report establishes at least 1 cited finding(s). (deterministic:findings_present ≥ 1)
