# Rubric

- R1 [must] Every factual sentence in the report cites collected evidence. (deterministic:citations_per_claim ≥ 1)
- R2 [must] Cited claims are supported by their quoted spans. (nli:claims_entailed ≥ 1)
- R3 [soft] The explanation is accurate and uses concrete examples. (judge:explanation_accuracy ≥ 0.7)
- R4 [must] The known unknown 'who the explanation is for and what they already know' is acknowledged and bounded in the report. (deterministic:known_unknown_acknowledged ≥ 1)
- R5 [must] The known unknown 'the sub-concepts the explanation must cover' is acknowledged and bounded in the report. (deterministic:known_unknown_acknowledged ≥ 1)
- R6 [must] The known unknown 'what the answer will be used for' is acknowledged and bounded in the report. (deterministic:known_unknown_acknowledged ≥ 1)
- R7 [must] Cited sources fall within the 90-day recency window. (deterministic:recency_window ≥ 1)
- R8 [must] Every cited link was fetched and resolved. (deterministic:links_resolved ≥ 1)
- R9 [must] The report establishes at least 1 cited finding(s). (deterministic:findings_present ≥ 1)
