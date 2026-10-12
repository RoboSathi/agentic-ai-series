# Rubric

- R1 [must] Every factual sentence in the report cites the ledger. (deterministic:citations_per_claim ≥ 1)
- R2 [must] Cited claims are supported by their quoted spans. (nli:claims_entailed ≥ 1)
- R3 [soft] Findings come from at least two independent domains. (deterministic:distinct_domains ≥ 1)
- R4 [soft] The findings answer the question as asked. (judge:user_item ≥ 0.7)
- R5 [must] The report covers 'what will be the change in swe work' with a citation. (deterministic:criterion_evidence ≥ 1)
- R6 [must] The report covers 'number/kind of swe roles' with a citation. (deterministic:criterion_evidence ≥ 1)
- R7 [must] The report covers 'starting from junior to architect' with a citation. (deterministic:criterion_evidence ≥ 1)
- R8 [soft] Primary sources are cited where they exist. (deterministic:primary_sources_cited ≥ 1)
- R9 [must] Every cited link was fetched and resolved. (deterministic:links_resolved ≥ 1)
- R10 [must] The report establishes at least 1 cited finding(s). (deterministic:findings_present ≥ 1)
