# Ablations

Each file here is one more configuration layer over `research_agent`'s own
overlay. `make <eval target> ra ABLATION=<name>` appends it to
`AGENTIC_OVERLAYS`, so the agent is built with that posture and nothing
else changes: same fixtures, same cases, same reference model. The eval
fingerprint covers the effective configuration, so every ablation gets its
own run and `make eval-diff ra A=<run> B=<run>` compares any two.

| file | what is withheld | what the evals show |
|---|---|---|
| `no_rubric_audit.yaml` | the rubric and the Goldfish Audit | criteria no longer map to acceptance items; nothing stops a run on unmet items |
| `injection_off.yaml` | all three content-defence layers | the override's endorsement reaches the report; private terms leave the agent |
| `injection_structural_only.yaml` | classifier and quarantine | exfiltration is blocked at the boundary; the endorsement still gets in |
| `injection_classifier_only.yaml` | structural checks and quarantine | hostile pages never reach a reader |
| `injection_quarantine_only.yaml` | structural checks and classifier | nothing leaves, but the endorsement is still a verbatim span |
| `same_model_judge.yaml` | judge independence | a judge-only flaw (trade-offs omitted) is no longer caught |
