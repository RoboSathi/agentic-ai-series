# observability — a turn as the numbers SLOs watch

`TurnAccountant` turns a finished turn into a `TurnOutcome` and settles it into the platform's SLO trackers.

- **Entry points:** `TurnAccountant`, `TurnOutcome` (`accounting.py`)
- **Configuration:** `observability.slo_targets` (platform)
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent)
