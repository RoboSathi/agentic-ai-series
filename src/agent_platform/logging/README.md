# logging — file logs and the traced call

Per-subsystem log files under `logs/<agent>/` (always) and the console (optional); `@instrument` records what a traced call was given and returned, in full.

- **Entry points:** `get_logger`, `log_directory`, `@instrument`, `@logged`, `get_current_trace_id`
- **Configuration:** `logging:`
- **Design:** [LLD §A2](../../../design_docs/LLD.md#a2-logging)
