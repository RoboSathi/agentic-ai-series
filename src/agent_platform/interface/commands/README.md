# interface/commands — the operator's slash commands

What an operator types in the console. Commands return data (`CommandResult`), never pixels, and never reach a model. Platform commands are listed in `registry._default_commands()`; an agent's arrive from its `AgentHost`.

- **Entry points:** `CommandRegistry`, `Command`, `CommandResult`, `ConsoleEffect`; `review_cli.py`, `memory_cli.py` (the `make review` / memory targets)
- **Configuration:** `commands:`
- **Design:** [LLD §A24 (command table)](../../../../design_docs/LLD.md#a24-interface-tui-commands-presentation-host)
