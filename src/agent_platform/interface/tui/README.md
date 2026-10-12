# interface/tui — the operator console

One Textual app: a live feed of stage cards, a status line, slash commands with completion, side panels a host offers, the Review screen and run controls. A consumer of observability and the stores, never a source of truth.

- **Entry points:** `AgenticTuiApp`, `main` (`python -m agent_platform.interface.tui`, `make agent <agent>`), widgets, `review_screen.py`
- **Configuration:** `tui:`
- **Design:** [LLD §A24](../../../../design_docs/LLD.md#a24-interface-tui-commands-presentation-host)
