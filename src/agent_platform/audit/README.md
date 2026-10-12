# audit — the append-only decision trail

Records every authorization decision with its standing and version stamps, and renders the merged trail for `/audit` and `make audit <agent>`. It decides nothing; it remembers.

- **Entry points:** `AuditLedger`, `create_audit_ledger`, `AuditEntry`, `AuditTrail` (`view.py`), `python -m agent_platform.audit.cli`
- **Configuration:** `audit:`
- **Design:** [LLD §A21](../../../design_docs/LLD.md#a21-audit)
