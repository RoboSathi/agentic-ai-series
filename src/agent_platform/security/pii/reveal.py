"""`make pii-reveal <agent> T=<token> [WHY=<reason>]`: what one redaction token stood for.

For audit, and audited: every reveal — found or not — writes an audit entry naming who asked,
which token and why, before anything is printed. The value is decrypted from the vault with the
same key that made the token; without that key nothing can be revealed.
"""

import argparse
import getpass
import sys

from agent_platform.audit import AuditEntry, create_audit_ledger
from agent_platform.config import get_settings
from agent_platform.logging import new_trace_id
from agent_platform.observability import DuckDbObservabilityStore
from agent_platform.security.pii.vault import RevealedValue, SqlitePIIVault, vault_key

ACTOR = "pii_reveal"
ACTION = "pii.revealed"


def reveal(token: str, reason: str, who: str) -> RevealedValue | None:
    """The token's value, after the audit entry for asking has been written."""
    key = vault_key()
    if key is None:
        raise SystemExit(f"No vault key: set {get_settings().security.pii.vault.key_env}")
    vault = SqlitePIIVault(key)
    found = vault.reveal(token)
    create_audit_ledger(store=DuckDbObservabilityStore(), agent_id=get_settings().agent.id).record(
        AuditEntry(
            trace_id=new_trace_id(),
            actor=ACTOR,
            action=ACTION,
            authority_context={"token": token, "requested_by": who, "found": found is not None},
            decision="allow" if found is not None else "deny",
            rationale=reason,
        )
    )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make pii-reveal <agent> T=<token>")
    parser.add_argument("token")
    parser.add_argument("--why", required=True, help="why the value is needed (audited)")
    args = parser.parse_args(argv)
    found = reveal(args.token, args.why, getpass.getuser())
    if found is None:
        print(f"No vault entry for {args.token} (expired, another agent's, or never seen).")
        return 1
    print(f"{found.category}: {found.value}")
    print(f"first seen {found.first_seen} · session {found.session_id} · trace {found.trace_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
