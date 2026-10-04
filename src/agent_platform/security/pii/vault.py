"""The PII vault: what a redaction token stood for, encrypted, for audit.

Redaction tokenizes (`[PERSON:3f9a1c2b7e10]`) rather than erases, so a trace stays joinable —
the same value always gets the same token — and an auditor can recover what a token stood for.
The token is an HMAC of the value under the vault key: a plain hash of a phone number or an SSN
is brute-forced in seconds, a keyed one is not. The value is kept AES-GCM encrypted under the
same key, in the platform's SQLite file, scoped by agent, and deleted after the retention window.

Nothing reveals a value by itself: `reveal` is called by the audit command, which writes an
audit entry for every reveal. Without a key there is no vault, and redaction masks instead.
"""

import base64
import hashlib
import hmac
import os
from datetime import timedelta
from typing import NamedTuple, Protocol, runtime_checkable

from Crypto.Cipher import AES

from agent_platform.config import get_settings
from agent_platform.config.settings import PiiVaultSettings
from agent_platform.logging import get_logger, log_context_fields
from agent_platform.observability.types import utc_now
from agent_platform.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

# AES-128/192/256: the key lengths the cipher accepts, in bytes.
_AES_KEY_LENGTHS = (16, 24, 32)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_pii_vault (
    agent_id    TEXT NOT NULL,
    token       TEXT NOT NULL,
    category    TEXT NOT NULL,
    nonce       TEXT NOT NULL,
    ciphertext  TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    trace_id    TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    PRIMARY KEY (agent_id, token)
);
CREATE INDEX IF NOT EXISTS idx_platform_pii_vault_seen
    ON platform_pii_vault (agent_id, first_seen);
"""


class VaultKeyError(RuntimeError):
    """The vault is required but its key is missing or the wrong length."""


class RevealedValue(NamedTuple):
    """One token read back: its category, the value, and where it was first seen."""

    token: str
    category: str
    value: str
    session_id: str
    trace_id: str
    first_seen: str


@runtime_checkable
class PIIVault(Protocol):
    """Tokens for values, and — for audit — values for tokens."""

    def token(self, category: str, value: str) -> str: ...

    def keep(self, category: str, token: str, value: str) -> None: ...

    def reveal(self, token: str) -> RevealedValue | None: ...


class SqlitePIIVault:
    """HMAC tokens; AES-GCM values in the platform store, one agent's rows at a time."""

    def __init__(
        self,
        key: bytes,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
        settings: PiiVaultSettings | None = None,
    ) -> None:
        if len(key) not in _AES_KEY_LENGTHS:
            raise VaultKeyError(f"the vault key must be {_AES_KEY_LENGTHS} bytes, got {len(key)}")
        self._key = key
        self._settings = settings or get_settings().security.pii.vault
        self._chars = get_settings().security.pii.token_chars
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._store.execute_script(_SCHEMA)
        self._purge_expired()

    def token(self, category: str, value: str) -> str:
        """The same value in the same category always gets the same token, under this key."""
        digest = hmac.new(self._key, f"{category}|{value}".encode(), hashlib.sha256)
        return digest.hexdigest()[: self._chars]

    def keep(self, category: str, token: str, value: str) -> None:
        """Record a token's value (once: the first sighting is kept)."""
        cipher = AES.new(self._key, AES.MODE_GCM)
        sealed, tag = cipher.encrypt_and_digest(value.encode())
        where = log_context_fields()
        self._store.execute(
            "INSERT OR IGNORE INTO platform_pii_vault (agent_id, token, category, nonce, "
            "ciphertext, session_id, trace_id, first_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._agent_id, token, category, _b64(bytes(cipher.nonce)), _b64(sealed + tag),
                where.get("session", ""), where.get("trace_id", ""), utc_now().isoformat(),
            ),
        )  # fmt: skip

    def reveal(self, token: str) -> RevealedValue | None:
        rows = self._store.execute(
            "SELECT category, nonce, ciphertext, session_id, trace_id, first_seen "
            "FROM platform_pii_vault WHERE agent_id = ? AND token = ?",
            (self._agent_id, token),
        ).rows
        if not rows:
            return None
        row = rows[0]
        return RevealedValue(
            token=token,
            category=row["category"],
            value=self._open(row["nonce"], row["ciphertext"]),
            session_id=row["session_id"],
            trace_id=row["trace_id"],
            first_seen=row["first_seen"],
        )

    def _open(self, nonce: str, sealed: str) -> str:
        raw = base64.b64decode(sealed)
        body, tag = raw[:-_TAG_BYTES], raw[-_TAG_BYTES:]
        cipher = AES.new(self._key, AES.MODE_GCM, nonce=base64.b64decode(nonce))
        return cipher.decrypt_and_verify(body, tag).decode()

    def _purge_expired(self) -> None:
        cutoff = utc_now() - timedelta(days=self._settings.retention_days)
        self._store.execute(
            "DELETE FROM platform_pii_vault WHERE agent_id = ? AND first_seen < ?",
            (self._agent_id, cutoff.isoformat()),
        )


# GCM's authentication tag, appended to the ciphertext.
_TAG_BYTES = 16


def vault_key(settings: PiiVaultSettings | None = None) -> bytes | None:
    """The vault key from its environment variable; None (with a warning) when unusable.

    A required vault with no usable key is an error at start-up, not a silent downgrade.
    """
    resolved = settings or get_settings().security.pii.vault
    key = os.environ.get(resolved.key_env, "").encode("utf-8")
    if len(key) in _AES_KEY_LENGTHS:
        return key
    reason = (
        f"{resolved.key_env} is unset"
        if not key
        else f"{resolved.key_env} must be {_AES_KEY_LENGTHS} bytes, got {len(key)}"
    )
    if resolved.required:
        raise VaultKeyError(f"The PII vault is required but {reason}")
    get_logger("pii").warning("No PII vault: redaction masks instead of tokenizing", reason=reason)
    return None


def create_pii_vault(
    store: PlatformStore | SqliteConnectionPool | None = None, *, agent_id: str | None = None
) -> SqlitePIIVault | None:
    """The configured vault, or None when no key is set (and the vault is not required)."""
    key = vault_key()
    return SqlitePIIVault(key, store, agent_id=agent_id) if key is not None else None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


__all__ = [
    "PIIVault",
    "RevealedValue",
    "SqlitePIIVault",
    "VaultKeyError",
    "create_pii_vault",
    "vault_key",
]
