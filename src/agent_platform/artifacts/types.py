"""Frozen artifacts: typed values whose content cannot drift unnoticed.

An artifact is frozen once — its payload is serialised and hashed at that
moment — and every later read can prove the payload is the one that was
frozen. The point is not secrecy but *agreement*: two components that
hold the same hash are provably talking about the same thing, and one
that holds a different payload finds out on `verify()` rather than by
acting on it.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.observability.types import utc_now
from agent_platform.primitives.hashing import canonical_hash


class ArtifactTamperedError(RuntimeError):
    """An artifact's payload no longer matches the hash it was frozen with."""


class FrozenArtifact(BaseModel):
    """One immutable, hashed payload, identified by kind and id."""

    model_config = ConfigDict(frozen=True)

    kind: str
    artifact_id: str
    payload: dict[str, Any]
    content_hash: str
    frozen_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def freeze(
        cls, kind: str, artifact_id: str, payload: BaseModel | dict[str, Any]
    ) -> "FrozenArtifact":
        """Serialise and hash a payload now; the hash is the artifact's identity."""
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
        return cls(
            kind=kind, artifact_id=artifact_id, payload=data, content_hash=canonical_hash(data)
        )

    def verify(self) -> bool:
        """Whether the payload still hashes to what was frozen."""
        return canonical_hash(self.payload) == self.content_hash

    def matches(self, value: BaseModel | dict[str, Any]) -> bool:
        """Whether a live value is the one this artifact froze."""
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
        return canonical_hash(data) == self.content_hash

    def require_intact(self) -> "FrozenArtifact":
        """Return self, or raise if the payload was altered after freezing."""
        if not self.verify():
            raise ArtifactTamperedError(
                f"{self.kind} '{self.artifact_id}' changed after it was frozen"
            )
        return self
