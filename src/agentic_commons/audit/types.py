"""Audit value types (LLD §5).

The audit trail answers a regulator's question, not an engineer's: *who
was allowed to do this, under what authority, against which version of
the model, the policy and the data?* Anything less is a log file.

`authority_context` is carried as a plain mapping rather than importing
`gatekeeper.AuthorityContext`: audit sits at layer 5 and gatekeeper at
layer 4, so the dependency would invert. The Gatekeeper serialises its
own context on the way in (HLD §layering).
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now

Decision = Literal["allow", "deny", "escalate"]


class AuditEntry(BaseModel):
    """One consequential decision, attributable to a versioned tuple (E33)."""

    trace_id: str
    actor: str
    action: str = ""
    authority_context: dict[str, Any] = Field(default_factory=dict)
    model_version: str = ""
    policy_version: str = ""
    data_version: str = ""
    decision: Decision
    rationale: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
