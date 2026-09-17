"""Type definitions for audit module."""

from pydantic import BaseModel


class AuditEvent(BaseModel):
    """An audit trail event."""

    event_type: str = ""
    details: dict = {}
