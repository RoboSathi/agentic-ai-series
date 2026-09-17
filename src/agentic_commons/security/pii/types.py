"""Type definitions for PII detection."""

from pydantic import BaseModel


class PIIScanResult(BaseModel):
    """Result of PII scanning."""

    clean: bool = True
    findings: list[str] = []
