"""The current date and time in a named timezone."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from agentic_commons.tools.types import ToolResult


class DateTimeArgs(BaseModel):
    timezone: str = Field(description="An IANA timezone such as 'Asia/Kolkata' or 'UTC'.")


def current_datetime(timezone: str) -> ToolResult:
    try:
        now = datetime.now(ZoneInfo(timezone.strip()))
    except (ZoneInfoNotFoundError, ValueError):
        return ToolResult(error=f"unknown timezone '{timezone}'")
    text = now.strftime("%A, %d %B %Y, %H:%M:%S %Z (UTC%z)")
    return ToolResult(success=True, output=text, data={"iso": now.isoformat()})
