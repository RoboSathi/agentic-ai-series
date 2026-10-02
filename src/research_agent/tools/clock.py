"""The current date and time in a named timezone."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from agentic_commons.tools.types import ToolResult
from research_agent.config.settings import AnswerSettings


class DateTimeArgs(BaseModel):
    timezone: str = Field(description="An IANA timezone such as 'Asia/Kolkata' or 'UTC'.")


class Clock:
    """`now` in the zone the model names, written as configured."""

    def __init__(self, settings: AnswerSettings) -> None:
        self._format = settings.clock_format
        self._unknown = settings.messages.unknown_timezone

    def __call__(self, timezone: str) -> ToolResult:
        try:
            now = datetime.now(ZoneInfo(timezone.strip()))
        except (ZoneInfoNotFoundError, ValueError):
            return ToolResult(error=self._unknown.format(timezone=timezone))
        return ToolResult(
            success=True, output=now.strftime(self._format), data={"iso": now.isoformat()}
        )
