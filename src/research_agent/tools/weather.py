"""Weather from Open-Meteo: geocode a place name, then read its current conditions and forecast.

A structured API beats scraping weather pages (script-rendered shells that read as empty), is
keyless, and answers in one round trip per step.
"""

from typing import Any

import httpx
from pydantic import BaseModel, Field

from agent_platform.logging import describe_error, logged
from agent_platform.tools.types import ToolResult
from research_agent.config.settings import AnswerSettings
from research_agent.tools import URLS_KEY


class WeatherArgs(BaseModel):
    location: str = Field(description="A city name, e.g. 'Pune' or 'Paris'.")


class OpenMeteoWeather:
    """The weather settings and an HTTP client in; a short written report out."""

    def __init__(self, settings: AnswerSettings, client: httpx.AsyncClient) -> None:
        self._settings = settings.weather
        self._messages = settings.messages
        self._client = client
        self._fields = {*self._settings.current_fields, *self._settings.daily_fields}

    @logged("research_agent")
    async def __call__(self, location: str) -> ToolResult:
        try:
            place = await self._geocode(location)
            if place is None:
                return ToolResult(error=self._messages.no_place.format(location=location))
            forecast = await self._forecast(place)
        except httpx.HTTPError as exc:
            return ToolResult(error=describe_error(exc))
        return ToolResult(
            success=True,
            output=self._report(place, forecast),
            data={URLS_KEY: [self._settings.forecast_url]},
        )

    async def _geocode(self, location: str) -> dict[str, Any] | None:
        name = location.partition(",")[0].strip()
        response = await self._client.get(
            self._settings.geocode_url,
            params={"name": name, "count": self._settings.geocode_results, "format": "json"},
            timeout=self._settings.timeout_seconds,
        )
        response.raise_for_status()
        results: list[dict[str, Any]] = response.json().get("results") or []
        return results[0] if results else None

    async def _forecast(self, place: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.get(
            self._settings.forecast_url,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": ",".join(self._settings.current_fields),
                "daily": ",".join(self._settings.daily_fields),
                "forecast_days": self._settings.forecast_days,
                "timezone": "auto",
            },
            timeout=self._settings.timeout_seconds,
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body

    def _report(self, place: dict[str, Any], forecast: dict[str, Any]) -> str:
        where = ", ".join(
            str(place[key]) for key in ("name", "admin1", "country") if place.get(key)
        )
        current = forecast["current"]
        notes = self._messages
        lines = [
            notes.weather_header.format(
                where=where, url=self._settings.forecast_url, time=current.get("time", "")
            ),
            notes.weather_now.format(
                conditions=self._line(current, forecast.get("current_units", {}))
            ),
        ]
        daily, units = forecast.get("daily", {}), forecast.get("daily_units", {})
        for index, day in enumerate(daily.get("time", [])):
            row = {key: values[index] for key, values in daily.items() if key != "time"}
            lines.append(notes.weather_day.format(day=day, conditions=self._line(row, units)))
        return "\n".join(lines)

    def _line(self, values: dict[str, Any], units: dict[str, str]) -> str:
        parts = []
        for key, value in values.items():
            if key not in self._fields:
                continue
            if key == "weather_code":
                parts.append(self._settings.codes.get(int(value), self._settings.unknown_code))
            else:
                # Words, not API field names: the model and the grounding check both read this.
                label = self._settings.field_labels.get(key, key)
                parts.append(f"{label} {value}{units.get(key, '')}")
        return ", ".join(parts)
