"""Weather from Open-Meteo: geocode a place name, then read its current conditions and forecast.

A structured API beats scraping weather pages (script-rendered shells that read as empty), is
keyless, and answers in one round trip per step.
"""

from typing import Any

import httpx
from pydantic import BaseModel, Field

from agentic_commons.logging import logged
from agentic_commons.tools.types import ToolResult
from research_agent.answer.tools.sources import URLS_KEY
from research_agent.config.settings import WeatherSettings


class WeatherArgs(BaseModel):
    location: str = Field(description="A city name, e.g. 'Pune' or 'Paris'.")


class OpenMeteoWeather:
    """`WeatherSettings` and an HTTP client in; a short written report out."""

    def __init__(self, settings: WeatherSettings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self._fields = {*settings.current_fields, *settings.daily_fields}

    @logged("research_agent")
    async def __call__(self, location: str) -> ToolResult:
        try:
            place = await self._geocode(location)
            if place is None:
                return ToolResult(error=f"no place found for '{location}'")
            forecast = await self._forecast(place)
        except httpx.HTTPError as exc:
            return ToolResult(error=f"{type(exc).__name__}: {exc}")
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
        lines = [
            f"Weather for {where} (source: {self._settings.forecast_url}), "
            f"as of {current.get('time', '')}"
        ]
        lines.append("Now: " + self._line(current, forecast.get("current_units", {})))
        daily, units = forecast.get("daily", {}), forecast.get("daily_units", {})
        for index, day in enumerate(daily.get("time", [])):
            row = {key: values[index] for key, values in daily.items() if key != "time"}
            lines.append(f"{day}: " + self._line(row, units))
        return "\n".join(lines)

    def _line(self, values: dict[str, Any], units: dict[str, str]) -> str:
        parts = []
        for key, value in values.items():
            if key not in self._fields:
                continue
            if key == "weather_code":
                parts.append(self._settings.codes.get(int(value), self._settings.unknown_code))
            else:
                parts.append(f"{key}={value}{units.get(key, '')}")
        return ", ".join(parts)
