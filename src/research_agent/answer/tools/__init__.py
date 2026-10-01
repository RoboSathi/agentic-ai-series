"""The direct-answer agent's tools: read-only, each returning data for the model."""

import httpx

from agentic_commons.tools.registry import NamespacedToolRegistry
from agentic_commons.tools.types import ToolDefinition
from agentic_commons.web import WebGateway
from research_agent.answer.tools.clock import DateTimeArgs, current_datetime
from research_agent.answer.tools.sources import URLS_KEY
from research_agent.answer.tools.weather import OpenMeteoWeather, WeatherArgs
from research_agent.answer.tools.web import FetchArgs, FetchUrl, SearchArgs, WebSearch
from research_agent.config.settings import AnswerSettings
from research_agent.prompts import answer as prompt
from research_agent.research.defenses import InjectionScreen

NAMESPACE = "answer"

__all__ = ["NAMESPACE", "URLS_KEY", "build_answer_registry"]


def build_answer_registry(
    settings: AnswerSettings,
    gateway: WebGateway,
    http: httpx.AsyncClient,
    screen: InjectionScreen | None,
) -> NamespacedToolRegistry:
    """Every tool the answerer may call. None has a side effect, so none needs authorizing."""
    registry = NamespacedToolRegistry()
    registry.register_all(
        [
            _tool(
                "web_search",
                prompt.SEARCH_DESCRIPTION,
                SearchArgs,
                WebSearch(gateway, settings),
            ),
            _tool(
                "fetch_url",
                prompt.FETCH_DESCRIPTION,
                FetchArgs,
                FetchUrl(gateway, settings, screen),
            ),
            _tool("current_datetime", prompt.DATETIME_DESCRIPTION, DateTimeArgs, current_datetime),
            _tool(
                "weather",
                prompt.WEATHER_DESCRIPTION,
                WeatherArgs,
                OpenMeteoWeather(settings.weather, http),
            ),
        ]
    )
    return registry


def _tool(name: str, description: str, args: type, handler: object) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        namespace=NAMESPACE,
        description=description,
        args_schema=args,
        handler=handler,  # type: ignore[arg-type]
        surface="locked",
        side_effecting=False,
    )
