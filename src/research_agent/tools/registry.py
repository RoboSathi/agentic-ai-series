"""Every tool the direct answerer may call, in one namespace, none with a side effect."""

import httpx

from agentic_commons.tools.registry import NamespacedToolRegistry
from agentic_commons.tools.types import ToolDefinition
from agentic_commons.web import WebGateway
from research_agent.config.settings import AnswerSettings
from research_agent.prompts import answer as prompt
from research_agent.research.defenses import InjectionScreen
from research_agent.tools.clock import Clock, DateTimeArgs
from research_agent.tools.weather import OpenMeteoWeather, WeatherArgs
from research_agent.tools.web import FetchArgs, FetchUrl, SearchArgs, WebSearch

NAMESPACE = "answer"

__all__ = ["NAMESPACE", "build_answer_registry"]


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
            _tool("current_datetime", prompt.DATETIME_DESCRIPTION, DateTimeArgs, Clock(settings)),
            _tool(
                "weather",
                prompt.WEATHER_DESCRIPTION,
                WeatherArgs,
                OpenMeteoWeather(settings, http),
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
