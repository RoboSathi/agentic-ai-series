"""The direct answer: a ReAct loop whose every step is a numbered card, checked before it is sent.

A scripted model plays the answerer; one fake tool returns a fixed forecast. No network.
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from agentic_commons.audit.ledger import NullAuditLedger
from agentic_commons.evals.entailment import LexicalEntailmentModel
from agentic_commons.evals.scoring import create_scoring_cascade
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.guardrails.response import ResponseGuardrail
from agentic_commons.hallucination_protection import DetectionLadder
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import ModelToolCall
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.observability.types import StatusEvent
from agentic_commons.tools.registry import NamespacedToolRegistry
from agentic_commons.tools.types import ToolDefinition, ToolResult
from research_agent.config import get_agent_settings
from research_agent.graph.answerer import DirectAnswerer
from research_agent.guardrails.answer_grounding import AnswerGrounding
from research_agent.research.telemetry import RunTelemetry
from research_agent.structured import StructuredCaller
from research_agent.tools import URLS_KEY
from tests.conftest import ScriptedLLM

pytestmark = pytest.mark.usefixtures("ra_config")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST = (
    "Weather for Pune, Maharashtra, India, as of 2026-10-01T21:30\n"
    "2026-10-02: light drizzle, temperature_2m_max=32.5°C, temperature_2m_min=22.7°C, "
    "precipitation_probability_max=61%"
)
CALL = ModelToolCall(id="c1", name="answer__weather", arguments='{"location": "Pune"}')


class _WeatherArgs(BaseModel):
    location: str = Field(description="A city.")


def _weather(location: str) -> ToolResult:
    return ToolResult(success=True, output=FORECAST, data={URLS_KEY: [FORECAST_URL]})


class _Roles:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def client_for(self, role: str) -> LLMClient:
        return self._llm

    def model_key(self, role: str) -> str:
        return f"scripted:{role}"


def _answerer(script: list[dict[str, Any]]) -> tuple[DirectAnswerer, list[StatusEvent]]:
    settings = get_agent_settings().answer
    feed = InMemoryLiveStatusFeed(max_events=0)
    events: list[StatusEvent] = []
    feed.subscribe(events.append)
    reporter = StageReporter(feed, run_id="answer-test")
    registry = NamespacedToolRegistry()
    registry.register_all(
        [
            ToolDefinition(
                name="weather",
                namespace="answer",
                description="Weather for a city.",
                args_schema=_WeatherArgs,
                handler=_weather,  # type: ignore[arg-type]
                surface="locked",
                side_effecting=False,
            )
        ]
    )
    caller = StructuredCaller(_Roles(ScriptedLLM(script)), PydanticValidator(), reporter)
    ladder = DetectionLadder(create_scoring_cascade(), entailment=LexicalEntailmentModel())
    grounding = AnswerGrounding(ResponseGuardrail(ladder), settings.grounding, settings.messages)
    telemetry = RunTelemetry(reporter, NullAuditLedger(), trace_id="answer-test")
    return DirectAnswerer(caller, registry, settings, telemetry, grounding), events


def _stages(events: list[StatusEvent]) -> list[str]:
    return [e.stage for e in events if e.status != "started"]


@pytest.mark.asyncio
async def test_each_step_is_a_numbered_thought_action_observation_card() -> None:
    answerer, events = _answerer(
        [
            {"tool_calls": [CALL]},
            {"content": "Tomorrow in Pune: light drizzle, up to 32.5°C."},
        ]
    )
    reply = await answerer.answer("weather in Pune tomorrow?")
    stages = _stages(events)
    assert stages[:4] == ["thought", "action", "observation", "thought"]
    assert "answer" in stages and "answer_context" not in stages
    # One card per model call: no separate request, response or usage cards.
    assert not {"llm_request", "llm_response_received", "llm_call"} & set(stages)
    thoughts = [e for e in events if e.stage == "thought"]
    assert thoughts[0].summary is not None and thoughts[0].summary.startswith("Step 1 · ")
    assert thoughts[1].summary is not None and thoughts[1].summary.startswith("Step 2 · ")
    assert thoughts[0].tokens_delta == 15
    assert FORECAST_URL in reply


@pytest.mark.asyncio
async def test_a_figure_no_tool_returned_is_removed_before_the_answer_is_sent() -> None:
    answerer, events = _answerer(
        [
            {"tool_calls": [CALL]},
            {"content": "Tomorrow in Pune:\n- light drizzle, up to 32.5°C\n- 90% chance of rain"},
            {"content": "Tomorrow in Pune:\n- light drizzle, up to 32.5°C\n- 90% chance of rain"},
        ]
    )
    reply = await answerer.answer("weather in Pune tomorrow?")
    assert "90%" not in reply and "32.5°C" in reply
    assert get_agent_settings().answer.messages.filtered_note in reply
    reflect = next(e for e in events if e.stage == "hallucination_check")
    assert reflect.status == "warning" and "90% chance of rain" in str(reflect.fields)
    # A figure found in no evidence is refuted by a rule; this agent still asks for one rewrite
    # from the evidence (`repair_direct_refutations`), and removes what the rewrite kept.
    assert _stages(events).count("revise") == 1


@pytest.mark.asyncio
async def test_an_answer_from_the_models_own_knowledge_is_not_grounded() -> None:
    answerer, events = _answerer([{"content": "Paris is the capital of France."}])
    assert await answerer.answer("capital of France?") == "Paris is the capital of France."
    assert "hallucination_check" not in _stages(events)
