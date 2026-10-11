"""The direct answer: a ReAct loop whose every step is a numbered card, checked before it is sent.

A scripted model plays the answerer; one fake tool returns a fixed forecast. No network.
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from agent_platform.audit.ledger import NullAuditLedger
from agent_platform.evals.entailment import LexicalEntailmentModel
from agent_platform.evals.scoring import create_scoring_cascade
from agent_platform.executor import ToolExecutorImpl
from agent_platform.executor.registry_handler import RegistryToolHandler
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check import DetectionLadder
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import ModelToolCall
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.types import StatusEvent
from agent_platform.tools.path import ToolPath
from agent_platform.tools.registry import NamespacedToolRegistry
from agent_platform.tools.types import ToolDefinition, ToolResult
from research_agent.config import get_agent_settings
from research_agent.graph.answerer import DirectAnswerer
from research_agent.guardrails.answer_grounding import AnswerGrounding
from research_agent.guardrails.page_evidence import PageEvidence
from research_agent.llm.structured import StructuredCaller
from research_agent.observability.telemetry import RunTelemetry
from research_agent.prompts import answer as prompt
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


async def _unreachable_page(url: str) -> ToolResult:
    """No network in tests: a cited page cannot be read, so its snippet stays the evidence."""
    return ToolResult(error=f"offline: {url}")


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
    return _answerer_over(ScriptedLLM(script))


def _answerer_over(model: ScriptedLLM) -> tuple[DirectAnswerer, list[StatusEvent]]:
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
    caller = StructuredCaller(_Roles(model), PydanticValidator(), reporter)
    ladder = DetectionLadder(create_scoring_cascade(), entailment=LexicalEntailmentModel())
    pages = PageEvidence(_unreachable_page, settings.grounding)
    grounding = AnswerGrounding(
        ResponseGuardrail(ladder), settings.grounding, settings.messages, pages
    )
    audit = NullAuditLedger()
    telemetry = RunTelemetry(reporter, audit, trace_id="answer-test")
    executor = ToolExecutorImpl(RegistryToolHandler(registry), reporter=reporter)
    path = ToolPath(registry, executor, audit)
    return DirectAnswerer(caller, registry, path, settings, telemetry, grounding), events


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
    # Platform cards (loop control, executor) sit between the steps; the step order is the claim.
    steps = [stage for stage in stages if stage in {"thought", "action", "observation"}]
    assert steps[:4] == ["thought", "action", "observation", "thought"]
    assert "answer" in stages and "answer_context" not in stages
    # One card per model call: no separate request, response or usage cards.
    assert not {"llm_request", "llm_response_received", "llm_call"} & set(stages)
    thoughts = [e for e in events if e.stage == "thought" and e.status == "completed"]
    assert thoughts[0].summary is not None and thoughts[0].summary.startswith("Step 1 · ")
    assert thoughts[1].summary is not None and thoughts[1].summary.startswith("Step 2 · ")
    assert thoughts[0].tokens_delta == 15
    # Each THOUGHT is announced as pending before its model call, then settled in place.
    statuses = [e.status for e in events if e.stage == "thought"]
    assert statuses == ["started", "completed", "started", "completed"]
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


def _tool_calls_until_the_cap() -> list[dict[str, Any]]:
    """A different call each step, so it is the cap that stops the loop — not the governor
    suppressing a repeat or catching a cycle."""
    cities = range(get_agent_settings().answer.budget.max_tool_calls)
    return [
        {"tool_calls": [CALL.model_copy(update={"arguments": f'{{"location": "Pune {i}"}}'})]}
        for i in cities
    ]


@pytest.mark.asyncio
async def test_a_last_step_that_asks_for_a_tool_still_answers_from_what_was_found() -> None:
    """Regression: the model had the facts, asked for one more page on its last step, and the
    loop sent "I could not form an answer" (an Asian Games medal-tally question, 2026-10-04)."""
    settings = get_agent_settings().answer
    model = ScriptedLLM(
        _tool_calls_until_the_cap()
        + [{"tool_calls": [CALL]}] * settings.final_attempts
        + [{"content": "Tomorrow in Pune: light drizzle, up to 32.5°C."}]
    )
    answerer, _ = _answerer_over(model)
    reply = await answerer.answer("weather in Pune tomorrow?")
    assert "32.5°C" in reply
    assert settings.messages.empty not in reply
    # The last try is a fresh conversation over the observations, with nothing to resume.
    fallback = next(r for r in model.requests if r.messages[0].content == prompt.FROM_OBSERVATIONS)
    assert fallback.tools is None and "light drizzle" in str(fallback.messages[1].content)


@pytest.mark.asyncio
async def test_the_last_step_is_told_it_is_the_last_and_offered_no_tools() -> None:
    model = ScriptedLLM(
        _tool_calls_until_the_cap() + [{"content": "Tomorrow in Pune: light drizzle, 32.5°C."}]
    )
    answerer, _ = _answerer_over(model)
    reply = await answerer.answer("weather in Pune tomorrow?")
    final = model.requests[len(_tool_calls_until_the_cap())]
    assert final.tools is None
    assert final.messages[-1].content == prompt.FINAL_STEP
    assert "32.5°C" in reply
