"""The fixture world's reference model — an `LLMClient` that answers every role deterministically.

It stands where a hosted model would: it receives the same prompts (role
instructions + one JSON payload + the answer schema) and returns JSON the
same validators check. It differs in one way that matters: it *knows the
fixture world* — its search index and the persona in `world.yaml` — so a
run over the fixture world is reproducible to the token, which is what
lets evals gate on exact invariants and ablations compare like with like.

It is deliberately naive where a real model is naive. It challenges only
vague or absolute answers (so the anti-sycophancy guard has work to do);
when it reads raw page text it follows instructions it finds there (so the
injection defences have something to stop); and when it is asked to judge
a report it wrote itself it approves it (so a same-model judge measurably
misses what an independent one catches).

Nothing in the agent imports this module except the composition root,
and only when `research_agent.model_backend` is `fixture`.
"""

import json
from collections.abc import Callable
from typing import Any

from agentic_commons.config.settings import ProviderName
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    TokenUsage,
)
from research_agent.config.settings import ResearchAgentSettings
from research_agent.fixture_world.behaviours import reading, scoping, writing
from research_agent.fixture_world.world import World, load_world
from research_agent.prompts.base import instructions_of, payload_of
from research_agent.research.matching import TermMatcher

_TIER = "fixture"

Behaviour = Callable[["FixtureWorldModel", dict[str, Any], str], dict[str, Any]]

_TASKS: dict[str, Behaviour] = {
    "gate": scoping.gate,
    "interview": scoping.interview,
    "plan": reading.plan,
    "extract": reading.extract,
    "research_read": reading.research_read,
    "reflect": reading.reflect,
    "synthesize": writing.synthesize,
    "judge": writing.judge,
}


class FixtureWorldModel(LLMClient):
    """One 'model' (one model key); roles sharing a key share an instance."""

    def __init__(self, world: World, model_name: str, matcher: TermMatcher) -> None:
        self.world = world
        self.matcher = matcher
        self._model = model_name
        # Titles of reports this model wrote: the self-preference a same-model judge shows.
        self.written: set[str] = set()
        self.calls = 0

    @property
    def tier(self) -> str:
        return _TIER

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls += 1
        payload = payload_of(request.messages)
        behaviour = _TASKS[str(payload.get("task", ""))]
        answer = json.dumps(behaviour(self, payload, instructions_of(request.messages)))
        prompt_chars = sum(len(message.content) for message in request.messages)
        return CompletionResult(
            content=answer,
            model=self._model,
            finish_reason="stop",
            tier=_TIER,
            usage=self._usage(prompt_chars, len(answer)),
        )

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)

    def _usage(self, prompt_chars: int, answer_chars: int) -> TokenUsage:
        rate = self.world.persona.tokens_per_char
        prompt, completion = int(prompt_chars * rate), int(answer_chars * rate)
        return TokenUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )


def fixture_client_factory(
    agent: ResearchAgentSettings,
) -> Callable[[ProviderName, str], LLMClient]:
    """A `RoleModelRouter` factory: one reference model per configured model key."""
    world = load_world(agent.resolved_fixture_world_path)
    matcher = TermMatcher(agent.matching)

    def build(provider: ProviderName, model: str) -> LLMClient:
        return FixtureWorldModel(world, f"{provider}:{model}", matcher)

    return build
