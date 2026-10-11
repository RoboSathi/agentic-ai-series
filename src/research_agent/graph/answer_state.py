"""The direct answer's state, and what belongs to one question but is not state.

messages       the conversation the model reads: system prompt, history, question,
               then each step's tool calls and their results
sources        the URLs the tools read, for the reply's Sources line
observations   what each successful tool call returned: the only thing that can ground
               a claim (an error or a refusal grounds nothing) — a search as one per hit
pages          the URLs read in full (fetched), as against seen only as a search snippet
"""

import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig

from agent_platform.hallucination_check import Observation
from agent_platform.llm.prompt import TurnLayout
from agent_platform.llm.types import Message
from agent_platform.loop_control import LoopGovernor
from agent_platform.observability.loop_reporter import LoopReporter
from agent_platform.observability.step_events import StepEvents
from research_agent.context.session import SessionContext

# Budget counters of one answer, beside `TOKENS` (research_agent.domain.budget).
STEPS, TOOL_CALLS = "steps", "tool_calls"
# Where a turn travels in the graph's config: it is per question, not state.
TURN_KEY = "turn"


class AnswerState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    sources: Annotated[list[str], operator.add]
    observations: Annotated[list[Observation], operator.add]
    pages: Annotated[list[str], operator.add]
    text: str
    # The budget counter that cut the loop short ("" when the model finished by itself).
    stop: str
    # Why no answer could be given ("" when one was).
    failure: str
    # How well the answer held up against its sources (None: not checked).
    quality: float | None


@dataclass
class Turn:
    """What belongs to one question and is not state: its budget, watchdog and step cards."""

    question: str
    # The one authority over this question's loop: steps, tool calls, tokens, the deadline,
    # repeats, cycles and convergence.
    governor: LoopGovernor
    context: SessionContext
    loop: LoopReporter
    # The platform's loop cards (LOOP CONTROL), numbered with the same step.
    steps: StepEvents
    # When the question was asked: "today" for the prompt and for the recency check.
    asked_at: datetime
    # The opening every step's request shares, so each CONTEXT card names its sections.
    layout: TurnLayout = field(default_factory=TurnLayout)

    def spent(self, counter: str) -> float:
        return self.governor.spent(counter)

    def elapsed(self) -> float:
        return self.governor.elapsed()


def turn_of(config: RunnableConfig) -> Turn:
    turn = config["configurable"][TURN_KEY]
    assert isinstance(turn, Turn)
    return turn


def initial_state(system: str, question: str, history: list[Message]) -> AnswerState:
    return {
        "messages": [
            Message(role="system", content=system),
            *history,
            Message(role="user", content=question),
        ],
        "sources": [],
        "observations": [],
        "pages": [],
        "text": "",
        "stop": "",
        "failure": "",
        "quality": None,
    }


__all__ = [
    "STEPS",
    "TOOL_CALLS",
    "TURN_KEY",
    "AnswerState",
    "Turn",
    "initial_state",
    "turn_of",
]
