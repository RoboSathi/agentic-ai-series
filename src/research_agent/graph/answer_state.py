"""The direct answer's state, and what belongs to one question but is not state.

messages       the conversation the model reads: system prompt, history, question,
               then each step's tool calls and their results
sources        the URLs the tools read, for the reply's Sources line
observations   what each successful tool call returned: the only thing that can ground
               a claim (an error or a refusal grounds nothing)
"""

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig

from agentic_commons.control_loop import BudgetGuard
from agentic_commons.control_loop.watchdogs import ToolControlLoop
from agentic_commons.hallucination_protection import Observation
from agentic_commons.llm.types import Message
from agentic_commons.observability.loop_reporter import LoopReporter
from research_agent.context.session import SessionContext

# Budget counters of one answer, beside `TOKENS` (research_agent.domain.budget).
STEPS, TOOL_CALLS = "steps", "tool_calls"
# Where a turn travels in the graph's config: it is per question, not state.
TURN_KEY = "turn"


class AnswerState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    sources: Annotated[list[str], operator.add]
    observations: Annotated[list[Observation], operator.add]
    text: str
    # The budget counter that cut the loop short ("" when the model finished by itself).
    stop: str
    # Why no answer could be given ("" when one was).
    failure: str


@dataclass
class Turn:
    """What belongs to one question and is not state: its budget, watchdog and step cards."""

    question: str
    budget: BudgetGuard
    control: ToolControlLoop
    context: SessionContext
    loop: LoopReporter

    def spent(self, counter: str) -> float:
        return self.budget.snapshot()[counter].spent


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
        "text": "",
        "stop": "",
        "failure": "",
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
