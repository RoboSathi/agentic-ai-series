"""One question's state through the graph.

messages   what the model reads: system prompt, history, the question, then each
           step's tool calls and their results
queries    every query that returned rows: the only evidence a figure can rest on
"""

import operator
from typing import Annotated, TypedDict

from agent_platform.llm.types import Message
from ops_agent.domain.query import QueryRecord


class OpsState(TypedDict):
    session_id: str
    messages: Annotated[list[Message], operator.add]
    queries: Annotated[list[QueryRecord], operator.add]
    steps: int
    tool_calls: int
    tool_failures: int
    # The model's own answer, before the figures check and the query appendix.
    answer: str
    response: str
    # Figures in the answer no query result supports (None when the check is withheld).
    unverified: int | None


def initial_state(session_id: str, messages: list[Message]) -> OpsState:
    return OpsState(
        session_id=session_id,
        messages=messages,
        queries=[],
        steps=0,
        tool_calls=0,
        tool_failures=0,
        answer="",
        response="",
        unverified=None,
    )


__all__ = ["OpsState", "QueryRecord", "initial_state"]
