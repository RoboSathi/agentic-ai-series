"""Isolated sub-graph runs, and the generic human-in-the-loop wait."""

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentic_commons.durability import SUMMARY, WAITING_ON
from agentic_commons.durability.hitl import await_decision
from agentic_commons.durability.isolated import IsolatedGraphRunner, UndeclaredInputError


class EchoState(TypedDict, total=False):
    question: str
    answer: str


def _echo_graph() -> Any:
    def answer(state: EchoState) -> dict[str, str]:
        return {"answer": state["question"].upper()}

    builder = StateGraph(EchoState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


class TestIsolatedGraphRunner:
    async def test_it_runs_on_exactly_the_inputs_given(self) -> None:
        runner = IsolatedGraphRunner(_echo_graph, frozenset({"question"}))
        result = await runner.run({"question": "why"})
        assert result["answer"] == "WHY"

    async def test_an_undeclared_input_is_refused(self) -> None:
        runner = IsolatedGraphRunner(_echo_graph, frozenset({"question"}))
        with pytest.raises(UndeclaredInputError):
            await runner.run({"question": "why", "history": ["everything said so far"]})

    async def test_each_run_builds_a_fresh_graph(self) -> None:
        built: list[int] = []

        def factory() -> Any:
            built.append(1)
            return _echo_graph()

        runner = IsolatedGraphRunner(factory, frozenset({"question"}))
        await runner.run({"question": "a"})
        await runner.run({"question": "b"})
        assert len(built) == 2 and runner.allowed_inputs == frozenset({"question"})


class WaitState(TypedDict, total=False):
    decision: str


class TestAwaitDecision:
    async def test_it_interrupts_then_returns_the_resume_value(self) -> None:
        def wait(state: WaitState) -> dict[str, str]:
            return {"decision": str(await_decision("user:s1", "Waiting on the user", step=1))}

        builder = StateGraph(WaitState)
        builder.add_node("wait", wait)
        builder.add_edge(START, "wait")
        builder.add_edge("wait", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "t1"}}
        await graph.ainvoke({}, config)
        snapshot = await graph.aget_state(config)
        payload = snapshot.interrupts[0].value
        assert payload[WAITING_ON] == "user:s1" and payload[SUMMARY] == "Waiting on the user"
        assert payload["step"] == 1
        final = await graph.ainvoke(Command(resume="yes"), config)
        assert final["decision"] == "yes"
