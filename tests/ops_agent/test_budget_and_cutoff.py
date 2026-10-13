"""A spent tool budget and a cut-off answer, said as what they are.

A call the loop stops is neither run nor charged, is not called a Gatekeeper refusal, and
the model is then offered no tools; an answer the token cap cut short is retried once with
a larger cap, and never sent cut off.
"""

import json
from typing import Any

from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    ModelToolCall,
    TokenUsage,
)
from agent_platform.loop_control import LoopBudget, build_loop_governor
from agent_platform.tools.types import ToolCall
from ops_agent.config import get_agent_settings
from tests.conftest import ScriptedLLM

ANSWER = "Validity was 1."


ACCEPT = json.dumps({"verdict": "accept", "rationale": "fine"})


def _past_convergence(run: Any) -> Any:
    """Near-identical rows read as no progress and end the turn; these tests are about what
    happens next, so the plateau check is set aside (every other check still runs)."""
    run.agent._nodes._calls.governor._convergence.plateaued = lambda gains: False
    return run


def _query_calls(count: int) -> list[ModelToolCall]:
    """`count` queries in one step, each with its own result (none a repeat, none alike)."""
    return [
        ModelToolCall(
            id=f"q{index}",
            name="analytics__run_query",
            arguments=json.dumps({"sql": f"SELECT count(*) + {index} AS n FROM audit_decisions"}),
        )
        for index in range(count)
    ]


async def test_a_stopped_call_is_not_charged(ops_config: None) -> None:
    governor = build_loop_governor(LoopBudget(max_steps=8, max_tool_calls=2))
    governor.begin_turn()
    governor.begin_step()
    decisions = [
        (await governor.before_action(ToolCall(action=f"a.t{n}", params={"n": n}))).decision
        for n in range(4)
    ]
    assert decisions == ["continue", "continue", "finalize", "finalize"]
    assert governor.spent("tool_calls") == 2


async def test_a_spent_budget_is_not_called_a_gatekeeper_refusal(harness: Any) -> None:
    budget = get_agent_settings().budget.max_tool_calls
    script = [{"tool_calls": _query_calls(budget + 1)}, {"content": ACCEPT}, {"content": ANSWER}]
    run = _past_convergence(harness(script))
    await run.agent.handle("tool call validity, every period?", "s1")
    stopped = [
        event
        for event in run.feed.events
        if event.stage == "tool_result" and event.status == "warning"
    ]
    assert stopped
    for event in stopped:
        summary = event.summary or ""
        assert "Gatekeeper" not in summary
        assert summary.endswith(f"not run · tool_budget · tool_calls: {budget} spent of {budget}")
    # What the model read: the budget, not a refusal.
    told = [m.content for m in run.llm.requests[-1].messages if m.role == "tool"]
    assert any("tool-call budget is spent" in text for text in told)
    assert not any("The Gatekeeper refused" in text for text in told)


async def test_once_the_budget_is_spent_no_tools_are_offered(harness: Any) -> None:
    budget = get_agent_settings().budget.max_tool_calls
    script = [{"tool_calls": _query_calls(budget)}, {"content": ACCEPT}, {"content": ANSWER}]
    run = _past_convergence(harness(script))
    await run.agent.handle("tool call validity, every period?", "s1")
    last = run.llm.requests[-1]
    assert not last.tools
    assert last.messages[-1].content == get_agent_settings().words.tools_spent


class CutOff(ScriptedLLM):
    """Answers cut off at the cap `cut` times, then answers in full."""

    def __init__(self, cut: int) -> None:
        super().__init__()
        self._cut = cut

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        cut = len(self.requests) <= self._cut
        return CompletionResult(
            content="Guard" if cut else ANSWER,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tier="scripted",
            finish_reason=get_agent_settings().cut_off_finish if cut else "stop",
        )


async def _turn(harness: Any, llm: ScriptedLLM) -> Any:
    run = harness([{"content": "unused"}])
    run.agent._nodes._llm = llm  # the scripted model whose answers are cut off
    return await run.agent.handle("how is validity?", "s1"), llm


async def test_a_cut_off_answer_is_retried_with_the_larger_cap(harness: Any) -> None:
    result, llm = await _turn(harness, CutOff(cut=1))
    settings = get_agent_settings()
    assert result.response.startswith(ANSWER)
    assert [r.max_tokens for r in llm.requests] == [settings.max_tokens, settings.retry_max_tokens]


async def test_an_answer_cut_off_twice_is_never_sent(harness: Any) -> None:
    result, _ = await _turn(harness, CutOff(cut=2))
    assert "Guard" not in result.response
    assert "cut off" in result.response and result.outcome == "unavailable"
