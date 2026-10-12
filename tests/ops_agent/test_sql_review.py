"""REFLECT: after a data query runs and before the model answers from it.

Model-written SQL (run_query) is read by a critic against the question; a catalogue call
(compute_metric) is decided in code, with no model call. The model (agent and critic
alike) is scripted; the MCP server, its source and the graph are real.
"""

import json
from typing import Any

from agent_platform.llm.types import CompletionRequest, CompletionResult, ModelToolCall
from ops_agent.config import get_agent_settings
from ops_agent.domain import QueryRecord
from ops_agent.guardrails import SqlReviewer
from tests.conftest import ScriptedLLM
from tests.ops_agent.conftest import COUNT_SQL, query_call

ANSWER = "There were 2 allows and 1 deny."
ACCEPT = json.dumps({"verdict": "accept", "rationale": "it counts by decision"})
REVISE = json.dumps(
    {"verdict": "revise", "issues": ["counts all rows, not denies"], "hint": "filter on deny"}
)
DENY_SQL = "SELECT count(*) AS n FROM audit_decisions WHERE decision = 'deny'"
QUERY = QueryRecord(sql=COUNT_SQL, columns=["decision", "n"], tool="analytics.run_query")


def _reflect(run: Any) -> list[Any]:
    return [event for event in run.feed.events if event.stage == "reflect"]


def _order(run: Any, *stages: str) -> list[str]:
    return [event.stage for event in run.feed.events if event.stage in stages]


def _metric_call(call_id: str = "m1") -> list[ModelToolCall]:
    arguments = json.dumps({"name": "tool_call_validity", "days": 7})
    return [ModelToolCall(id=call_id, name="analytics__compute_metric", arguments=arguments)]


async def test_reflect_comes_after_the_result_and_before_the_answer(harness: Any) -> None:
    run = harness([{"tool_calls": query_call()}, {"content": ACCEPT}, {"content": ANSWER}])
    result = await run.agent.handle("How many of each decision?", "s1")
    assert result.response.startswith(ANSWER)
    [card] = _reflect(run)
    assert card.status == "completed" and "ACCEPT" in (card.summary or "")
    # The critic read the question and the SQL, and no draft answer existed yet.
    review = run.llm.requests[1].messages[-1].content
    assert "How many of each decision?" in review and COUNT_SQL in review
    order = _order(run, "tool_result", "reflect", "llm_response_received")
    last_result = max(i for i, stage in enumerate(order) if stage == "tool_result")
    assert order.index("reflect") > last_result
    assert order[-1] == "llm_response_received"
    # The tool step, the review, one answer.
    assert len(run.llm.requests) == 3


async def test_a_revise_is_fixed_before_anything_is_answered(harness: Any) -> None:
    script = [
        {"tool_calls": query_call()},
        {"content": REVISE},
        {"tool_calls": query_call(DENY_SQL, "c2")},
        {"content": ACCEPT},
        {"content": "There was 1 deny."},
    ]
    run = harness(script)
    result = await run.agent.handle("How many denies?", "s1")
    assert result.response.startswith("There was 1 deny.")
    # The finding reached the model's next step, before it wrote any answer.
    finding = run.llm.requests[2].messages[-1]
    assert finding.role == "user" and "counts all rows, not denies" in finding.content
    first, second = _reflect(run)
    assert first.status == "warning" and second.status == "completed"
    assert first.fields["Decision"][0]["text"].startswith("REVISE · the model fixes the query")
    # Two tool steps, two reviews, one answer.
    assert len(run.llm.requests) == 5


async def test_past_the_cap_the_model_goes_on_unreviewed(harness: Any) -> None:
    limit = get_agent_settings().sql_review.max_reviews
    script: list[dict[str, Any]] = []
    for index in range(limit):
        # Distinct SQL each time: the loop governor suppresses a repeated call.
        sql = f"SELECT decision FROM audit_decisions LIMIT {index + 1}"
        script += [{"tool_calls": query_call(sql, f"c{index}")}, {"content": REVISE}]
    script += [{"tool_calls": query_call(DENY_SQL, "last")}, {"content": "There was 1 deny."}]
    run = harness(script)
    await run.agent.handle("How many denies?", "s1")
    cards = _reflect(run)
    assert len(cards) == limit + 1
    assert "reviews are spent" in cards[limit - 1].fields["Decision"][0]["text"]
    # The last step's SQL goes to no critic: the card says it was not reviewed.
    assert cards[-1].summary == get_agent_settings().sql_review.card.capped_heading


async def test_a_catalogue_call_is_decided_in_code_without_a_model_call(harness: Any) -> None:
    run = harness([{"tool_calls": _metric_call()}, {"content": "Validity was 1."}])
    await run.agent.handle("tool call validity, last week?", "s1")
    [card] = _reflect(run)
    words = get_agent_settings().sql_review.card
    assert card.status == "completed" and card.summary == words.deterministic_heading
    assert card.fields["Model"][0].startswith("no model call · decided in code")
    assert card.fields["Reasoning"] == words.deterministic_reasoning
    assert card.fields["Asked"][1] == "compute_metric(name=tool_call_validity, days=7)"
    # No review call: the tool step, then the answer.
    assert len(run.llm.requests) == 2


async def test_several_queries_in_one_step_share_one_review(harness: Any) -> None:
    both = query_call() + query_call(DENY_SQL, "c2")
    run = harness([{"tool_calls": both}, {"content": ACCEPT}, {"content": ANSWER}])
    await run.agent.handle("How many of each, and denies?", "s1")
    assert len(_reflect(run)) == 1
    assert len(run.llm.requests) == 3
    assert DENY_SQL in run.llm.requests[1].messages[-1].content


async def test_a_turn_with_no_data_query_has_no_reflect(harness: Any) -> None:
    run = harness([{"content": "No query needed."}])
    await run.agent.handle("hello", "s1")
    assert _reflect(run) == []


async def test_without_the_guard_there_is_no_review(harness: Any) -> None:
    script = [{"tool_calls": query_call()}, {"content": ANSWER}]
    run = harness(script, features={"hallucination_guard": False})
    await run.agent.handle("How many?", "s1")
    assert _reflect(run) == [] and len(run.llm.requests) == 2


class Failing(ScriptedLLM):
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        raise TimeoutError("no answer")


async def test_a_critic_that_fails_accepts_and_says_so(ops_config: None) -> None:
    reviewer = SqlReviewer(Failing(), get_agent_settings().sql_review)
    review = await reviewer.review("q", [QUERY])
    assert review.verdict == "accept" and review.failed


async def test_an_answer_out_of_form_accepts(ops_config: None) -> None:
    reviewer = SqlReviewer(
        ScriptedLLM([{"content": "looks fine to me"}]), get_agent_settings().sql_review
    )
    review = await reviewer.review("q", [QUERY])
    assert review.verdict == "accept"
    assert review.failed == get_agent_settings().sql_review.card.unparsed


def test_only_model_written_sql_goes_to_the_critic(ops_config: None) -> None:
    reviewer = SqlReviewer(ScriptedLLM(), get_agent_settings().sql_review)
    metric = QUERY.model_copy(update={"tool": "analytics.compute_metric"})
    listing = QUERY.model_copy(update={"tool": "analytics.describe_view"})
    assert reviewer.data([QUERY, metric, listing]) == [QUERY, metric]
    assert reviewer.reviewed([QUERY, metric, listing]) == [QUERY]


async def test_a_rejected_query_is_a_failed_tool_result(harness: Any) -> None:
    """The shared TOOL RESULT card (`StepEvents.tool_outcome`) names how far a call got."""
    run = harness(
        [{"tool_calls": query_call("DELETE FROM audit_decisions")}, {"content": "Could not."}]
    )
    await run.agent.handle("Delete them", "s1")
    [result] = [
        event
        for event in run.feed.events
        if event.stage == "tool_result"
        and event.fields.get("qualified_tool") == "analytics.run_query"
    ]
    assert result.status == "failed" and (result.summary or "").endswith("failed")
    # A refused query returned no rows: nothing to reflect on.
    assert _reflect(run) == []
