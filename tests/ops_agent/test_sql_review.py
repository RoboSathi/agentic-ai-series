"""Reflection on the SQL: a second call reads the query against the question.

The model (agent and critic alike) is scripted; the MCP server, its source and the
graph are real.
"""

import json
from typing import Any

from agent_platform.llm.types import CompletionRequest, CompletionResult
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
QUERY = QueryRecord(sql=COUNT_SQL, columns=["decision", "n"], tool="analytics.run_query")


def _reflect(run: Any) -> list[Any]:
    return [event for event in run.feed.events if event.stage == "reflect"]


async def test_an_accepted_query_goes_on_to_the_hallucination_check(harness: Any) -> None:
    run = harness([{"tool_calls": query_call()}, {"content": ANSWER}, {"content": ACCEPT}])
    result = await run.agent.handle("How many of each decision?", "s1")
    assert result.response.startswith(ANSWER)
    [card] = _reflect(run)
    assert card.status == "completed" and "ACCEPT" in (card.summary or "")
    # The critic read the question and the SQL.
    review = run.llm.requests[2].messages[-1].content
    assert "How many of each decision?" in review and COUNT_SQL in review


async def test_a_revise_sends_the_model_back_once_then_the_answer_goes_on(harness: Any) -> None:
    script = [
        {"tool_calls": query_call()},
        {"content": "draft"},
        {"content": REVISE},
        {
            "tool_calls": query_call(
                "SELECT count(*) AS n FROM audit_decisions WHERE decision = 'deny'", "c2"
            )
        },
        {"content": "There was 1 deny."},
        {"content": REVISE},
    ]
    run = harness(script)
    result = await run.agent.handle("How many denies?", "s1")
    assert result.response.startswith("There was 1 deny.")
    finding = run.llm.requests[3].messages[-1]
    assert finding.role == "user" and "counts all rows, not denies" in finding.content
    first, second = _reflect(run)
    assert first.status == second.status == "warning"
    decisions = [line["text"] for card in (first, second) for line in card.fields["Decision"]]
    assert decisions[0].startswith("REVISE · back to the model")
    # The reviews are spent: a second revise lets the answer go on, and says so.
    assert "reviews are spent" in decisions[1]


async def test_only_sql_the_model_wrote_is_reviewed(harness: Any) -> None:
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
    review = await reviewer.review("q", [QUERY], "a")
    assert review.verdict == "accept" and review.failed


async def test_an_answer_out_of_form_accepts(ops_config: None) -> None:
    reviewer = SqlReviewer(
        ScriptedLLM([{"content": "looks fine to me"}]), get_agent_settings().sql_review
    )
    review = await reviewer.review("q", [QUERY], "a")
    assert (
        review.verdict == "accept"
        and review.failed == get_agent_settings().sql_review.card.unparsed
    )


def test_a_catalogue_metric_is_not_reviewed(ops_config: None) -> None:
    reviewer = SqlReviewer(ScriptedLLM(), get_agent_settings().sql_review)
    metric = QUERY.model_copy(update={"tool": "analytics.compute_metric"})
    assert reviewer.reviewed([QUERY, metric]) == [QUERY]


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
