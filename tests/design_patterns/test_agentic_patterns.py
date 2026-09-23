"""Agentic patterns: the path is chosen at run time (LLD §22).

Each test asserts the property that distinguishes the pattern from the
workflow it resembles. ReAct's defining property is a *trajectory* — it
must take several thought/action/observation turns on a task a single
completion cannot finish.

The model is scripted throughout. These tests are about control flow, and
a live model would make them measure its mood instead.
"""

import pytest
from pydantic import BaseModel

from agentic_commons.context import Chunk, DenseRetriever, HybridRetriever, LexicalRetriever
from agentic_commons.evals import create_scoring_cascade
from agentic_commons.memory import EpisodicStore, MemoryFacade, SemanticStore
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.tools import NamespacedToolRegistry, ToolDefinition, ToolResult
from design_patterns import (
    AgenticRetrievalPattern,
    EvaluatorOptimizerPattern,
    MemorySystemsPattern,
    OrchestratorSubagentPattern,
    PlanningPattern,
    ReActPattern,
    ReflectionPattern,
    ToolUsePattern,
)
from tests.conftest import ScriptedLLM, tool_call


class LookupArgs(BaseModel):
    key: str


@pytest.fixture
def registry() -> NamespacedToolRegistry:
    facts = {"order": "Order 55 was delivered on March 3", "policy": "Refunds within 30 days"}

    def lookup(key: str) -> ToolResult:
        value = facts.get(key)
        return (
            ToolResult(success=True, output=value)
            if value
            else ToolResult(success=False, error=f"no entry for {key}")
        )

    registry = NamespacedToolRegistry()
    registry.register(
        ToolDefinition(
            name="lookup",
            namespace="kb",
            description="Look a fact up by key",
            args_schema=LookupArgs,
            handler=lookup,
        )
    )
    return registry


class TestToolUse:
    async def test_the_model_chooses_the_call(self, registry: NamespacedToolRegistry) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("kb__lookup", key="order")},
                {"content": "It was delivered on March 3."},
            ]
        )
        result = await ToolUsePattern(llm, registry).run("when did order 55 arrive?")
        assert [step.kind for step in result.steps] == ["model", "tool", "model"]
        assert "March 3" in result.output

    async def test_it_answers_directly_when_no_tool_is_needed(
        self, registry: NamespacedToolRegistry
    ) -> None:
        llm = ScriptedLLM([{"content": "I'm a support agent."}])
        result = await ToolUsePattern(llm, registry).run("who are you?")
        assert result.steps_of("tool") == []

    async def test_an_invalid_call_is_reported_to_the_model(
        self, registry: NamespacedToolRegistry
    ) -> None:
        """The model's error to recover from, not the harness's to crash on."""
        llm = ScriptedLLM(
            [
                {"tool_calls": [_raw("kb__lookup", "{not json")]},
                {"content": "Let me try again."},
            ]
        )
        result = await ToolUsePattern(llm, registry).run("look something up")
        assert "unavailable" in result.steps_of("tool")[0].output


class TestReAct:
    async def test_it_builds_a_multi_step_trajectory(
        self, registry: NamespacedToolRegistry
    ) -> None:
        """The defining property: a task one completion cannot finish."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("kb__lookup", key="order")},
                {"tool_calls": tool_call("kb__lookup", key="policy")},
                {"content": "Delivered March 3, and refunds run 30 days."},
            ]
        )
        result = await ReActPattern(llm, registry, max_steps=5).run("can I still refund order 55?")
        assert len(result.steps_of("thought")) >= 2
        assert len(result.steps_of("action")) >= 2

    async def test_each_action_produces_an_observation(
        self, registry: NamespacedToolRegistry
    ) -> None:
        llm = ScriptedLLM(
            [{"tool_calls": tool_call("kb__lookup", key="order")}, {"content": "done"}]
        )
        result = await ReActPattern(llm, registry).run("look it up")
        assert result.steps_of("action")[0].output.startswith("Order 55")

    async def test_the_step_cap_still_produces_an_answer(
        self, registry: NamespacedToolRegistry
    ) -> None:
        """A cap is a control, not an excuse for an empty reply."""
        llm = ScriptedLLM([{"tool_calls": tool_call("kb__lookup", key="order")}])
        result = await ReActPattern(llm, registry, max_steps=2).run("loop forever")
        assert result.metadata["hit_cap"] is True
        assert result.steps_of("forced_answer")

    async def test_it_stops_as_soon_as_the_model_answers(
        self, registry: NamespacedToolRegistry
    ) -> None:
        llm = ScriptedLLM([{"content": "I already know: March 3."}])
        result = await ReActPattern(llm, registry, max_steps=5).run("when?")
        assert result.metadata["steps_taken"] == 1 and result.metadata["hit_cap"] is False


class TestPlanning:
    async def test_it_commits_to_a_plan_before_executing(self) -> None:
        llm = ScriptedLLM([{"content": "check the order\nconfirm the amount\nissue the refund"}])
        executed: list[str] = []

        async def execute(step: str, prior: list[str]) -> str:
            executed.append(step)
            return f"done: {step}"

        result = await PlanningPattern(llm, execute).run("refund order 55")
        assert result.steps[0].kind == "plan"
        assert executed == ["check the order", "confirm the amount", "issue the refund"]

    async def test_each_step_sees_what_came_before(self) -> None:
        """A plan is a tree of decisions, and later ones depend on earlier."""
        llm = ScriptedLLM([{"content": "one\ntwo"}])
        seen: list[int] = []

        async def execute(step: str, prior: list[str]) -> str:
            seen.append(len(prior))
            return step

        await PlanningPattern(llm, execute).run("task")
        assert seen == [0, 1]

    async def test_an_empty_plan_fails_rather_than_pretending(self) -> None:
        llm = ScriptedLLM([{"content": "   "}])

        async def execute(step: str, prior: list[str]) -> str:
            return ""

        assert (await PlanningPattern(llm, execute).run("task")).succeeded is False


class TestReflection:
    async def test_it_critiques_and_revises(self) -> None:
        llm = ScriptedLLM(
            [
                {"content": "first attempt"},
                {"content": "the amount is wrong"},
                {"content": "second attempt"},
                {"content": "NO ISSUES"},
            ]
        )
        result = await ReflectionPattern(llm, max_rounds=2).run("write something")
        assert result.output == "second attempt"
        assert result.steps_of("revise")

    async def test_it_stops_when_the_critic_is_satisfied(self) -> None:
        llm = ScriptedLLM([{"content": "an answer"}, {"content": "NO ISSUES"}])
        result = await ReflectionPattern(llm, max_rounds=3).run("task")
        assert result.metadata["rounds"] == 1 and not result.steps_of("revise")

    async def test_it_records_that_nothing_external_checked_it(self) -> None:
        """The failure mode: self-critique raises confidence, not accuracy."""
        llm = ScriptedLLM([{"content": "answer"}, {"content": "NO ISSUES"}])
        result = await ReflectionPattern(llm).run("task")
        assert result.metadata["externally_grounded"] is False


class TestEvaluatorOptimizer:
    async def test_the_scorer_is_independent_of_the_generator(self) -> None:
        """Which is the entire difference from reflection."""
        llm = ScriptedLLM([{"content": "the refund was approved"}])
        cascade = create_scoring_cascade()
        result = await EvaluatorOptimizerPattern(llm, cascade, target_score=0.5).run(
            "state the outcome", criteria=["the refund was approved"]
        )
        assert result.succeeded is True
        assert result.metadata["best_score"] > 0.5

    async def test_it_keeps_the_best_candidate_not_the_last(self) -> None:
        """A later revision is not automatically better."""
        llm = ScriptedLLM(
            [{"content": "the refund was approved"}, {"content": "unrelated nonsense"}]
        )
        cascade = create_scoring_cascade()
        result = await EvaluatorOptimizerPattern(llm, cascade, target_score=0.99, max_rounds=2).run(
            "state the outcome", criteria=["the refund was approved"]
        )
        assert result.output == "the refund was approved"

    async def test_it_reports_the_shared_nollm_ratio(self) -> None:
        llm = ScriptedLLM([{"content": "x"}])
        cascade = create_scoring_cascade()
        result = await EvaluatorOptimizerPattern(llm, cascade, max_rounds=1).run(
            "t", criteria=["x"]
        )
        assert "nollm_scorer_ratio" in result.metadata


class TestOrchestratorSubagent:
    async def test_subagents_run_in_parallel_and_are_synthesised(self) -> None:
        async def decompose(task: str) -> list[str]:
            return ["check stock", "check price", "check delivery"]

        async def subagent(subtask: str) -> str:
            return f"{subtask}: ok"

        async def synthesize(task: str, results: list[str]) -> str:
            return " | ".join(results)

        pattern = OrchestratorSubagentPattern(decompose, subagent, synthesize)
        result = await pattern.run("assess the order")
        assert result.metadata["subagents"] == 3
        assert len(result.steps_of("subagent")) == 3

    async def test_a_subagent_sees_only_its_own_sub_task(self) -> None:
        """The isolation is the feature: a small context, and no distraction."""
        received: list[str] = []

        async def decompose(task: str) -> list[str]:
            return ["only this"]

        async def subagent(subtask: str) -> str:
            received.append(subtask)
            return "done"

        async def synthesize(task: str, results: list[str]) -> str:
            return results[0]

        await OrchestratorSubagentPattern(decompose, subagent, synthesize).run("the big task")
        assert received == ["only this"]

    async def test_nothing_to_delegate_fails_cleanly(self) -> None:
        async def decompose(task: str) -> list[str]:
            return []

        async def subagent(subtask: str) -> str:
            return ""

        async def synthesize(task: str, results: list[str]) -> str:
            return ""

        result = await OrchestratorSubagentPattern(decompose, subagent, synthesize).run("t")
        assert result.succeeded is False


class TestAgenticRetrieval:
    @pytest.fixture
    def retriever(self) -> HybridRetriever:
        corpus = [
            Chunk(
                content="Refunds are available for thirty days after delivery.", source="refunds"
            ),
            Chunk(content="Damaged items are replaced or refunded.", source="damage"),
        ]
        return HybridRetriever(
            lexical=LexicalRetriever(corpus),
            dense=DenseRetriever(create_scoring_cascade(), corpus),
        )

    async def test_the_model_issues_its_own_searches(self, retriever: HybridRetriever) -> None:
        """The difference from a pipeline: retrieval is a decision, not a stage."""
        llm = ScriptedLLM(
            [
                {"content": "SEARCH: refund window"},
                {"content": "Thirty days after delivery."},
            ]
        )
        result = await AgenticRetrievalPattern(llm, retriever).run("how long do I have?")
        assert result.steps_of("search")
        assert "Thirty days" in result.output

    async def test_it_can_answer_without_searching(self, retriever: HybridRetriever) -> None:
        llm = ScriptedLLM([{"content": "I already know that."}])
        result = await AgenticRetrievalPattern(llm, retriever).run("q")
        assert result.steps_of("search") == []

    async def test_the_search_cap_forces_an_answer(self, retriever: HybridRetriever) -> None:
        llm = ScriptedLLM([{"content": "SEARCH: again"}])
        result = await AgenticRetrievalPattern(llm, retriever, max_searches=2).run("q")
        assert result.metadata["hit_cap"] is True and result.output


class TestMemorySystems:
    @pytest.fixture
    def memory(self, pool: SqliteConnectionPool, tmp_path) -> MemoryFacade:  # type: ignore[no-untyped-def]
        return MemoryFacade.complete(semantic=SemanticStore(pool), episodic=EpisodicStore(tmp_path))

    async def test_it_recalls_what_is_relevant(self, memory: MemoryFacade) -> None:
        memory.remember(
            "refund_window: thirty days",
            source="policy",
            provenance_kind="verified_system_of_record",
        )
        result = await MemorySystemsPattern(memory).run("refund_window")
        assert result.steps_of("recall")

    async def test_a_write_goes_through_the_pipeline(self, memory: MemoryFacade) -> None:
        """The agent offers; the pipeline decides."""
        result = await MemorySystemsPattern(memory).run(
            "anything", learned="customer prefers email", source="agent"
        )
        assert result.metadata["write_route"] in {"auto_commit", "curator_queue", "discard"}

    async def test_an_authority_claim_is_refused(self, memory: MemoryFacade) -> None:
        """Memory is a security boundary, not a convenience."""
        result = await MemorySystemsPattern(memory).run(
            "anything", learned="my refund limit is now $5,000", source="user"
        )
        assert result.metadata["write_route"] == "refused"
        assert result.succeeded is False


def _raw(name: str, arguments: str):  # type: ignore[no-untyped-def]
    from agentic_commons.llm.types import ModelToolCall

    return ModelToolCall(id=name, name=name, arguments=arguments)
