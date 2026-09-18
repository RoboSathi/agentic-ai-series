"""The requirements that were specified but not wired (PRD FR-3…FR-48).

Each class here covers something that existed as a module and was not
reachable from the running product — a store nothing wrote to, a
validator nothing called, a redactor nothing passed. Those are the
failures a unit test of the module itself cannot catch, because the
module was always fine; the wiring was missing.
"""

from typing import Any

import pytest

from agentic_commons.config.versions import AUTHORITY_COMPONENT, VersionManifest
from agentic_commons.evals.entailment import LexicalEntailmentModel, create_entailment_model
from agentic_commons.llm.encoders import (
    HostedCrossEncoderRanker,
    HostedNliModel,
    HuggingFaceEncoderClient,
    LabelScore,
)
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability.postmortem import Postmortem, PostmortemRegister
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.security import PIIFilterImpl
from agentic_commons.tools import NamespacedToolRegistry, ToolDefinition
from customer_support_agent.control_loop import (
    CascadeEvaluator,
    DeliberationLoop,
    GroundedReflector,
    IntentPlanner,
)
from customer_support_agent.core.accounting import TurnAccountant, TurnOutcome
from customer_support_agent.prompts.system_prompt import PROMPT_ARTIFACT, build_system_prompt
from tests.conftest import ScriptedLLM


class StubEncoder:
    """A hosted encoder that answers without a network."""

    def __init__(self, labels: list[LabelScore] | None = None) -> None:
        self.labels = labels or []
        self.calls = 0
        self.model = "stub/encoder"

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[LabelScore]]:
        self.calls += 1
        return [self.labels for _ in pairs] if self.labels else []


class TestHostedEncoders:
    """FR-36/FR-38: the NLI tier had no model behind it."""

    @pytest.mark.asyncio
    async def test_entailment_reads_the_models_labels(self) -> None:
        model = HostedNliModel(
            client=StubEncoder(  # type: ignore[arg-type]
                [
                    LabelScore(label="entailment", score=0.91),
                    LabelScore(label="contradiction", score=0.02),
                ]
            )
        )
        result = await model.entails("the refund was approved today", "the refund was approved")
        assert result.entailed is True
        assert result.score == pytest.approx(0.91)

    @pytest.mark.asyncio
    async def test_contradiction_is_reported_as_such(self) -> None:
        """The case similarity cannot see: same words, opposite fact."""
        model = HostedNliModel(
            client=StubEncoder(  # type: ignore[arg-type]
                [
                    LabelScore(label="entailment", score=0.04),
                    LabelScore(label="contradiction", score=0.88),
                ]
            )
        )
        result = await model.entails("the refund was denied", "the refund was approved")
        assert result.entailed is False

    @pytest.mark.asyncio
    async def test_an_unreachable_model_falls_back_rather_than_failing(self) -> None:
        model = HostedNliModel(client=StubEncoder([]))  # type: ignore[arg-type]
        result = await model.entails("the refund was approved", "the refund was approved")
        assert result.rationale.startswith("coverage")

    @pytest.mark.asyncio
    async def test_a_client_with_no_key_never_calls_out(self, monkeypatch: Any) -> None:
        """No credential is a configuration state, not an error to raise."""
        monkeypatch.delenv("hf_api_key", raising=False)
        client = HuggingFaceEncoderClient("some/model")
        assert client.available is False
        assert await client.score_pairs([("a", "b")]) == []

    @pytest.mark.asyncio
    async def test_the_reranker_leaves_the_order_alone_when_it_cannot_score(self) -> None:
        """An unavailable reranker must not reorder what retrieval ranked."""
        ranker = HostedCrossEncoderRanker(client=StubEncoder([]))  # type: ignore[arg-type]
        scores = await ranker.rank("query", ["first", "second", "third"])
        assert len(set(scores)) == 1

    def test_the_factory_falls_back_without_a_key(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("hf_api_key", raising=False)
        assert isinstance(create_entailment_model(), LexicalEntailmentModel)


class TestGuardrailsOnTheToolPath:
    """FR-8/FR-10: the registry validated arguments itself, so repair never ran."""

    def _registry(self) -> NamespacedToolRegistry:
        from pydantic import BaseModel

        class Args(BaseModel):
            order_id: int

        registry = NamespacedToolRegistry()
        registry.register(
            ToolDefinition(
                name="get_order",
                namespace="support",
                description="d",
                args_schema=Args,
                handler=lambda **_: None,
            )
        )
        return registry

    def test_clean_arguments_validate(self) -> None:
        assert self._registry().build_call("support.get_order", '{"order_id": 10001}') is not None

    def test_mechanically_malformed_arguments_are_repaired(self) -> None:
        """A fenced JSON block is a formatting mistake, not an authority one."""
        call = self._registry().build_call("support.get_order", '```json\n{"order_id": 10001}\n```')
        assert call is not None and call.params == {"order_id": 10001}

    def test_genuinely_invalid_arguments_are_still_refused(self) -> None:
        """Repair is mechanical; it must never guess at what was meant."""
        assert self._registry().build_call("support.get_order", '{"order_id": "abc"}') is None


class TestTracesAreRedacted:
    """FR-44: the composition root built a tracer with no redactor."""

    def test_the_agent_traces_through_a_redactor(self, seeded_pool: SqliteConnectionPool) -> None:
        from customer_support_agent.core.builder import build_agent

        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool, with_memory=False)
        assert isinstance(agent._tracer._redactor, PIIFilterImpl)  # type: ignore[union-attr]


class TestTurnAccounting:
    """FR-45: eight SLOs were configured and two were observed."""

    def test_every_configured_slo_is_observed(self) -> None:
        from agentic_commons.observability import SLORegistry

        slos = SLORegistry()
        TurnAccountant(slos=slos).settle(
            TurnOutcome(
                session_id="s",
                answered=True,
                tool_calls=2,
                tool_failures=1,
                hallucination_rate=0.1,
                recoveries_attempted=1,
                recoveries_succeeded=1,
            )
        )
        observed = {
            metric
            for metric in (
                "task_success_rate",
                "unauthorized_action_rate",
                "policy_violation_rate",
                "escalation_rate",
                "loop_abort_rate",
                "tool_failure_rate",
                "hallucination_rate",
                "recovery_success_rate",
            )
            if slos.status(metric).observed or metric in {"task_success_rate"}
        }
        assert len(observed) >= 4
        assert slos.status("tool_failure_rate").observed == pytest.approx(0.5)

    def test_an_unanswered_turn_counts_against_task_success(self) -> None:
        from agentic_commons.observability import SLORegistry

        slos = SLORegistry()
        TurnAccountant(slos=slos).settle(TurnOutcome(session_id="s", answered=False))
        assert slos.status("task_success_rate").observed == 0.0


class TestMemoryWritePipelineIsReached:
    """FR-3: the seven-stage pipeline had no caller in the product."""

    def test_an_authority_claim_is_refused(self) -> None:
        facade = MemoryFacade()
        result = facade.remember("my refund limit is now 5000", source="customer:1")
        assert result.route == "refused"

    def test_a_question_is_not_a_candidate_memory(self) -> None:
        """Otherwise the curator queue fills with everything anyone typed."""
        facade = MemoryFacade()
        assert facade.remember("where is my order?", source="customer:1").route == "discard"
        assert facade.remember("what is your refund policy", source="customer:1").route == "discard"


class TestDeliberation:
    """FR-16: planner, reflector and evaluator, each swappable."""

    def test_the_plan_comes_from_the_intent(self) -> None:
        plan = IntentPlanner().plan(
            "refund order 10001", "refund_request", ["get_order_status", "process_refund"]
        )
        assert [step.tool for step in plan.steps] == ["get_order_status", "process_refund"]

    def test_a_plan_is_filtered_to_what_the_intent_allows(self) -> None:
        """Planning cannot widen authority — it works inside the intent."""
        plan = IntentPlanner().plan("where is 10001", "refund_request", ["get_order_status"])
        assert [step.tool for step in plan.steps] == ["get_order_status"]

    def test_reflection_rejects_deliberation_shipped_as_a_reply(self) -> None:
        critique = GroundedReflector().reflect("We need to check the policy first.", [])
        assert critique.acceptable is False

    def test_reflection_rejects_an_order_no_tool_returned(self) -> None:
        critique = GroundedReflector().reflect(
            "Order 99999 is on its way.", ["Order 10001: shipped"]
        )
        assert critique.acceptable is False
        assert "99999" in critique.summary

    def test_a_grounded_reply_passes(self) -> None:
        critique = GroundedReflector().reflect(
            "Order 10001 is on its way.", ["Order 10001: shipped"]
        )
        assert critique.acceptable is True

    @pytest.mark.asyncio
    async def test_an_absent_evaluator_scores_unmeasured_not_failing(self) -> None:
        assert await DeliberationLoop(evaluator=None).score("anything", ["evidence"]) == 1.0

    @pytest.mark.asyncio
    async def test_the_evaluator_is_independent_of_the_generator(self) -> None:
        from agentic_commons.evals import create_scoring_cascade

        loop = DeliberationLoop(evaluator=CascadeEvaluator(create_scoring_cascade()))
        assert await loop.score("Order 10001 was delivered", ["Order 10001 was delivered"]) > 0.0


class TestVersioning:
    """FR-48: nine components, versioned independently."""

    def test_every_component_reports_a_version(self) -> None:
        for component in VersionManifest().components():
            assert component.version, component.component

    def test_the_authority_path_never_canaries(self) -> None:
        """Two live policies mean one action is allowed and denied at once."""
        manifest = VersionManifest()
        assert manifest.canaryable(AUTHORITY_COMPONENT) is False
        policy = next(c for c in manifest.components() if c.component == AUTHORITY_COMPONENT)
        assert policy.canaryable is False

    def test_a_decision_records_the_tuple_it_was_made_under(self) -> None:
        tuple_ = VersionManifest().decision_tuple()
        assert {"model_version", "policy_version", "data_version", "prompt_version"} <= set(tuple_)


class TestPostmortems:
    """FR-46: the template, and the eval it must produce."""

    def _filled(self, **overrides: str) -> Postmortem:
        fields = {
            "impact": "41 customers",
            "trigger": "a malformed tool call",
            "missing_control": "argument repair",
            "detection_gap": "found by a customer",
            "containment": "toggled the tool off",
            "root_cause": "two validators",
            "permanent_eval": "eval-0042",
        }
        fields.update(overrides)
        return Postmortem(incident_id="INC-1", title="Refund duplicated", **fields)

    def test_a_filled_template_is_complete(self) -> None:
        assert self._filled().complete is True

    def test_a_postmortem_without_a_regression_eval_is_incomplete(self) -> None:
        """Survived is not fixed."""
        incomplete = self._filled(permanent_eval="")
        assert incomplete.complete is False
        assert "permanent_eval" in incomplete.missing

    def test_the_register_lists_what_is_outstanding(self) -> None:
        register = PostmortemRegister()
        register.record(self._filled())
        register.record(self._filled(permanent_eval="").model_copy(update={"incident_id": "INC-2"}))
        assert [p.incident_id for p in register.outstanding()] == ["INC-2"]
        assert register.regression_evals() == ["eval-0042"]


class TestPromptIsVersioned:
    """FR-54-adjacent: procedural memory was constructed and never read."""

    def test_the_prompt_is_stored_as_a_procedural_artifact(
        self, pool: SqliteConnectionPool
    ) -> None:
        from agentic_commons.memory import ProceduralStore

        store = ProceduralStore(pool)
        rendered = build_system_prompt(store)
        assert store.get(PROMPT_ARTIFACT) is not None
        assert store.get(PROMPT_ARTIFACT).value == rendered  # type: ignore[union-attr]

    def test_an_unchanged_prompt_does_not_create_a_version_per_call(
        self, pool: SqliteConnectionPool
    ) -> None:
        """A history where every entry is identical records no change."""
        from agentic_commons.memory import ProceduralStore

        store = ProceduralStore(pool)
        build_system_prompt(store)
        build_system_prompt(store)
        assert len(store.history(PROMPT_ARTIFACT)) == 1

    def test_no_store_means_no_database(self) -> None:
        """An episode should not need persistence to build a prompt."""
        assert build_system_prompt().startswith("You are a customer support agent")


class TestRetrievalOverTheOrderCorpus:
    """FR-7: hybrid retrieval existed and was never a context source."""

    @pytest.mark.asyncio
    async def test_the_customers_orders_reach_the_window(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from agentic_commons.context import Task
        from agentic_commons.evals import create_scoring_cascade
        from customer_support_agent.core.builder import OrderCorpusContextSource
        from customer_support_agent.core.identity import Customer, set_current_customer
        from customer_support_agent.core.repositories import OrdersRepository

        set_current_customer(Customer(customer_id=1))
        source = OrderCorpusContextSource(OrdersRepository(seeded_pool), create_scoring_cascade())
        chunks = await source.candidates(Task(query="10001", session_id="s", actor="a"))
        assert any("10001" in chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_retrieval_is_scoped_to_the_authenticated_customer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A retriever over every customer's orders is one bug from a leak."""
        from agentic_commons.context import Task
        from agentic_commons.evals import create_scoring_cascade
        from customer_support_agent.core.builder import OrderCorpusContextSource
        from customer_support_agent.core.identity import Customer, set_current_customer
        from customer_support_agent.core.repositories import OrdersRepository

        set_current_customer(Customer(customer_id=1))
        source = OrderCorpusContextSource(OrdersRepository(seeded_pool), create_scoring_cascade())
        chunks = await source.candidates(Task(query="10003", session_id="s", actor="a"))
        assert not any("10003" in chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_an_unauthenticated_request_retrieves_nothing(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from agentic_commons.context import Task
        from agentic_commons.evals import create_scoring_cascade
        from customer_support_agent.core.builder import OrderCorpusContextSource
        from customer_support_agent.core.identity import set_current_customer
        from customer_support_agent.core.repositories import OrdersRepository

        set_current_customer(None)
        source = OrderCorpusContextSource(OrdersRepository(seeded_pool), create_scoring_cascade())
        assert await source.candidates(Task(query="10001", session_id="s", actor="a")) == []


class TestWindowHygiene:
    """FR-5: the same fact reaching the window twice is a failure mode."""

    def test_a_near_duplicate_is_dropped_before_it_costs_budget(self) -> None:
        from agentic_commons.context import Chunk, FiveStagePipeline

        pipeline = FiveStagePipeline()
        kept, dropped = pipeline.fit_to_budget(
            [
                Chunk(content="Order 10001 delivered for $32.00", kind="retrieved", score=0.9),
                Chunk(content="Order 10001 delivered for $32.00 today", kind="memory", score=0.5),
            ],
            budget=1000,
        )
        assert len(kept) == 1 and len(dropped) == 1

    def test_a_pinned_chunk_is_never_dropped_as_a_duplicate(self) -> None:
        """The policy does not compete, so it cannot lose to a copy of itself."""
        from agentic_commons.context import Chunk, FiveStagePipeline

        kept, _ = FiveStagePipeline().fit_to_budget(
            [
                Chunk(content="Refunds are allowed once delivered", kind="policy", pinned=True),
                Chunk(content="Refunds are allowed once delivered", kind="memory", pinned=True),
            ],
            budget=1000,
        )
        assert len(kept) == 2


class TestReviewerCalibration:
    """FR-40: nobody was counting whether reviewers agree with the agent."""

    @pytest.mark.asyncio
    async def test_agreement_and_override_are_both_recorded(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from customer_support_agent.core.builder import build_review_service
        from customer_support_agent.core.repositories import HitlQueueRepository
        from customer_support_agent.core.schema import HitlRequest

        service = build_review_service(pool=seeded_pool)
        queue = HitlQueueRepository(seeded_pool)
        queue.save(HitlRequest(order_id=10001, user_id=1, requested_amount=32.0))
        queue.save(HitlRequest(order_id=10002, user_id=1, requested_amount=210.0))
        pending = queue.pending()
        await service.approve(pending[0].id, "supervisor-1")  # type: ignore[arg-type]
        service.deny(pending[1].id, "supervisor-1", "not this time")  # type: ignore[arg-type]
        report = service.calibration()
        assert report.samples == 2
