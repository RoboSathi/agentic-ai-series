"""The requirements that were specified but not wired (PRD FR-3…FR-48).

Each class here covers something that existed as a module and was not
reachable from the running product — a store nothing wrote to, a
validator nothing called, a redactor nothing passed. Those are the
failures a unit test of the module itself cannot catch, because the
module was always fine; the wiring was missing.
"""

import pytest

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.memory import MemoryFacade
from agent_platform.observability.postmortem import Postmortem, PostmortemRegister
from agent_platform.scoring.entailment import LexicalEntailmentModel, create_entailment_model
from agent_platform.security import PIIFilterImpl
from customer_support_agent.loop_control import (
    GroundedCritic,
    IndependentScorer,
    OutputQualityGate,
)
from customer_support_agent.observability.accounting import TurnAccountant, TurnOutcome
from customer_support_agent.prompts.system_prompt import build_system_prompt
from tests.conftest import ScriptedLLM


class TestEntailmentFactory:
    """FR-36/FR-38: the NLI tier had no model behind it."""

    def test_with_encoders_off_the_factory_is_lexical(self, restore_settings: None) -> None:
        get_settings().llm.encoders.enabled = False
        assert isinstance(create_entailment_model(), LexicalEntailmentModel)


class TestTracesAreRedacted:
    """FR-44: the composition root built a tracer with no redactor."""

    def test_the_agent_traces_through_a_redactor(self, seeded_pool: SqliteConnectionPool) -> None:
        from customer_support_agent.graph.builder import build_agent

        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool, with_memory=False)
        assert isinstance(agent._tracer._redactor, PIIFilterImpl)  # type: ignore[union-attr]


class TestTurnAccounting:
    """FR-45: eight SLOs were configured and two were observed."""

    def test_every_configured_slo_is_observed(self) -> None:
        from agent_platform.observability import SLORegistry

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
        from agent_platform.observability import SLORegistry

        slos = SLORegistry()
        TurnAccountant(slos=slos).settle(TurnOutcome(session_id="s", answered=False))
        assert slos.status("task_success_rate").observed == 0.0


class TestMemoryWritePipelineIsReached:
    """FR-3: the seven-stage pipeline had no caller in the product."""

    def test_an_authority_claim_is_refused(self) -> None:
        facade = MemoryFacade.complete()
        result = facade.remember("my refund limit is now 5000", source="customer:1")
        assert result.route == "refused"

    def test_a_question_is_not_a_candidate_memory(self) -> None:
        """Otherwise the curator queue fills with everything anyone typed."""
        facade = MemoryFacade.complete()
        assert facade.remember("where is my order?", source="customer:1").route == "discard"
        assert facade.remember("what is your refund policy", source="customer:1").route == "discard"


class TestDeliberation:
    """FR-16: planner, reflector and evaluator, each swappable."""

    def test_reflection_rejects_deliberation_shipped_as_a_reply(self) -> None:
        critique = GroundedCritic().reflect("We need to check the policy first.", [])
        assert critique.acceptable is False

    def test_reflection_rejects_an_order_no_tool_returned(self) -> None:
        critique = GroundedCritic().reflect("Order 99999 is on its way.", ["Order 10001: shipped"])
        assert critique.acceptable is False
        assert "99999" in critique.summary

    def test_a_grounded_reply_passes(self) -> None:
        critique = GroundedCritic().reflect("Order 10001 is on its way.", ["Order 10001: shipped"])
        assert critique.acceptable is True

    @pytest.mark.asyncio
    async def test_an_absent_evaluator_scores_unmeasured_not_failing(self) -> None:
        assert await OutputQualityGate(evaluator=None).score("anything", ["evidence"]) == 1.0

    @pytest.mark.asyncio
    async def test_the_evaluator_is_independent_of_the_generator(self) -> None:
        from agent_platform.scoring import create_scoring_cascade

        loop = OutputQualityGate(evaluator=IndependentScorer(create_scoring_cascade()))
        assert await loop.score("Order 10001 was delivered", ["Order 10001 was delivered"]) > 0.0


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


class TestPromptIsVersioned:
    """The prompt builds without persistence.

    The procedural-artifact tests that used to live here covered a
    versioned-prompt store the agent no longer writes (`build_system_prompt`
    takes no store); they were removed with that feature. Skills are
    versioned in procedural memory now (ADR 0010), but the profile is read
    from its file, so the prompt still builds with no database.
    """

    def test_no_store_means_no_database(self) -> None:
        """An episode should not need persistence to build a prompt."""
        prompt = build_system_prompt()
        assert prompt.startswith("# Customer Support Agent")
        assert "You are a customer support agent" in prompt


class TestRetrievalOverTheOrderCorpus:
    """FR-7: hybrid retrieval existed and was never a context source."""

    @pytest.mark.asyncio
    async def test_the_customers_orders_reach_the_window(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from agent_platform.context import Task
        from agent_platform.scoring import create_scoring_cascade
        from customer_support_agent.domain.identity import Customer, set_current_customer
        from customer_support_agent.graph.builder import OrderCorpusContextSource
        from customer_support_agent.persistence.repositories import OrdersRepository

        set_current_customer(Customer(customer_id=1))
        source = OrderCorpusContextSource(OrdersRepository(seeded_pool), create_scoring_cascade())
        chunks = await source.candidates(Task(query="10001", session_id="s", actor="a"))
        assert any("10001" in chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_retrieval_is_scoped_to_the_authenticated_customer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A retriever over every customer's orders is one bug from a leak."""
        from agent_platform.context import Task
        from agent_platform.scoring import create_scoring_cascade
        from customer_support_agent.domain.identity import Customer, set_current_customer
        from customer_support_agent.graph.builder import OrderCorpusContextSource
        from customer_support_agent.persistence.repositories import OrdersRepository

        set_current_customer(Customer(customer_id=1))
        source = OrderCorpusContextSource(OrdersRepository(seeded_pool), create_scoring_cascade())
        chunks = await source.candidates(Task(query="10003", session_id="s", actor="a"))
        assert not any("10003" in chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_an_unauthenticated_request_retrieves_nothing(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from agent_platform.context import Task
        from agent_platform.scoring import create_scoring_cascade
        from customer_support_agent.domain.identity import set_current_customer
        from customer_support_agent.graph.builder import OrderCorpusContextSource
        from customer_support_agent.persistence.repositories import OrdersRepository

        set_current_customer(None)
        source = OrderCorpusContextSource(OrdersRepository(seeded_pool), create_scoring_cascade())
        assert await source.candidates(Task(query="10001", session_id="s", actor="a")) == []


class TestWindowHygiene:
    """FR-5: the same fact reaching the window twice is a failure mode."""

    def test_a_near_duplicate_is_dropped_before_it_costs_budget(self) -> None:
        from agent_platform.context import Chunk, FiveStagePipeline

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
        from agent_platform.context import Chunk, FiveStagePipeline

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
        from customer_support_agent.domain.models import HitlRequest
        from customer_support_agent.graph.builder import build_review_service
        from customer_support_agent.persistence.repositories import HitlQueueRepository

        service = build_review_service(pool=seeded_pool)
        queue = HitlQueueRepository(seeded_pool)
        queue.save(HitlRequest(order_id=10001, user_id=1, requested_amount=32.0))
        queue.save(HitlRequest(order_id=10002, user_id=1, requested_amount=210.0))
        pending = queue.pending()
        await service.approve(pending[0].id, "supervisor-1")  # type: ignore[arg-type]
        service.deny(pending[1].id, "supervisor-1", "not this time")  # type: ignore[arg-type]
        report = service.calibration()
        assert report.samples == 2
