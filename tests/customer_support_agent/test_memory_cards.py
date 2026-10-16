"""What memory bookkeeping says: quiet, and only what a reader can use.

A working-memory write printed `operation`, `written`, `category`, `key`
and `memory_value` on a card of its own; a recall printed six scoring
numbers per fact; a per-turn event printed a constant memory label and who
recorded it. None of it drew a decision, so it is logged and counted in the
turn summary instead.
"""

from types import SimpleNamespace
from typing import Any

from agent_platform.memory.dream import TurnRecorder
from agent_platform.memory.facade import _event_fields, _recall_fields
from agent_platform.memory.types import MemoryFact, Provenance, RecallReport, ScoredFact
from agent_platform.memory.working import WorkingMemoryStore
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.security.pii.known_values import register_known
from customer_support_agent.graph.agent import CustomerSupportAgent
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.memory.turn_memory import FinishedTurn, TurnMemory


class _Support(NodeSupport):
    def __init__(self) -> None:
        self.done: list[dict[str, Any]] = []
        self._working_memory = WorkingMemoryStore()
        self._security: Any = SimpleNamespace(redact=lambda text: text)

    def _done(self, stage: str, *args: Any, **kwargs: Any) -> None:
        self.done.append({"stage": stage, **kwargs})


class TestAWorkingMemoryWriteIsBookkeeping:
    def test_it_is_quiet_and_says_only_what_was_set(self) -> None:
        support = _Support()
        support._note(AgentState(session_id="s"), "current_intent", "REQUEST_REFUND", "intent")
        [event] = support.done
        assert event["quiet"] is True
        assert event["fields"] == {"key": "current_intent", "memory_value": "REQUEST_REFUND"}


def _event(text: str, key: str) -> MemoryFact:
    return MemoryFact(
        key=key, value=text, memory_type="episodic", provenance=Provenance(source="agent")
    )


class TestARecallSaysWhatWasFoundNotHowItWasScored:
    def _fields(self) -> dict[str, Any]:
        report = RecallReport(answered_by="lexical", embed_calls=0)
        ranked = [ScoredFact(fact=_event("query: refund", "event_x_1"), score=0.6, tier="lexical")]
        return _recall_fields(
            SimpleNamespace(text="refund", key="", subject_id="89"), report, ranked
        )

    def test_the_recalled_text_carries_no_event_id(self) -> None:
        [row] = self._fields()["recalled"]
        assert row.startswith("query: refund") and "event_x_1" not in row

    def test_nothing_that_is_constant_is_reported(self) -> None:
        fields = self._fields()
        assert "for" not in fields and not fields.get("skipped")


class TestAnEpisodicWriteNamesTheEventOnce:
    def test_the_text_and_its_id_and_nothing_constant(self) -> None:
        fields = _event_fields(_event("query: now refund 10975", "event_tui_9"))
        assert fields == {"event": "query: now refund 10975", "event_id": "event_tui_9"}


class TestTheDreamInboxIsNamed:
    def test_a_completed_turn_reports_the_event_it_queued(self) -> None:
        feed = InMemoryLiveStatusFeed()
        reporter = StageReporter(feed, "run")
        reporter.begin_turn()
        queued: list[Any] = []
        memory = TurnMemory(
            WorkingMemoryStore(),
            None,
            TurnRecorder(SimpleNamespace(emit=queued.append), "csa"),
            lambda text: text,
            reporter,
        )
        memory.complete(
            FinishedTurn(
                session_id="s", customer_id=89, spoken="show orders", response="", remembered=""
            )
        )
        reporter.end_turn()
        summary = next(e for e in feed.events if e.stage == "turn_summary")
        assert len(queued) == 1
        assert summary.fields["dream_inbox"] == "1 turn(s) queued · 0 action(s), 0 observation(s)"
        assert [e.stage for e in feed.events] == ["turn_summary"]


class TestTheTurnSummaryHoldsWhatWorkingMemoryHolds:
    def _agent(self) -> CustomerSupportAgent:
        agent = object.__new__(CustomerSupportAgent)
        store = WorkingMemoryStore()
        memory = store.get("s")
        # Registered as a database read of the customer's row registers it (`known_columns`).
        register_known("PERSON", "Linus Hoare")
        memory.set("verified_customer", "Linus Hoare (account #89)")
        memory.set("order_under_discussion", "order 10975")
        memory.add_history("show orders", "Listed the orders.")
        memory.add_history("refund 10975", "Refunded.")
        agent.__dict__.update(_working_memory=store, _security=SimpleNamespace(redact=lambda t: t))
        return agent

    def test_the_facts_and_how_many_turns_it_carries(self) -> None:
        held = self._agent()._held("s")
        # The name is a token, never the value (ADR P-40): the leak this test now guards.
        assert "Linus Hoare" not in held["verified_customer"]
        assert held["verified_customer"].endswith("(account #89)")
        assert {k: v for k, v in held.items() if k != "verified_customer"} == {
            "order_under_discussion": "order 10975",
            "history": "2 turns",
        }

    def test_a_session_with_nothing_held_says_nothing(self) -> None:
        assert self._agent()._held("nobody") == {}
