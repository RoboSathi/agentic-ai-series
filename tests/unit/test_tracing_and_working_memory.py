"""Tracing, the live trace stream, and the active window (LLD §2, §8, §20).

These three are tested together because they are the same promise from
three angles: whatever the harness does should be visible afterwards —
in the file log, on the screen, and in what the next turn knows.
"""

import logging
from typing import Any

import pytest
from rich.text import Text

from agentic_commons.logging import LoggerFactory, get_logger, instrument
from agentic_commons.logging.decorators import logged
from agentic_commons.memory import MemoryFacade, MemoryQuery, Provenance
from agentic_commons.memory.types import MemoryFact
from agentic_commons.memory.working.memory import WorkingMemory
from agentic_commons.observability import (
    InMemoryLiveStatusFeed,
    StageReporter,
    traced,
)
from agentic_commons.tui.log_feed import LiveFeedLogHandler, TraceStream


def _fact(key: str, value: str, session: str = "") -> MemoryFact:
    return MemoryFact(
        key=key,
        value=value,
        metadata={"session_id": session} if session else {},
        provenance=Provenance(source="test"),
    )


class TestTracedDecorator:
    """`@traced` reports a stage to the feed and the log, both or neither."""

    class Component:
        def __init__(self, reporter: StageReporter | None = None) -> None:
            self._reporter = reporter
            self._logger = get_logger("observability")

        @traced("unit_stage")
        def work(self, value: int) -> str:
            return f"done {value}"

        @traced("async_stage")
        async def async_work(self) -> str:
            return "async done"

        @traced("failing_stage")
        def explode(self) -> None:
            raise RuntimeError("boom")

    def test_a_sync_method_reports_started_then_completed(self) -> None:
        feed = InMemoryLiveStatusFeed()
        assert self.Component(StageReporter(feed, "run")).work(1) == "done 1"
        assert [(e.stage, e.status) for e in feed.events] == [
            ("unit_stage", "started"),
            ("unit_stage", "completed"),
        ]

    @pytest.mark.asyncio
    async def test_an_async_method_is_awaited_not_merely_scheduled(self) -> None:
        """A sync wrapper would trace the coroutine object, not the result."""
        feed = InMemoryLiveStatusFeed()
        assert await self.Component(StageReporter(feed, "run")).async_work() == "async done"
        assert feed.events[-1].status == "completed"

    def test_a_failure_is_reported_and_re_raised(self) -> None:
        """Tracing that swallows the exception hides the event worth tracing."""
        feed = InMemoryLiveStatusFeed()
        with pytest.raises(RuntimeError):
            self.Component(StageReporter(feed, "run")).explode()
        assert feed.events[-1].status == "failed"
        assert feed.events[-1].detail == "boom"

    def test_a_component_with_no_reporter_still_runs(self) -> None:
        """Nothing watching is the normal case outside the console."""
        assert self.Component().work(2) == "done 2"


class TestInstrument:
    """One annotation on the class, so a new method cannot be forgotten."""

    def test_every_public_method_is_traced(self, caplog: pytest.LogCaptureFixture) -> None:
        @instrument("observability")
        class Service:
            def act(self) -> str:
                return "acted"

            def _hidden(self) -> str:
                return "hidden"

        with caplog.at_level(logging.DEBUG, logger="agentic.observability"):
            Service().act()
            Service()._hidden()
        traced_calls = [r.getMessage() for r in caplog.records if "CALLED" in r.getMessage()]
        assert any("Service.act" in message for message in traced_calls)
        assert not any("_hidden" in message for message in traced_calls)

    def test_an_already_logged_method_is_not_traced_twice(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Double-tracing makes the file log look like a loop."""

        @instrument("observability")
        class Service:
            @logged("observability")
            def act(self) -> str:
                return "acted"

        with caplog.at_level(logging.DEBUG, logger="agentic.observability"):
            Service().act()
        calls = [r for r in caplog.records if "CALLED" in r.getMessage()]
        assert len(calls) == 1

    def test_payloads_are_not_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        """The argument you need to read back is always the long one."""
        long_value = "x" * 500

        @instrument("observability")
        class Service:
            def act(self, value: str) -> str:
                return value

        with caplog.at_level(logging.DEBUG, logger="agentic.observability"):
            Service().act(long_value)
        assert any(long_value in record.getMessage() for record in caplog.records)


class TestTraceStream:
    """Every log record, mirrored to the panel while it is switched on."""

    def test_records_reach_the_sink_until_it_is_disabled(self) -> None:
        lines: list[Text] = []
        stream = TraceStream(lines.append)
        stream.enable()
        get_logger("memory").debug("something happened", key="value")
        assert stream.enabled
        assert any("something happened" in line.plain for line in lines)

        stream.disable()
        get_logger("memory").debug("after the switch")
        assert not stream.enabled
        assert not any("after the switch" in line.plain for line in lines)

    def test_a_subsystem_whose_logger_is_built_later_still_reaches_the_screen(self) -> None:
        """The subscription outlives the loggers that existed when it started."""
        lines: list[Text] = []
        stream = TraceStream(lines.append)
        stream.enable()
        try:
            LoggerFactory._loggers.pop("evals", None)
            get_logger("evals").info("late subsystem")
            assert any("late subsystem" in line.plain for line in lines)
        finally:
            stream.disable()

    def test_the_line_names_its_subsystem(self) -> None:
        record = logging.LogRecord(
            "agentic.gatekeeper", logging.INFO, __file__, 1, "refused", None, None
        )
        rendered = LiveFeedLogHandler.render(record).plain
        assert rendered.strip().startswith("gatekeeper")
        assert "refused" in rendered

    def test_a_broken_sink_never_breaks_the_caller(self) -> None:
        """A UI that can break logging is worse than no UI."""

        def explode(_: Text) -> None:
            raise RuntimeError("panel gone")

        handler = LiveFeedLogHandler(explode)
        handler.handleError = lambda record: None  # type: ignore[method-assign]
        handler.emit(logging.LogRecord("agentic.tui", logging.INFO, __file__, 1, "hi", None, None))


class TestWorkingMemory:
    """The active window: per session, read back, and dropped on demand."""

    def test_one_session_cannot_see_another_session_window(self) -> None:
        """Two conversations in one process is the normal case, not an edge."""
        window = WorkingMemory(capacity=5)
        window.write(_fact("order", "order 1", session="a"), Provenance(source="t"))
        window.write(_fact("order", "order 2", session="b"), Provenance(source="t"))
        held_by_a = window.read(MemoryQuery(session_id="a"))
        assert [fact.value for fact in held_by_a] == ["order 1"]

    def test_the_newest_fact_comes_back_first(self) -> None:
        window = WorkingMemory(capacity=5)
        window.write(_fact("a", "first", session="s"), Provenance(source="t"))
        window.write(_fact("b", "second", session="s"), Provenance(source="t"))
        assert window.read(MemoryQuery(session_id="s", limit=1))[0].value == "second"

    def test_eviction_is_per_session(self) -> None:
        window = WorkingMemory(capacity=2)
        for index in range(4):
            window.write(_fact(str(index), str(index), session="s"), Provenance(source="t"))
        window.write(_fact("other", "other", session="t"), Provenance(source="t"))
        assert window.size("s") == 2
        assert window.size("t") == 1

    def test_ending_one_session_leaves_the_others(self) -> None:
        window = WorkingMemory(capacity=5)
        window.write(_fact("a", "1", session="keep"), Provenance(source="t"))
        window.write(_fact("b", "2", session="drop"), Provenance(source="t"))
        window.clear("drop")
        assert window.sessions == ["keep"]


class TestMemoryFacadeWindow:
    def test_note_lands_in_the_window_and_reads_back(self) -> None:
        facade = MemoryFacade()
        facade.note("verified_customer", "Ada (#1)", "session-1", "agent", category="identity")
        window = facade.active_window("session-1")
        assert [fact.value for fact in window] == ["Ada (#1)"]

    def test_a_noted_fact_is_treated_as_the_system_of_record(self) -> None:
        """Tool results and database reads are verified, not self-reported."""
        facade = MemoryFacade()
        facade.note("order", "order 10001 delivered", "session-2", "agent")
        assert facade.active_window("session-2")[0].provenance.kind == "verified_system_of_record"

    def test_ending_a_session_clears_its_window_only(self) -> None:
        facade = MemoryFacade()
        facade.note("a", "1", "one", "agent")
        facade.note("b", "2", "two", "agent")
        facade.end_session("one")
        assert facade.active_window("one") == []
        assert len(facade.active_window("two")) == 1


class TestWorkingMemoryInTheAgent:
    """The window is written by the graph and read back into the context."""

    @pytest.mark.asyncio
    async def test_a_turn_establishes_facts_the_next_turn_can_see(
        self, seeded_pool: Any, scripted_llm: Any
    ) -> None:
        from customer_support_agent.core.builder import build_agent

        memory = MemoryFacade()
        agent = build_agent(llm=scripted_llm, pool=seeded_pool, memory=memory, observe=False)
        await agent.handle("where is order 10001", session_id="s1")
        held = {fact.key for fact in memory.active_window("s1")}
        assert "current_intent" in held

    @pytest.mark.asyncio
    async def test_the_window_reaches_the_assembled_context(
        self, seeded_pool: Any, scripted_llm: Any
    ) -> None:
        """A store nothing reads is not memory — this is that regression."""
        from agentic_commons.context import Task
        from customer_support_agent.core.builder import WorkingMemoryContextSource

        memory = MemoryFacade()
        memory.note("verified_customer", "Ada (#1)", "s2", "agent", category="identity")
        chunks = await WorkingMemoryContextSource(memory).candidates(
            Task(query="anything", session_id="s2", actor="agent")
        )
        assert [chunk.content for chunk in chunks] == ["verified_customer: Ada (#1)"]
        assert all(chunk.pinned for chunk in chunks)
