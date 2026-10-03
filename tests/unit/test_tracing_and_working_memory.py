"""Tracing and the active window (LLD §2, §8, §20).

Tested together because they are the same promise from two angles:
whatever the harness does should be visible afterwards — in the file log,
and in what the next turn knows.
"""

import logging

import pytest

from agent_platform.logging import get_logger, instrument
from agent_platform.logging.decorators import logged
from agent_platform.memory import MemoryFacade, Provenance
from agent_platform.memory.types import MemoryFact
from agent_platform.observability import (
    InMemoryLiveStatusFeed,
    StageReporter,
    traced,
)


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


class TestMemoryFacadeWindow:
    def test_note_lands_in_the_window_and_reads_back(self) -> None:
        facade = MemoryFacade.complete()
        facade.note("verified_customer", "Ada (#1)", "session-1", "agent", category="identity")
        window = facade.active_window("session-1")
        assert [fact.value for fact in window] == ["Ada (#1)"]

    def test_ending_a_session_clears_its_window_only(self) -> None:
        facade = MemoryFacade.complete()
        facade.note("a", "1", "one", "agent")
        facade.note("b", "2", "two", "agent")
        facade.end_session("one")
        assert facade.active_window("one") == []
        assert len(facade.active_window("two")) == 1
