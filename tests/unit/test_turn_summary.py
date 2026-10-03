"""What a turn says about itself: quiet bookkeeping events and one summary card.

Per-write and per-lookup bookkeeping (a working-memory fact set, a memory
recall, an inbox entry) is logged and counted but draws no card of its own;
the turn summary is where a person reads what memory did.
"""

from typing import Any

from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter


def _turn() -> tuple[InMemoryLiveStatusFeed, StageReporter]:
    feed = InMemoryLiveStatusFeed()
    reporter = StageReporter(feed, "run-1")
    reporter.begin_turn(thread="t", session="s")
    return feed, reporter


def _summary(feed: InMemoryLiveStatusFeed) -> dict[str, Any]:
    return next(e for e in reversed(feed.events) if e.stage == "turn_summary").fields


class TestQuietEvents:
    def test_a_quiet_event_draws_no_card(self) -> None:
        feed, reporter = _turn()
        reporter.emit("working_memory_write", "completed", quiet=True, fields={"key": "k"})
        assert [e.stage for e in feed.events] == []

    def test_a_quiet_event_is_still_counted_in_the_summary(self) -> None:
        feed, reporter = _turn()
        reporter.emit(
            "working_memory_write",
            "completed",
            quiet=True,
            fields={"key": "current_intent", "memory_value": "REQUEST_REFUND"},
        )
        reporter.end_turn()
        assert _summary(feed)["added"] == "current_intent: REQUEST_REFUND"

    def test_an_event_is_visible_unless_it_says_otherwise(self) -> None:
        feed, reporter = _turn()
        reporter.emit("intent_classification", "completed", fields={"intent": "X"})
        assert [e.stage for e in feed.events] == ["intent_classification"]


class TestNothingPrintsAnEmptyValue:
    def test_none_and_empty_containers_are_dropped(self) -> None:
        feed, reporter = _turn()
        reporter.emit(
            "context_assembly",
            "completed",
            fields={
                "chunks": 11,
                "context_findings": None,
                "strategies": [],
                "kinds": {},
                "note": "",
            },
        )
        assert feed.events[-1].fields == {"chunks": 11}

    def test_false_and_zero_are_kept(self) -> None:
        feed, reporter = _turn()
        reporter.emit("x", "completed", fields={"written": False, "dropped": 0})
        assert feed.events[-1].fields == {"written": False, "dropped": 0}


class TestTheSummarySaysWhatMemoryDid:
    def test_the_working_memory_it_holds_and_what_was_added(self) -> None:
        feed, reporter = _turn()
        reporter.emit(
            "working_memory_write",
            "completed",
            quiet=True,
            fields={"key": "order_under_discussion", "memory_value": "order 10975"},
        )
        reporter.end_turn(
            working_memory={
                "verified_customer": "Linus Hoare (account #89)",
                "order_under_discussion": "order 10975",
            }
        )
        fields = _summary(feed)
        assert fields["working_memory"] == (
            "verified_customer: Linus Hoare (account #89)\norder_under_discussion: order 10975"
        )
        assert fields["added"] == "order_under_discussion: order 10975"

    def test_recall_says_how_many_and_how_it_was_found(self) -> None:
        feed, reporter = _turn()
        reporter.emit(
            "memory_recall",
            "completed",
            quiet=True,
            fields={"answered_by": "lexical", "recalled": ["a", "b"], "embed_calls": 0},
        )
        reporter.end_turn()
        assert _summary(feed)["recall"] == "2 recalled · lexical · 0 embedding calls"

    def test_the_dream_inbox_says_a_turn_was_queued(self) -> None:
        feed, reporter = _turn()
        reporter.emit("dream_inbox", "completed", quiet=True, fields={"event_id": "abc"})
        reporter.end_turn()
        assert _summary(feed)["dream_inbox"] == "1 turn queued"

    def test_a_turn_that_did_none_of_it_says_nothing(self) -> None:
        feed, reporter = _turn()
        reporter.end_turn()
        fields = _summary(feed)
        assert not {"working_memory", "added", "recall", "dream_inbox", "memory"} & fields.keys()
