"""Personal data is a token from where it enters to where it leaves (ADR P-40).

Under test: the known values (a name a model would miss is found by exact match), the
source's tokenise / restore round trip (no vault needed), that tokenising twice changes
nothing, the Executor's two crossings (arguments restored going in, results tokenised coming
out), the pass every writer makes (log records and status events), and `/harness pii off`
turning all of it into the identity. No model is loaded: the detectors here are rules.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from agent_platform.config import FeatureSet, active_features, get_settings, set_active_features
from agent_platform.core.primitives.pii_tokens import framed_token
from agent_platform.core.primitives.scrubbing import install_scrubber, scrub
from agent_platform.core.primitives.string_tree import map_strings
from agent_platform.deterministic_checks import (
    AdmittedChecker,
    DeterministicCheckRequest,
    DeterministicCheckResult,
)
from agent_platform.gatekeeper.types import AuthorityContext
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.types import StatusEvent
from agent_platform.runtime.executor import ApprovedToolCall, ToolExecutorImpl
from agent_platform.security.pii import active, detector
from agent_platform.security.pii.active import (
    install_pii_scrubbing,
    restore_if_on,
    scrub_if_on,
    tokenise_if_on,
)
from agent_platform.security.pii.detector import PatternPIIDetector, PIIFilterImpl
from agent_platform.security.pii.known_values import KnownValueDetector, KnownValues, register_known
from agent_platform.security.pii.source import PiiSource
from agent_platform.tools.types import ToolCall

# A name no NER model has context for, and no other test writes.
NAME = "Zorbanthius"
EMAIL = "zorbanthius@example.com"


def _posture(pii: bool) -> FeatureSet:
    return active_features(get_settings().features).with_override("pii", pii)


@pytest.fixture
def pii_on() -> Iterator[None]:
    set_active_features(_posture(True))
    register_known("PERSON", NAME)
    yield
    set_active_features(None)


@pytest.fixture
def pii_off() -> Iterator[None]:
    set_active_features(_posture(False))
    yield
    set_active_features(None)


def _source() -> PiiSource:
    values = KnownValues(min_chars=3, max_entries=10)
    values.register("PERSON", NAME)
    rules = PIIFilterImpl(detectors=[KnownValueDetector(values), PatternPIIDetector()], vault=None)
    return PiiSource(rules, rules)


class TestKnownValues:
    def test_find_a_lone_name_a_model_has_no_context_for(self) -> None:
        values = KnownValues(min_chars=3, max_entries=10)
        values.register("PERSON", NAME)
        [span] = KnownValueDetector(values).detect(f"verified_customer: {NAME} (account #19)")
        assert (span.category, span.text) == ("PERSON", NAME)

    def test_match_whole_words_only_and_ignore_case(self) -> None:
        values = KnownValues(min_chars=3, max_entries=10)
        values.register("PERSON", "Ada")
        assert [s.text for s in values.find("ADA said Adams is not Ada.")] == ["ADA", "Ada"]

    def test_a_value_too_short_to_tell_from_a_word_is_not_kept(self) -> None:
        values = KnownValues(min_chars=3, max_entries=10)
        values.register("PERSON", "Al")
        assert values.find("Al is here") == []


class TestTheSource:
    def test_tokenises_and_restores_without_a_vault(self) -> None:
        source = _source()
        text = f"Thanks, {NAME} — we wrote to {EMAIL}."
        tokenised = source.tokenise(text)
        assert NAME not in tokenised and EMAIL not in tokenised
        assert len(framed_token().findall(tokenised)) == 2
        assert source.restore(tokenised) == text

    def test_tokenising_twice_changes_nothing(self) -> None:
        source = _source()
        once = source.tokenise(f"{NAME} at {EMAIL}")
        assert source.tokenise(once) == once

    def test_a_token_it_never_made_stays_a_token(self) -> None:
        stranger = "[PERSON:0123456789ab]"
        assert _source().restore(f"hello {stranger}") == f"hello {stranger}"

    def test_admission_reports_what_it_replaced_never_the_value(self) -> None:
        admitted = _source().admit(f"I am {NAME}")
        assert admitted.categories == ["PERSON"]
        assert NAME not in admitted.model_dump_json()

    def test_payloads_keep_their_shape(self) -> None:
        source = _source()
        tree = source.tokenise_tree({"who": NAME, "id": 7, "notes": [EMAIL]})
        assert tree["id"] == 7
        assert source.restore_tree(tree) == {"who": NAME, "id": 7, "notes": [EMAIL]}


class _Echo:
    """A tool that records what it was handed and hands the arguments back."""

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    async def execute(self, call: ToolCall) -> dict[str, Any]:
        self.seen = dict(call.params)
        return {"customer": call.params["name"]}


@pytest.mark.usefixtures("pii_on")
class TestTheExecutor:
    async def test_a_tool_gets_values_and_returns_tokens(self) -> None:
        token = tokenise_if_on(NAME)
        echo = _Echo()
        call = ToolCall(action="crm.lookup", params={"name": token})
        outcome = await ToolExecutorImpl(echo).execute(
            ApprovedToolCall.ungoverned(call, AuthorityContext(actor="test"))
        )
        assert echo.seen == {"name": NAME}
        assert NAME not in outcome.result.output
        assert outcome.result.data == {"customer": token}


@pytest.mark.usefixtures("pii_on")
class TestEveryWriterScrubs:
    def test_a_log_record_and_its_fields(self) -> None:
        install_pii_scrubbing()
        scrubbed = scrub({"message": f"greeted {NAME}", "to": EMAIL})
        assert NAME not in str(scrubbed) and EMAIL not in str(scrubbed)

    def test_a_status_event(self) -> None:
        install_pii_scrubbing()
        feed = InMemoryLiveStatusFeed(max_events=0)
        shown: list[StatusEvent] = []
        feed.subscribe(shown.append)
        StageReporter(feed, "t").emit(
            "working_memory_write", "completed", summary=f"SET · {NAME}", fields={"n": 1}
        )
        [event] = shown
        assert NAME not in (event.summary or "")
        assert event.fields["n"] == 1

    def test_a_pass_that_writes_does_not_recurse(self) -> None:
        def writes(value: Any) -> Any:
            return f"outer({scrub(value)})"

        install_scrubber(writes)
        try:
            assert scrub("x") == "outer(x)"
        finally:
            install_pii_scrubbing()


@pytest.mark.usefixtures("pii_off")
class TestHarnessOff:
    def test_every_hook_is_the_identity(self) -> None:
        register_known("PERSON", NAME)
        text = f"{NAME} at {EMAIL}"
        assert tokenise_if_on(text) == text
        assert scrub_if_on(text) == text
        assert restore_if_on(text) == text


class TestLimits:
    def test_the_oldest_known_value_goes_first(self) -> None:
        values = KnownValues(min_chars=3, max_entries=1)
        values.register("PERSON", "Alpha")
        values.register("PERSON", "Bravo")
        assert [s.text for s in values.find("Alpha Bravo")] == ["Bravo"]

    def test_re_registering_a_value_changes_nothing(self) -> None:
        values = KnownValues(min_chars=3, max_entries=5)
        values.register("PERSON", NAME)
        version = values.version
        values.register("PERSON", NAME)
        assert values.version == version

    def test_the_oldest_token_is_forgotten_and_stays_a_token(self) -> None:
        values = KnownValues(min_chars=3, max_entries=10)
        values.register("PERSON", NAME)
        values.register("PERSON", "Quillaby")
        rules = PIIFilterImpl(detectors=[KnownValueDetector(values)], vault=None)
        source = PiiSource(rules, rules, max_tokens=1, cache_entries=1)
        first = source.tokenise(NAME)
        second = source.tokenise("Quillaby")
        assert source.restore(first) == first
        assert source.restore(second) == "Quillaby"

    def test_an_empty_message_is_admitted_as_it_is(self) -> None:
        assert _source().admit("").text == ""

    def test_the_source_follows_a_filter_it_is_given_as_a_function(self) -> None:
        rules = _source()._rules
        source = PiiSource(lambda: rules, rules)
        assert NAME not in source.tokenise(f"hi {NAME}")


class TestThePatternEngine:
    def test_known_values_still_lead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        pii = settings.security.pii.model_copy(update={"engine": "pattern"})
        patched = settings.model_copy(
            update={"security": settings.security.model_copy(update={"pii": pii})}
        )
        monkeypatch.setattr(detector, "get_settings", lambda: patched)
        kinds = [type(d) for d in detector._default_detectors()]
        assert kinds == [KnownValueDetector, PatternPIIDetector]


class _Rules:
    """Another agent's checker: its facts carry a customer's name."""

    marker = "mine"

    def check(self, request: DeterministicCheckRequest) -> DeterministicCheckResult:
        return DeterministicCheckResult(passed=True, facts={"customer": NAME, "total": 32})


@pytest.mark.usefixtures("pii_on")
class TestAdmittedChecker:
    def test_facts_enter_as_tokens(self) -> None:
        result = AdmittedChecker(_Rules()).check(
            DeterministicCheckRequest(subject_id="1", query="refund")
        )
        assert NAME not in str(result.facts) and result.facts["total"] == 32

    def test_the_wrapped_checker_is_still_reachable(self) -> None:
        checker = AdmittedChecker(_Rules())
        assert checker.marker == "mine" and isinstance(checker.inner, _Rules)


class TestPayloadsOfAnyShape:
    def test_dataclasses_and_sets_are_walked(self) -> None:
        @dataclass
        class Note:
            who: str

        walked = map_strings({"note": Note(NAME), "tags": {NAME}}, str.upper)
        assert walked == {"note": {"who": NAME.upper()}, "tags": [NAME.upper()]}


class TestTheFilterFollowsTheConfiguration:
    """A process that loads another agent's configuration detects with that agent's entities,
    while the one token map carries on (a value tokenised before is still restored)."""

    def test_new_entities_new_filter_same_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        before = active._filter()
        source = active._source()
        presidio = settings.security.pii.presidio.model_copy(update={"entities": ["US_SSN"]})
        pii = settings.security.pii.model_copy(update={"presidio": presidio})
        patched = settings.model_copy(
            update={"security": settings.security.model_copy(update={"pii": pii})}
        )
        monkeypatch.setattr(active, "get_settings", lambda: patched)
        assert active._filter() is not before
        assert active._source() is source
