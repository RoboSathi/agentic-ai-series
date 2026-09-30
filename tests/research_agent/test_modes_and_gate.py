"""Q&A vs Research: the gate, the reframe menu, /qa and /research, time windows, set-up checks."""

import pytest

from agentic_commons.config import get_settings
from research_agent.config import get_agent_settings
from research_agent.console.setup import missing_model_key, search_warning
from research_agent.domain.budget import Budget
from research_agent.domain.scope import CoverageItem
from research_agent.scoping import checklist
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.gate import resolve_reframe_choice
from research_agent.scoping.window import TimeWindowParser
from tests.research_agent.conftest import BuildHarness

ORRIN_QUESTION = "What is the cycle life of the Orrin FerroCell?"


def _parser() -> TimeWindowParser:
    return TimeWindowParser(get_agent_settings().scoping.time_window)


class TestTimeWindows:
    @pytest.mark.parametrize(
        ("text", "days"),
        [
            ("Important AI news in the last 1 month", 30),
            ("what happened in the past 3 weeks", 21),
            ("news from last month", 30),
            ("this week in robotics", 7),
            ("today's headlines", 1),
            ("news from the past 24 hours", 1),
            ("Who are the players in the market today", None),
            ("365", 365),
            ("2 years", 730),
            ("how do batteries work", None),
        ],
    )
    def test_phrases_become_day_counts(self, ra_config: None, text: str, days: int | None) -> None:
        assert _parser().days(text) == days

    def test_the_brief_reads_a_month_as_thirty_days(self, ra_config: None) -> None:
        item = CoverageItem(
            item_id="recency",
            prompt="p",
            brief_field="recency_days",
            status="covered",
            value="last 1 month",
        )
        budget = get_agent_settings().budget
        brief = BriefBuilder(get_agent_settings().scoping, budget, budget).build(
            "general", "t", [item], []
        )
        assert brief.recency_days == 30

    def test_a_stated_window_is_not_asked_again(self) -> None:
        items = [CoverageItem(item_id="recency", prompt="p", brief_field="recency_days")]
        assert checklist.prefill_window(items, 30)[0].status == "covered"
        assert checklist.prefill_window(items, None)[0].status == "open"


class TestReframeMenu:
    @pytest.mark.parametrize(
        ("reply", "mode"),
        [
            ("1", "qa"),
            ("2", "research"),
            ("news", "qa"),
            ("mapping a field", "research"),
            ("Q&A please", "qa"),
            ("banana", None),
            ("7", None),
        ],
    )
    def test_replies_resolve_deterministically(
        self, ra_config: None, reply: str, mode: str | None
    ) -> None:
        assert resolve_reframe_choice(reply, get_agent_settings().scoping.reframe_options) == mode

    async def test_the_menu_shows_both_modes_with_examples(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle("tell me about batteries", "s")
        for option in get_agent_settings().scoping.reframe_options:
            assert option.example in reply.response
        again = await h.agent.handle("2", "s")
        scope = h.agent.session("s").scope
        assert again.phase == "SCOPING" and scope is not None and scope.archetype == "general"
        assert scope.topic.startswith("tell me about batteries")


class TestRouting:
    async def test_a_direct_question_is_answered_in_one_turn(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle(ORRIN_QUESTION, "s")
        assert reply.phase == "DELIVERED" and reply.report is not None
        assert reply.report.status == "COMPLETE"
        assert [s.heading for s in reply.report.sections][0] == "Answer"
        assert get_agent_settings().messages.qa_followup in reply.response
        contract = h.agent.session("s").run_values["contract"]
        assert contract.brief.archetype == "qa" and contract.verify()

    async def test_a_decision_goes_to_research(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle("Help me choose a grid storage option for my microgrid", "s")
        scope = h.agent.session("s").scope
        assert reply.phase == "SCOPING" and scope is not None
        assert scope.mode == "research" and scope.archetype == "decision_support"

    async def test_a_research_cue_without_a_kind_is_general(self, harness: BuildHarness) -> None:
        h = harness()
        await h.agent.handle("Research the cycle life of iron-air cells", "s")
        scope = h.agent.session("s").scope
        assert scope is not None and scope.archetype == "general"

    async def test_a_window_in_the_question_is_carried(self, harness: BuildHarness) -> None:
        h = harness()
        await h.agent.handle("Help me choose a storage option from the last 2 years", "s")
        scope = h.agent.session("s").scope
        assert scope is not None and scope.recent_days == 730
        recency = checklist.item(scope.items, "recency")
        assert recency is not None and recency.status == "covered"

    async def test_slash_research_goes_deeper_on_the_last_question(
        self, harness: BuildHarness
    ) -> None:
        h = harness()
        await h.agent.handle(ORRIN_QUESTION, "s")
        reply = await h.agent.handle("/research", "s")
        scope = h.agent.session("s").scope
        assert reply.phase == "SCOPING" and scope is not None
        assert scope.archetype == "general" and scope.topic.startswith("What is the cycle life")
        done = await h.agent.handle("/done", "s")
        assert done.phase == "RUBRIC_REVIEW"
        final = await h.agent.handle("/approve", "s")
        assert final.report is not None and final.report.sections[0].heading == "Findings"

    async def test_slash_qa_forces_a_quick_answer(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle("/qa Explain the round-trip efficiency of iron-air cells", "s")
        assert reply.phase == "DELIVERED"
        assert h.agent.session("s").run_values["contract"].brief.archetype == "qa"

    async def test_a_bare_command_with_nothing_to_research(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle("/research", "s")
        assert reply.response == get_agent_settings().messages.new_scope


class TestLiveSetup:
    def test_the_demo_needs_no_key(self, ra_config: None) -> None:
        assert missing_model_key(get_settings(), get_agent_settings()) is None

    def test_routed_models_need_the_nvidia_key(
        self, ra_config: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("nvidia_api_key", raising=False)
        routed = get_agent_settings().model_copy(update={"model_backend": "routed"})
        message = missing_model_key(get_settings(), routed)
        assert message is not None and "nvidia_api_key" in message and "synthesizer" in message

    def test_live_search_without_tavily_warns(
        self, ra_config: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("tavily_api_key", raising=False)
        live = get_settings().model_copy(
            update={"web": get_settings().web.model_copy(update={"mode": "live"})}
        )
        assert (
            search_warning(live, get_agent_settings()) == get_agent_settings().setup.no_search_key
        )
        assert search_warning(get_settings(), get_agent_settings()) is None

    def test_prepare_stops_with_a_sentence(
        self, ra_config: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.config import reload_agent_settings
        from research_agent.console import ResearchHost

        monkeypatch.delenv("nvidia_api_key", raising=False)
        monkeypatch.setenv("RESEARCH_AGENT__MODEL_BACKEND", "routed")
        reload_agent_settings()
        with pytest.raises(SystemExit, match="nvidia_api_key"):
            ResearchHost().prepare()


def test_budgets_stay_inside_the_free_tiers(ra_config: None) -> None:
    """The shipped live defaults, not the demo profile the tests run on."""
    import yaml

    from agentic_commons.config import REPO_ROOT

    raw = yaml.safe_load((REPO_ROOT / "src/research_agent/config/config.yaml").read_text())
    agent = raw["research_agent"]
    assert Budget.model_validate(agent["budget"]).max_searches <= 8
    assert Budget.model_validate(agent["qa"]["budget"]).max_searches <= 3


class TestGreeting:
    def test_the_demo_lists_exactly_the_worlds_topics(self, ra_config: None) -> None:
        from research_agent.console import ResearchHost
        from research_agent.fixture_world.world import load_world

        text = ResearchHost().greeting()
        world = load_world(get_agent_settings().resolved_fixture_world_path)
        assert all(topic.example in text for topic in world.topics)
        assert get_agent_settings().greeting.wont in text

    def test_live_names_its_providers_and_claims_no_demo_topics(self, ra_config: None) -> None:
        from research_agent.console.greeting import build_greeting

        live = get_settings().model_copy(
            update={"web": get_settings().web.model_copy(update={"mode": "live"})}
        )
        agent = get_agent_settings().model_copy(update={"model_backend": "routed"})
        text = build_greeting(live, agent, ["quota note"])
        assert agent.greeting.live_mode in text and "Kelvari" not in text
        assert "quota note" in text


class TestNewsSearch:
    async def test_a_short_window_searches_news_within_it(
        self, harness: BuildHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic_commons.web import SearchRequest, WebGateway

        seen: list[SearchRequest] = []
        original = WebGateway.search

        async def spy(self: WebGateway, request: SearchRequest) -> object:
            seen.append(request)
            return await original(self, request)

        monkeypatch.setattr(WebGateway, "search", spy)
        h = harness()
        reply = await h.agent.handle("Important Kelvari AI news in the last 1 month", "s")
        assert reply.report is not None and reply.report.status == "COMPLETE"
        assert seen and all(r.recent_days == 30 and r.topic == "news" for r in seen)
        cited = {c.url for c in reply.report.citations}
        assert "https://kelvari-startups.example/ai-funding-q2" not in cited

    async def test_an_exhausted_search_quota_is_an_honest_partial(
        self, harness: BuildHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic_commons.resilience import QuotaExhaustedError
        from agentic_commons.web import SearchRequest, WebGateway, WebUnavailableError

        async def spent(self: WebGateway, request: SearchRequest) -> object:
            raise WebUnavailableError("search failed") from QuotaExhaustedError("spent")

        monkeypatch.setattr(WebGateway, "search", spent)
        h = harness()
        reply = await h.agent.handle(ORRIN_QUESTION, "s")
        stop = get_agent_settings().research.quota_stop
        assert reply.report is not None and reply.report.status == "PARTIAL"
        assert stop in h.agent.session("s").run_values["trace"].budget_stops
