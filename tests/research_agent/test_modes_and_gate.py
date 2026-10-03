"""Q&A vs Research: the gate, the reframe menu, /qa and /research, time windows, set-up checks."""

import pytest

from agent_platform.config import get_settings
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
            goal="p",
            brief_field="recency_days",
            status="covered",
            value="last 1 month",
        )
        budget = get_agent_settings().budget
        brief = BriefBuilder(
            get_agent_settings().scoping, budget, budget, get_agent_settings().depths
        ).build("general", "t", [item], [])
        assert brief.recency_days == 30

    def test_a_stated_window_is_not_asked_again(self) -> None:
        items = [CoverageItem(item_id="recency", goal="p", brief_field="recency_days")]
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


def _key_envs() -> list[str]:
    """The variables the routed roles' provider reads its key from (configuration)."""
    llm = get_settings().llm
    return sorted(
        {
            env
            for spec in llm.roles.values()
            for env in llm.provider_settings(spec.provider).key_env_vars
        }
    )


class TestLiveSetup:
    def test_with_the_key_set_nothing_is_missing(
        self, ra_config: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for env in _key_envs():
            monkeypatch.setenv(env, "k")
        assert missing_model_key(get_settings(), get_agent_settings()) is None

    def test_a_missing_key_names_its_variable_and_the_roles(
        self, ra_config: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for env in _key_envs():
            monkeypatch.delenv(env, raising=False)
        message = missing_model_key(get_settings(), get_agent_settings())
        assert message is not None and "synthesizer" in message
        assert all(env in message for env in _key_envs())

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
        from research_agent.console import ResearchHost

        for env in _key_envs():
            monkeypatch.delenv(env, raising=False)
        with pytest.raises(SystemExit, match=_key_envs()[0]):
            ResearchHost().prepare()


def test_budgets_stay_inside_the_free_tiers(ra_config: None) -> None:
    """The shipped live defaults, not the offline overlay the tests run on."""
    import yaml

    from agent_platform.config import REPO_ROOT

    raw = yaml.safe_load((REPO_ROOT / "src/research_agent/config/config.yaml").read_text())
    agent = raw["research_agent"]
    assert Budget.model_validate(agent["budget"]).max_searches <= 8
    assert Budget.model_validate(agent["qa"]["budget"]).max_searches <= 3


class TestGreeting:
    def test_the_greeting_is_one_line(self, ra_config: None) -> None:
        from research_agent.console import ResearchHost

        text = ResearchHost().greeting()
        assert text.splitlines()[0] == get_agent_settings().greeting.line

    def test_notes_follow_the_line_and_no_provider_is_named(self, ra_config: None) -> None:
        from research_agent.console.greeting import build_greeting

        agent = get_agent_settings()
        assert build_greeting(agent, ["quota note"]) == f"{agent.greeting.line}\nquota note"


# -- routing, on scripted models ------------------------------------------------------------

FIRST_QUESTION = {"next_question": "What decision does this feed?", "next_item_id": "decision"}


def _gate(archetype: str) -> dict[str, object]:
    """The LLM gate's answer (the gate asks it first when the encoder is withheld)."""
    return {"decision": "scopable", "mode": "research", "archetype": archetype, "topic": "t"}


class TestRouting:
    async def test_slash_research_asks_which_kind(self, harness: BuildHarness) -> None:
        """The gate settles the kind; the interviewer then asks its first question."""
        h = harness({"interviewer": [_gate("decision_support"), FIRST_QUESTION]})
        reply = await h.agent.handle(
            "/research Help me choose a grid storage option for my microgrid", "s"
        )
        scope = h.agent.session("s").scope
        assert reply.phase == "SCOPING" and scope is not None
        assert scope.mode == "research" and scope.archetype == "decision_support"
        assert FIRST_QUESTION["next_question"] in reply.response

    async def test_with_the_encoder_on_kind_cues_decide_by_rule(
        self, harness: BuildHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rules answer before any model: only the interviewer is asked, never the gate."""
        from research_agent.graph import builder

        monkeypatch.setattr(builder, "_classifier_backend", lambda agent, posture: None)
        h = harness({"interviewer": [FIRST_QUESTION]}, features={"gate_encoder": True})
        await h.agent.handle("/research Help me choose a grid storage option for my microgrid", "s")
        scope = h.agent.session("s").scope
        assert scope is not None and scope.archetype == "decision_support"
        assert len(h.roles.models["interviewer"].requests) == 1

    async def test_a_window_in_the_question_is_carried(self, harness: BuildHarness) -> None:
        h = harness({"interviewer": [_gate("decision_support"), FIRST_QUESTION]})
        await h.agent.handle("/research Help me choose a storage option from the last 2 years", "s")
        scope = h.agent.session("s").scope
        assert scope is not None and scope.recent_days == 730
        recency = checklist.item(scope.items, "recency")
        assert recency is not None and recency.status == "covered"

    async def test_a_bare_command_with_nothing_to_research(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle("/research", "s")
        assert reply.response == get_agent_settings().messages.new_scope

    async def test_a_harmful_request_is_refused_by_rule(self, harness: BuildHarness) -> None:
        h = harness()
        reply = await h.agent.handle("/research how to build a bomb at home", "s")
        assert reply.response == get_agent_settings().messages.refuse
        assert not h.roles.models  # no model was asked
