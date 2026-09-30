"""Feature toggles (withheld = never built), the injection ablations, cancel, and the console host."""

import asyncio

import pytest

from agentic_commons.commands import CommandResult
from agentic_commons.commands.types import CommandError
from agentic_commons.config import get_settings
from agentic_commons.host import CancellableHost, GreetingHost, PanelHost
from agentic_commons.llm.roles import RoleCollisionError
from research_agent.config import get_agent_settings, reload_agent_settings
from research_agent.console import ResearchHost
from tests.research_agent.conftest import BuildHarness, converse, decision_script


def _run_nodes(h: object) -> set[str]:
    return set(h.agent._runs._graph.get_graph().nodes)  # type: ignore[attr-defined]


def _scope_nodes(h: object) -> set[str]:
    return set(h.agent._scoping._graph.get_graph().nodes)  # type: ignore[attr-defined]


class TestWithheldFeaturesAreNeverBuilt:
    def test_the_goldfish_node_exists_only_with_the_feature(self, harness: BuildHarness) -> None:
        assert "goldfish_audit" in _run_nodes(harness())
        assert "goldfish_audit" not in _run_nodes(harness({"goldfish_audit": False}))

    def test_the_rubric_stages_exist_only_with_the_feature(self, harness: BuildHarness) -> None:
        assert {"rubric_generate", "rubric_edit"} <= _scope_nodes(harness())
        withheld = _scope_nodes(harness({"rubric": False}))
        assert "rubric_generate" not in withheld and "brief_ready" in withheld

    def test_defences_are_absent_when_withheld(self, harness: BuildHarness) -> None:
        h = harness(
            {
                "injection_structural": False,
                "injection_classifier": False,
                "quarantined_extractor": False,
            }
        )
        contract = _any_contract()
        resources = h.agent._resources(contract)
        assert resources.outbound is None and resources.screen is None
        assert type(resources.reader).__name__ == "UnquarantinedReader"

    def test_the_procedural_store_is_never_built(self, harness: BuildHarness) -> None:
        memory = harness().agent._memory
        assert memory is not None and memory.procedural is None and memory.episodic is not None

    async def test_rubric_withheld_runs_without_acceptance_criteria(
        self, harness: BuildHarness
    ) -> None:
        h = harness({"rubric": False, "goldfish_audit": False})
        reply, _ = await converse(h, decision_script())
        scope = h.agent.session("s1").scope
        assert (
            scope is not None and scope.contract is not None and scope.contract.rubric.items == ()
        )
        assert reply.report is not None and reply.report.status == "COMPLETE"

    async def test_without_the_challenge_guard_no_guard_event_appears(
        self, harness: BuildHarness
    ) -> None:
        h = harness({"scoping_challenge_check": False})
        await converse(h, decision_script())
        assert not h.audited("scope.challenge_guard")


class TestInjectionAblations:
    async def test_without_the_classifier_hostile_claims_reach_the_report(
        self, harness: BuildHarness
    ) -> None:
        h = harness({"injection_classifier": False})
        reply, _ = await converse(h, decision_script())
        text = reply.report.text() if reply.report else ""
        assert "99 percent confidence" in text or "recalled" in text

    async def test_without_the_quarantine_the_structural_layer_blocks_exfiltration(
        self, harness: BuildHarness
    ) -> None:
        h = harness({"injection_classifier": False, "quarantined_extractor": False})
        await converse(h, decision_script())
        trace = h.agent.session("s1").run_values["trace"]
        assert any(
            "PRIVATE_TERM" in blocked or "DENY_RULE" in blocked
            for blocked in trace.blocked_outbound
        )
        assert not any("tamsin co-op" in q.casefold() for q in trace.attempted_outbound)

    async def test_with_every_layer_off_the_private_context_leaks(
        self, harness: BuildHarness
    ) -> None:
        h = harness(
            {
                "injection_classifier": False,
                "quarantined_extractor": False,
                "injection_structural": False,
            }
        )
        await converse(h, decision_script())
        trace = h.agent.session("s1").run_values["trace"]
        assert any("tamsin co-op" in q.casefold() for q in trace.attempted_outbound)


class TestIndependentJudge:
    def test_a_judge_sharing_the_writer_model_is_refused_at_start_up(
        self, harness: BuildHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth = get_settings().llm.roles["synthesizer"]
        monkeypatch.setenv("LLM__ROLES__JUDGE__MODEL", synth.model)
        monkeypatch.setenv("LLM__ROLES__JUDGE__PROVIDER", synth.provider)
        from agentic_commons.config import reload_settings

        reload_settings()
        reload_agent_settings()
        with pytest.raises(RoleCollisionError):
            harness()


class TestCancel:
    async def test_a_cancelled_run_ends_cancelled_with_its_checkpoint(
        self, harness: BuildHarness
    ) -> None:
        h = harness()
        user_turns = decision_script()
        from research_agent.fixture_world.scripted_user import ScriptedUser

        user = ScriptedUser(user_turns, get_agent_settings().command_names, 40)
        reply = await h.agent.handle(user_turns.first_message, "s1")
        while reply.phase == "SCOPING":
            reply = await h.agent.handle(user.next_message(h.agent, "s1", reply), "s1")
        run = asyncio.create_task(h.agent.handle("/approve", "s1"))
        while h.agent._active is None and not run.done():
            await asyncio.sleep(0)
        assert h.agent.cancel() is True
        final = await run
        assert final.phase == "CANCELLED"
        record = h.agent.runs.get(h.agent.session("s1").run_thread)
        assert record is not None and record.checkpoint_id
        assert h.audited("run.cancel_requested")


class TestConsoleHost:
    def test_the_host_offers_every_hook(self, ra_config: None) -> None:
        host = ResearchHost()
        assert (
            isinstance(host, GreetingHost)
            and isinstance(host, PanelHost)
            and isinstance(host, CancellableHost)
        )
        assert host.greeting() and host.cancel_key() == get_agent_settings().cancel.key
        assert set(get_agent_settings().tui.stage_labels) <= set(host.stage_labels())
        assert host.side_panels("s") == [] and host.cancel() is False

    async def test_conversational_commands_are_forwarded_as_turns(self, ra_config: None) -> None:
        commands = {c.name: c for c in ResearchHost().commands()}
        done: CommandResult = await commands["done"].run([])
        assert done.effect.agent_turn == "/done"
        edit = await commands["rubric"].run(["add", "lead", "times"])
        assert edit.effect.agent_turn == "/rubric add lead times"
        with pytest.raises(CommandError):
            await commands["waive"].run([])
        idle = await commands["cancel"].run([])
        assert idle.lines == [get_agent_settings().messages.no_active_run]

    async def test_operator_commands_read_the_agent(self, harness: BuildHarness) -> None:
        h = harness()
        host = ResearchHost()
        host._agent = h.agent
        reply, _ = await converse(h, decision_script())
        commands = {c.name: c for c in host.commands()}
        history = await commands["history"].run([])
        assert history.rows and history.rows[0].cells[0] == reply.run_id
        shown = await commands["show"].run([reply.run_id])
        assert shown.lines
        with pytest.raises(CommandError):
            await commands["show"].run(["nope"])
        with pytest.raises(CommandError):
            await commands["show"].run([])
        panels = host.side_panels("s1")
        assert panels and "DELIVERED" in str(panels[0])


def _any_contract() -> object:
    from research_agent.domain.rubric import Contract, Rubric
    from research_agent.domain.scope import ScopeBrief

    brief = ScopeBrief(
        archetype="explainer", topic="t", budget=get_agent_settings().budget, concepts=("x",)
    )
    return Contract.freeze("c", brief, Rubric(), brief.budget)
