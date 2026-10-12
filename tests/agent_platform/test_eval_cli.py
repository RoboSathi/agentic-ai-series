"""`python -m agent_platform.evals` (what `make eval` runs) and the console's `/eval`."""

from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.evals.core import cli
from agent_platform.evals.core.pipeline.orchestrator import EvalPipeline, StageOutcome
from agent_platform.evals.core.reporting import report_path
from agent_platform.evals.core.store import DuckDbEvalStore
from agent_platform.interface.commands.evals import EvalCommand
from agent_platform.interface.commands.types import CommandError
from tests.agent_platform.test_eval_core import FakeGrader, FakePlugin


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> FakePlugin:
    fake = FakePlugin(grader=FakeGrader(fail={"c1"}))
    monkeypatch.setattr(cli, "resolve_eval_plugin", lambda: fake)
    return fake


class TestCli:
    def test_a_failed_gate_is_a_result_not_an_error(
        self, plugin: FakePlugin, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["run", "--section", "oracle"]) == 0
        run = DuckDbEvalStore().latest_run("golden")
        assert run is not None and report_path("oracle", run.run_id).exists()
        printed = capsys.readouterr().out
        assert "fake_metric" in printed and run.run_id in printed and "FAIL" in printed

    def test_the_run_ends_with_the_time_each_section_took(
        self, plugin: FakePlugin, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["run", "--section", "oracle"])
        printed = capsys.readouterr().out
        sections = get_settings().evals.reporting.overview.sections
        for key in ("agent", "oracle", "total"):
            assert sections[key] in printed.split("Time taken")[-1]

    def test_a_section_that_raises_does_not_stop_the_next(
        self, plugin: FakePlugin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []

        async def broken(self: EvalPipeline, *args: object) -> list[StageOutcome]:
            raise RuntimeError("agent down")

        monkeypatch.setattr(EvalPipeline, "golden", broken)

        def security(self: EvalPipeline, full: bool) -> None:
            ran.append("promptfoo")

        monkeypatch.setattr(EvalPipeline, "security", security)
        monkeypatch.setattr(cli, "_show", lambda outcome: outcome)
        monkeypatch.setattr(cli, "_print_timings", lambda clock, total: None)
        assert cli.main(["run"]) == 1
        assert ran == ["promptfoo"]

    def test_every_overview_is_shown_together_after_the_last_section(
        self, plugin: FakePlugin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        async def golden(self: EvalPipeline, sections: list[str], full: bool, shown: Any) -> None:
            for section in sections:
                shown(section, section)

        def security(self: EvalPipeline, full: bool) -> str:
            events.append("ran promptfoo")
            return "promptfoo"

        monkeypatch.setattr(EvalPipeline, "golden", golden)
        monkeypatch.setattr(EvalPipeline, "security", security)
        monkeypatch.setattr(cli, "_show", lambda outcome: events.append(f"shown {outcome}"))
        monkeypatch.setattr(cli, "_print_timings", lambda clock, total: events.append("timed"))
        cli.main(["run"])
        assert events == [
            "ran promptfoo",
            "shown oracle",
            "shown deepeval",
            "shown promptfoo",
            "timed",
        ]

    def test_the_section_ctrl_c_stopped_is_named_and_timed(self) -> None:
        from agent_platform.evals.core.pipeline.orchestrator import SectionClock

        clock = SectionClock()
        with clock.measure("agent"):
            pass
        with pytest.raises(KeyboardInterrupt), clock.measure("deepeval"):
            raise KeyboardInterrupt
        assert list(clock.elapsed) == ["agent", "deepeval"]
        assert clock.unfinished == "deepeval"

    def test_a_section_that_raised_is_timed_but_not_called_stopped(self) -> None:
        from agent_platform.evals.core.pipeline.orchestrator import SectionClock

        clock = SectionClock()
        with pytest.raises(RuntimeError), clock.measure("promptfoo"):
            raise RuntimeError("npx missing")
        assert "promptfoo" in clock.elapsed and clock.unfinished is None

    def test_durations_read_at_a_glance(self) -> None:
        assert cli.duration(42.4) == "42s"
        assert cli.duration(506) == "8m 26s"
        assert cli.duration(3780) == "1h 03m"

    @pytest.mark.parametrize(
        ("argv", "golden", "red_team", "full"),
        [
            ([], ["oracle", "deepeval"], True, False),
            (["--section", "oracle"], ["oracle"], False, False),
            (["--section", "deepeval", "--full"], ["deepeval"], False, True),
            (["--section", "promptfoo"], None, True, False),
        ],
    )
    def test_each_section_runs_alone_or_all_in_order(
        self,
        plugin: FakePlugin,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        golden: list[str] | None,
        red_team: bool,
        full: bool,
    ) -> None:
        calls: list[tuple[str, object, bool]] = []

        async def traced(
            self: EvalPipeline, sections: list[str], full: bool, sink: object
        ) -> list[StageOutcome]:
            calls.append(("golden", list(sections), full))
            return []

        monkeypatch.setattr(EvalPipeline, "golden", traced)
        monkeypatch.setattr(
            EvalPipeline, "security", lambda _, full: calls.append(("promptfoo", None, full))
        )
        monkeypatch.setattr(cli, "_show", lambda outcome: outcome)
        cli.main(["run", *argv])
        expected: list[tuple[str, object, bool]] = [("golden", golden, full)] if golden else []
        expected += [("promptfoo", None, full)] if red_team else []
        assert calls == expected

    def test_report_rebuilds_and_runs_lists(self, plugin: FakePlugin) -> None:
        cli.main(["run", "--section", "oracle"])
        assert cli.main(["report"]) == 0
        assert cli.main(["runs"]) == 0

    def test_an_agent_with_no_bakeoff_is_a_message(self, plugin: FakePlugin) -> None:
        assert cli.main(["run", "--section", "intent"]) == 1

    def test_an_unknown_section_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["run", "--section", "judge"])


class TestEvalCommand:
    async def test_no_runs_is_a_message_not_an_error(self) -> None:
        result = await EvalCommand(DuckDbEvalStore(":memory:")).run([])
        assert result.ok and any("No eval runs" in line for line in result.lines)

    async def test_report_and_case_read_the_store(self) -> None:
        store = DuckDbEvalStore(":memory:")
        (outcome,) = await EvalPipeline(FakePlugin(grader=FakeGrader(fail={"c1"})), store).golden()  # type: ignore[arg-type]
        command = EvalCommand(store)
        report = await command.run(["report"])
        assert report.title.endswith("FAIL")
        assert any(str(report_path("oracle", outcome.run_id)) in line for line in report.lines)
        case = await command.run(["case", "c1"])
        assert any(row.cells[1] == "FAIL" for row in case.rows)
        runs = await command.run(["runs"])
        assert runs.rows[0].style == "red"

    async def test_an_unknown_action_is_an_error(self) -> None:
        with pytest.raises(CommandError):
            await EvalCommand(DuckDbEvalStore(":memory:")).run(["manifest"])
