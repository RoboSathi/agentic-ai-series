"""`python -m agent_platform.evals` (what `make eval` runs) and the console's `/eval`."""

import pytest

from agent_platform.commands.evals import EvalCommand
from agent_platform.commands.types import CommandError
from agent_platform.config import get_settings
from agent_platform.evals.core import cli
from agent_platform.evals.core.pipeline.orchestrator import EvalPipeline, StageOutcome
from agent_platform.evals.core.reporting import report_path
from agent_platform.evals.core.store import DuckDbEvalStore
from tests.unit.test_eval_core import FakeGrader, FakePlugin


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> FakePlugin:
    fake = FakePlugin(grader=FakeGrader(fail={"c1"}))
    monkeypatch.setattr(cli, "resolve_eval_plugin", lambda: fake)
    return fake


class TestCli:
    def test_a_failed_gate_is_a_result_not_an_error(
        self, plugin: FakePlugin, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["run", "--stage", "deepeval"]) == 0
        run = DuckDbEvalStore().latest_run("golden")
        assert run is not None and report_path("deepeval", run.run_id).exists()
        printed = capsys.readouterr().out
        assert "fake_metric" in printed and run.run_id in printed and "FAIL" in printed

    def test_the_run_ends_with_the_time_each_section_took(
        self, plugin: FakePlugin, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["run", "--stage", "deepeval"])
        printed = capsys.readouterr().out
        sections = get_settings().evals.reporting.overview.sections
        for key in ("agent", "oracle", "total"):
            assert sections[key] in printed.split("Time taken")[-1]

    def test_a_section_that_raises_does_not_stop_the_next(
        self, plugin: FakePlugin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []

        async def broken(self: EvalPipeline) -> StageOutcome:
            raise RuntimeError("judge down")

        monkeypatch.setattr(EvalPipeline, "golden", broken)
        monkeypatch.setattr(EvalPipeline, "security", lambda _: ran.append("promptfoo") or None)
        monkeypatch.setattr(cli, "_show", lambda outcome: outcome)
        monkeypatch.setattr(cli, "_print_timings", lambda outcomes, total: None)
        assert cli.main(["run"]) == 1
        assert ran == ["promptfoo"]

    def test_durations_read_at_a_glance(self) -> None:
        assert cli.duration(42.4) == "42s"
        assert cli.duration(506) == "8m 26s"
        assert cli.duration(3780) == "1h 03m"

    def test_the_full_run_is_golden_then_the_red_team(
        self, plugin: FakePlugin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        golden = EvalPipeline.golden

        async def traced(self: EvalPipeline) -> StageOutcome:
            order.append("golden")
            outcome = await golden(self)
            monkeypatch.setattr(
                EvalPipeline, "security", lambda _: order.append("promptfoo") or outcome
            )
            return outcome

        monkeypatch.setattr(EvalPipeline, "golden", traced)
        assert cli.main(["run"]) == 0
        assert order == ["golden", "promptfoo"]

    def test_report_rebuilds_and_runs_lists(self, plugin: FakePlugin) -> None:
        cli.main(["run", "--stage", "deepeval"])
        assert cli.main(["report"]) == 0
        assert cli.main(["runs"]) == 0

    def test_an_agent_with_no_bakeoff_is_a_message(self, plugin: FakePlugin) -> None:
        assert cli.main(["run", "--stage", "intent"]) == 1

    def test_an_unknown_stage_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["run", "--stage", "judge"])


class TestEvalCommand:
    async def test_no_runs_is_a_message_not_an_error(self) -> None:
        result = await EvalCommand(DuckDbEvalStore(":memory:")).run([])
        assert result.ok and any("No eval runs" in line for line in result.lines)

    async def test_report_and_case_read_the_store(self) -> None:
        store = DuckDbEvalStore(":memory:")
        outcome = await EvalPipeline(FakePlugin(grader=FakeGrader(fail={"c1"})), store).golden()  # type: ignore[arg-type]
        command = EvalCommand(store)
        report = await command.run(["report"])
        assert report.title.endswith("FAIL")
        assert any(str(report_path("deepeval", outcome.run_id)) in line for line in report.lines)
        case = await command.run(["case", "c1"])
        assert any(row.cells[1] == "FAIL" for row in case.rows)
        runs = await command.run(["runs"])
        assert runs.rows[0].style == "red"

    async def test_an_unknown_action_is_an_error(self) -> None:
        with pytest.raises(CommandError):
            await EvalCommand(DuckDbEvalStore(":memory:")).run(["manifest"])
