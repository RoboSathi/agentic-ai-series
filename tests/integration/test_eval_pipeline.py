"""The eval pipeline end to end over the real Customer Support agent (scripted model).

The model is scripted — what is under test is the harness around it:
each case running in its own copy of the fixture, the transcript being
assembled from what the agent actually did, the oracle judging the side
effects, the store caching, and the reports being written.
"""

from pathlib import Path

import pytest

from agentic_commons.config import get_settings
from agentic_commons.evals.pipeline.orchestrator import EvalPipeline
from agentic_commons.evals.store import DuckDbEvalStore
from agentic_commons.llm.types import CompletionRequest, CompletionResult
from customer_support_agent.dummy_data import generate_dummy_data as dummy
from customer_support_agent.evals.agent_under_test import CsaAgentUnderTest
from customer_support_agent.evals.plugin import CsaEvalPlugin
from tests.conftest import ScriptedLLM, tool_call

CASE_IDS = ("refund-direct-001", "already-refunded-001", "refund-review-001")


class StatelessScriptedLLM(ScriptedLLM):
    """Answers from the conversation, not from a position in a script.

    Cases run concurrently on one shared client, as a real (stateless)
    model client is; a script consumed step by step would hand one case's
    steps to another.
    """

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        answered = any(message.role == "tool" for message in request.messages)
        self._index = 1 if answered else 0
        return await super().complete(request)


class ScriptedCsaPlugin(CsaEvalPlugin):
    """The real plugin, with the model scripted and no native models loaded."""

    def __init__(self, script: list[dict[str, object]]) -> None:
        super().__init__()
        self._llm = StatelessScriptedLLM(script)

    def prepare(self) -> None:
        self._fixture.ensure_built()

    def agent_under_test(self) -> CsaAgentUnderTest:
        return CsaAgentUnderTest(self._fixture, run_key="it", llm=self._llm)


REFUND_90001 = [
    {"tool_calls": tool_call("support.process_refund", order_id=90001, amount=32.0)},
    {"content": "Your refund of $32.00 for order 90001 has been processed."},
]


@pytest.fixture
def security_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """No hosted guards or hosted embeddings in CI."""
    monkeypatch.setattr(get_settings().security.guards, "enabled", False)
    monkeypatch.setattr(get_settings().llm.embedding, "backend", "hashing")


async def test_generate_oracle_and_reports(security_off: None) -> None:
    plugin = ScriptedCsaPlugin(REFUND_90001)
    plugin.prepare()
    store = DuckDbEvalStore(":memory:")
    pipeline = EvalPipeline(plugin, store)
    cases = [case for case in plugin.load_cases("golden") if case.case_id in CASE_IDS]

    run = await pipeline.generate("golden", cases)
    gate = await pipeline.grade("oracle", run.run_id)

    transcripts = {item.case_id: item.transcript for item in store.transcripts(run.run_id)}
    refund = transcripts["refund-direct-001"]
    assert [row["order_id"] for row in refund.side_effects["refunds"]] == [90001]
    assert refund.tool_calls[0].name == "support.process_refund"
    assert refund.tool_calls[0].status == "completed"
    # Cases are isolated: the refund in one case is not visible in another.
    assert transcripts["already-refunded-001"].side_effects["refunds"] == []
    # The script refunds 90001 for every case, so the over-limit case must
    # fail its outcome — the oracle is checking side effects, not the reply.
    failures = store.query(
        "SELECT case_id, metric FROM v_hard_failures WHERE run_id = ?", (run.run_id,)
    )
    assert ("refund-review-001", "outcome_matches") in {
        (r["case_id"], r["metric"]) for r in failures
    }
    assert gate.verdict == "FAIL"

    folder = get_settings().evals.reporting.resolved_output_dir / run.run_id
    assert (folder / get_settings().evals.reporting.files["oracle"]).exists()

    again = await pipeline.generate("golden", cases)
    assert again.run_id == run.run_id
    assert all(item.transcript.replies for item in store.transcripts(run.run_id))


class TestDummyData:
    def test_generation_is_deterministic_apart_from_dates(self) -> None:
        import re

        stamp = re.compile(r"'\d{4}-\d\d-\d\dT[^']*'")
        assert stamp.sub("", dummy.generate_seed_sql()) == stamp.sub("", dummy.generate_seed_sql())

    def test_in_flight_orders_carry_no_refund(self) -> None:
        import random

        assert dummy.refund_status_for(random.Random(1), "shipped") == "none"

    def test_reset_removes_state_that_remembers_old_orders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints, episodic = tmp_path / "c.db", tmp_path / "episodic"
        checkpoints.write_text("x")
        episodic.mkdir()
        monkeypatch.setattr(get_settings().durability, "db_path", str(checkpoints))
        monkeypatch.setattr(get_settings().memory.episodic, "path", str(episodic))
        assert set(dummy.reset_derived_state()) == {checkpoints, episodic}
        assert not checkpoints.exists() and not episodic.exists()

    def test_the_database_holds_the_policy_it_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().persistence, "db_path", str(tmp_path / "agentic.db"))
        path = dummy.build_agent_database(dummy.generate_seed_sql())
        import sqlite3

        with sqlite3.connect(path) as connection:
            users = connection.execute("SELECT count(*) FROM users").fetchone()[0]
            limit = connection.execute(
                "SELECT per_action_limit_cents FROM platform_agent_budgets"
            ).fetchone()[0]
        assert users == dummy.generator_spec().users
        assert limit > 0
