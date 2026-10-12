"""The eval pipeline end to end over the real Customer Support agent (scripted model).

The model is scripted — what is under test is the harness around it:
each case running in its own copy of the fixture, the transcript being
assembled from what the agent actually did, the oracle judging the side
effects, the store caching, and the report being written.
"""

from pathlib import Path

import pytest

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import GoldenCase, QualityMetricSpec
from agent_platform.evals.core.pipeline.orchestrator import EvalPipeline
from agent_platform.evals.core.reporting import report_path
from agent_platform.evals.core.store import DuckDbEvalStore
from agent_platform.llm.types import CompletionRequest, CompletionResult
from customer_support_agent.config import get_agent_settings
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
        # A skill's prefetched reads are results too, but not of anything the model asked.
        prefetched = get_agent_settings().prefetch.call_id.split("{", maxsplit=1)[0]
        answered = any(
            message.role == "tool" and not (message.tool_call_id or "").startswith(prefetched)
            for message in request.messages
        )
        self._index = 1 if answered else 0
        return await super().complete(request)


class ScriptedCsaPlugin(CsaEvalPlugin):
    """The real plugin, with the model scripted and no native models loaded."""

    def __init__(self, script: list[dict[str, object]]) -> None:
        super().__init__()
        self._llm = StatelessScriptedLLM(script)

    def load_cases(self, suite: str) -> list[GoldenCase]:
        return [case for case in super().load_cases(suite) if case.case_id in CASE_IDS]

    def prepare(self) -> None:
        self._fixture.ensure_built()

    def agent_under_test(self) -> CsaAgentUnderTest:
        return CsaAgentUnderTest(self._fixture, run_key="it", llm=self._llm)

    def quality_metrics(self) -> list[QualityMetricSpec]:
        return []  # the judge is not under test here


REFUND_90001 = [
    {"tool_calls": tool_call("support.process_refund", order_id=90001, amount=32.0)},
    {"content": "Your refund of $32.00 for order 90001 has been processed."},
]


@pytest.fixture
def security_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """No hosted guards or hosted embeddings in CI."""
    monkeypatch.setattr(get_settings().security.guards, "enabled", False)
    monkeypatch.setattr(get_settings().llm.embedding, "backend", "hashing")


async def test_generate_oracle_and_report(security_off: None) -> None:
    plugin = ScriptedCsaPlugin(REFUND_90001)
    plugin.prepare()
    store = DuckDbEvalStore(":memory:")
    outcome = await EvalPipeline(plugin, store).golden()

    transcripts = {item.case_id: item.transcript for item in store.transcripts(outcome.run_id)}
    refund = transcripts["refund-direct-001"]
    assert [row["order_id"] for row in refund.side_effects["refunds"]] == [90001]
    # The refund skill's reads are answered from the deterministic checks' row, so
    # no tool runs for them: the model's call is the only one.
    assert [call.name for call in refund.tool_calls] == ["support.process_refund"]
    call = refund.tool_calls[-1]
    assert call.status == "completed" and call.turn == 0
    # The Gatekeeper's standing reaches the transcript, and the oracle grades it.
    assert call.standing == "VOIDABLE"
    changed = refund.side_effects["order_changes"]
    assert changed and changed[0]["before_status"] == changed[0]["after_status"] == "delivered"
    # Cases are isolated: the refund in one case is not visible in another.
    assert transcripts["already-refunded-001"].side_effects["refunds"] == []
    # The script refunds 90001 for every case, so the over-limit case must
    # fail its outcome — the oracle is checking side effects, not the reply.
    failures = store.query(
        "SELECT case_id, metric FROM v_hard_failures WHERE run_id = ?", (outcome.run_id,)
    )
    failed = {(r["case_id"], r["metric"]) for r in failures}
    assert ("refund-review-001", "outcome_matches") in failed
    assert ("refund-direct-001", "standing_matches") not in failed
    assert outcome.gate.verdict == "FAIL"
    assert outcome.report == report_path("deepeval", outcome.run_id) and outcome.report.exists()

    again = await EvalPipeline(plugin, store).golden()
    assert again.run_id == outcome.run_id
    assert all(item.transcript.replies for item in store.transcripts(outcome.run_id))


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
        checkpoints, memory = tmp_path / "c.db", tmp_path / "memory"
        checkpoints.write_text("x")
        (memory / "semantic").mkdir(parents=True)
        monkeypatch.setattr(get_settings().durability, "db_path", str(checkpoints))
        monkeypatch.setattr(get_settings().memory.files, "root", str(memory))
        assert set(dummy.reset_derived_state()) == {checkpoints, memory}
        assert not checkpoints.exists() and not memory.exists()

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
