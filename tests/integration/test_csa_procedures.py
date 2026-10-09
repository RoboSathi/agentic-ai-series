"""The support agent's profile and procedures, end to end (ADR 0010).

Built by the real composition root against the seeded database; only the
model is scripted, so these assert what the model would actually have
been sent, and what the audit trail says it was sent.
"""

import pytest

from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.config import FeatureSet, get_settings
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.graph import build_agent
from tests.conftest import ScriptedLLM, tool_call

_REFUND = "I want a refund for order 10001"
# Withheld so the turn is hermetic: the other memory classes reach an embedding model, and
# the screens and the grounding check reach hosted models — none of them is under test here.
_WITHHELD = (
    "memory.semantic",
    "memory.episodic",
    "content_moderation",
    "prompt_injection",
    "pii",
    "hallucination_guard",
)


def _features(procedural: bool) -> FeatureSet:
    features = get_settings().features.with_override("memory.procedural", procedural)
    for path in _WITHHELD:
        features = features.with_override(path, False)
    return features


@pytest.fixture
def audit() -> DuckDbAuditLedger:
    return DuckDbAuditLedger(DuckDbObservabilityStore(":memory:"), agent_id=get_settings().agent.id)


def _script() -> ScriptedLLM:
    return ScriptedLLM(
        [
            {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
            {"content": "Your refund has been issued."},
        ]
    )


async def _system_prompt(
    pool: SqliteConnectionPool, audit: DuckDbAuditLedger, *, procedural: bool
) -> str:
    llm = _script()
    agent = build_agent(llm=llm, pool=pool, audit=audit, features=_features(procedural))
    await agent.handle(_REFUND)
    assert llm.requests, "the scripted model was never called"
    return str(llm.requests[0].messages[0].content)


def _activations(audit: DuckDbAuditLedger) -> list[AuditEntry]:
    action = get_settings().skills.audit.activation_action
    return [entry for entry in audit.recent(100) if entry.action == action]


class TestARefundTurn:
    async def test_it_carries_the_profile_and_the_refund_procedure(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        system = await _system_prompt(seeded_pool, audit, procedural=True)
        assert system.startswith("# Customer Support Agent")
        assert "# Procedure: refund-workflow" in system
        assert "whatever its size" in system
        assert "# Procedure: cancel-order" not in system

    async def test_the_audit_trail_names_the_version_used(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        await _system_prompt(seeded_pool, audit, procedural=True)
        [entry] = _activations(audit)
        assert "refund-workflow@1" in entry.authority_context["procedures"]
        assert entry.authority_context["profile"].startswith("profile@")


class TestWithoutProceduralMemory:
    async def test_the_profile_is_all_there_is(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        system = await _system_prompt(seeded_pool, audit, procedural=False)
        assert system.startswith("# Customer Support Agent")
        assert "# Procedure:" not in system
        assert _activations(audit) == []
