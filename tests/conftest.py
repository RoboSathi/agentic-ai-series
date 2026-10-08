"""Shared test fixtures.

Two principles run through this file.

First, no test touches a real model or a real database file. The LLM is
scripted, so a test asserts on the *harness's* behaviour rather than on
whatever the model felt like producing that afternoon; the database is
`:memory:`, so tests cannot leak into each other or into the dev data.

Second, config is restored after any test that changes it. The settings
object is a process-wide singleton, so a test that flips a toggle and
leaves it flipped fails a different test later, in a different file,
for no visible reason.
"""

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest

from agent_platform.config import (
    clear_overlays,
    get_settings,
    reload_settings,
    set_active_features,
)
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    ModelToolCall,
    TokenUsage,
)
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from agent_platform.resilience.rate_limiter import reset_rate_limiter
from customer_support_agent.config.settings import reload_agent_settings
from customer_support_agent.domain.identity import Customer, set_current_customer
from customer_support_agent.domain.models import Order, User
from customer_support_agent.persistence.schema import ensure_schema

_ANALYTICS_PATH_VAR = "OBSERVABILITY__ANALYTICS_DB_PATH"
_CHECKPOINTS_PATH_VAR = "DURABILITY__DB_PATH"
_RATE_LIMIT_PATH_VAR = "RESILIENCE__RATE_LIMIT__STORE_PATH"


class ScriptedLLM(LLMClient):
    """An LLM that returns a fixed script.

    Every graph test needs the model to do something specific — call
    this tool, then that one, then answer. A real model does that
    *usually*, which is not a basis for a CI gate.
    """

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = script or [{"content": "ok"}]
        self.requests: list[CompletionRequest] = []
        self._index = 0

    @property
    def tier(self) -> str:
        return "scripted"

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, latency_ms=0.0)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return the next scripted step, repeating the last one if exhausted."""
        self.requests.append(request)
        step = self.script[min(self._index, len(self.script) - 1)]
        self._index += 1
        return CompletionResult(
            content=step.get("content", ""),
            tool_calls=step.get("tool_calls"),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tier="scripted",
            cost_usd=0.0001,
        )


# The seeded customers' codes. `u1` owns orders 10001 and 10002; `u2`
# owns 10003, which is what makes the ownership tests meaningful.
CUSTOMER_CODE = "ADA7XYZ"
OTHER_CODE = "GRC9ABC"


def tool_call(name: str, **arguments: Any) -> list[ModelToolCall]:
    """Build a model tool call for a script step."""
    return [ModelToolCall(id=name, name=name, arguments=json.dumps(arguments))]


@pytest.fixture
def pool() -> Iterator[SqliteConnectionPool]:
    """An in-memory database, isolated per test."""
    connection = SqliteConnectionPool(IN_MEMORY)
    yield connection
    connection.close()


@pytest.fixture
def seeded_pool(pool: SqliteConnectionPool) -> SqliteConnectionPool:
    """A database with the domain schema and a handful of known orders.

    Fixed values, not generated ones: a test that asserts "$32 is
    auto-approved" needs a $32 order to exist by construction.
    """
    return seed_known_orders(pool)


def seed_known_orders(pool: SqliteConnectionPool) -> SqliteConnectionPool:
    """The `seeded_pool` data, onto any pool (one that counts its queries, say)."""
    ensure_schema(pool)
    from customer_support_agent.persistence.repositories import OrdersRepository, UsersRepository

    users = UsersRepository(pool)
    orders = OrdersRepository(pool)
    users.save(
        User(
            user_id=1,
            user_code=CUSTOMER_CODE,
            created_at="2026-01-01",
            name="Ada Lovelace",
            email="ada@example.com",
        )
    )
    users.save(
        User(
            user_id=2,
            user_code=OTHER_CODE,
            created_at="2026-01-02",
            name="Grace Hopper",
            email="grace@example.com",
        )
    )
    orders.save(
        Order(
            order_id=10001,
            user_id=1,
            timestamp="2026-02-01",
            order_status="delivered",
            refund_status="none",
            amount=32.00,
        )
    )
    orders.save(
        Order(
            order_id=10002,
            user_id=1,
            timestamp="2026-02-02",
            order_status="delivered",
            refund_status="none",
            amount=210.00,
        )
    )
    orders.save(
        Order(
            order_id=10003,
            user_id=2,
            timestamp="2026-02-03",
            order_status="delivered",
            refund_status="completed",
            amount=45.00,
        )
    )
    return pool


@pytest.fixture(scope="session", autouse=True)
def offline_nli_model() -> Iterator[None]:
    """Tests never load or download the local NLI checkpoint.

    The configured provider is a local DeBERTa cross-encoder; loading it
    costs seconds and hundreds of MB, and the first run downloads it. The
    factory resolves it lazily through `shared_local_nli_model`, so this
    replaces that one seam with the deterministic, offline lexical model.
    A test that needs a real cross-encoder injects one.
    """
    from agent_platform.evals.entailment import LexicalEntailmentModel

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("agent_platform.llm.local_nli.shared_local_nli_model", LexicalEntailmentModel)
        yield


@pytest.fixture(scope="session", autouse=True)
def isolated_analytics_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the analytics database out of the developer's own.

    This file's first principle is that no test touches a real database,
    and the SQLite half has always honoured it. The DuckDB half did not:
    `DuckDbObservabilityStore()` opens the configured
    `observability.analytics_db_path`, which is the dev file. DuckDB
    takes an exclusive lock, so a developer with the TUI open watched
    ninety tests fail on `Could not set lock` — a failure that says
    nothing about the code and moves whenever someone opens a terminal.
    """
    # Every agent a test builds checkpoints its turns; the same rule keeps
    # those out of the developer's checkpoint file.
    isolated = {
        _ANALYTICS_PATH_VAR: tmp_path_factory.mktemp("analytics") / "observability.duckdb",
        _CHECKPOINTS_PATH_VAR: tmp_path_factory.mktemp("durability") / "checkpoints.db",
        # A test must never spend (or wait on) the developer's real
        # provider budget, which the TUI may be drawing on right now.
        _RATE_LIMIT_PATH_VAR: tmp_path_factory.mktemp("rate_limit") / "rate_limits.sqlite",
        # The analytics mart is the developer's dashboard data.
        "ANALYTICS__MART_DIR": tmp_path_factory.mktemp("mart"),
        # Eval history, reports and fixtures are developer data too.
        **_eval_paths(tmp_path_factory.mktemp("evals")),
    }
    previous = {name: os.environ.get(name) for name in isolated}
    os.environ.update({name: str(path) for name, path in isolated.items()})
    reload_settings()
    reload_agent_settings()
    reset_rate_limiter()
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    reload_settings()


@pytest.fixture(scope="session", autouse=True)
def no_trace_export() -> Iterator[None]:
    """No test exports a trace, whatever key the developer's shell exports.

    With a LangSmith key in the environment every agent a test builds would
    send its scripted turns to the developer's dashboard and spend their
    trace quota. Without one, the exporter is the no-op it is in CI.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name in get_settings().observability.langsmith.key_env_vars:
            patch.delenv(name, raising=False)
        yield


def _eval_paths(root: Any) -> dict[str, Any]:
    return {
        "EVALS__STORE__DB_PATH": root / "evals.duckdb",
        "EVALS__REPORTING__OUTPUT_DIR": root / "reports",
        "EVALS__PROMPTFOO__OUTPUT_DIR": root / "promptfoo",
        "CUSTOMER_SUPPORT_AGENT__EVALS__FIXTURE_DB_PATH": root / "fixture.db",
        "CUSTOMER_SUPPORT_AGENT__EVALS__WORKSPACE_DIR": root / "workspace",
    }


@pytest.fixture(autouse=True)
def declared_feature_posture() -> Iterator[None]:
    """Start every test from the configuration, not the last agent built.

    `build_agent` publishes the posture it composed so a console can
    re-toggle it, which makes the live set process state. Clearing it
    here is the same discipline the settings singleton gets: a test that
    builds a narrowed agent must not quietly narrow the next one.
    """
    set_active_features(None)
    yield
    set_active_features(None)


@pytest.fixture(autouse=True)
def authenticated_customer() -> Iterator[Customer]:
    """Bind a customer for every test, and clear it afterwards.

    Order lookups are scoped to the authenticated customer, so a test
    that binds nobody sees no orders. `u1` owns the seeded orders;
    a test about another customer rebinds explicitly.
    """
    customer = Customer(customer_id=1, name="Ada Lovelace", email="ada@example.com")
    set_current_customer(customer)
    yield customer
    set_current_customer(None)


@pytest.fixture
def verified_agent_session() -> str:
    """Session id whose customer is already verified.

    Most tests are about what the agent does *after* verification.
    Making each of them perform the handshake would test the handshake
    a hundred times and the behaviour under test once.
    """
    return "verified-session"


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    """An LLM that answers without calling anything."""
    return ScriptedLLM()


@pytest.fixture
def restore_settings() -> Iterator[None]:
    """Undo any config mutation a test performs.

    The live feature posture is process state in exactly the way the
    settings singleton is: `build_agent` publishes what it composed, and
    a later build with no explicit set reads it back. Left standing, one
    test's narrowed agent becomes the next test's default.
    """
    yield
    clear_overlays()
    set_active_features(None)
    reload_settings()


_CONTENT_FEATURES = ("pii", "prompt_injection", "content_moderation")


def _set_content_features(value: bool) -> None:
    """Flip the three content-level screening features together."""
    settings = get_settings()
    for feature in _CONTENT_FEATURES:
        settings.features = settings.features.with_override(feature, value)


@pytest.fixture
def agent_policy(seeded_pool: SqliteConnectionPool) -> Any:
    """The persisted authority a Gatekeeper rules by.

    Seeded to match what the agent actually deploys with, then *loaded*
    through the repository rather than constructed in place. The
    Gatekeeper's whole point is that its limits come from the system of
    record; a test that hands it an invented policy is testing a
    Gatekeeper nobody runs, and an unseeded database yields the
    `unconfigured` default that denies everything — which passes a
    "refused" assertion for entirely the wrong reason.
    """
    from agent_platform.persistence import SqlitePlatformStore
    from agent_platform.security.auth.policy import AgentPolicyRepository

    agent_id = get_settings().agent.id
    store = SqlitePlatformStore(seeded_pool, agent_id=agent_id)
    now = "2026-01-01T00:00:00+00:00"
    store.execute(
        "INSERT OR REPLACE INTO platform_agents (agent_id, agent_name, version, enabled, "
        "role, authority_rank, max_refund_amount, refund_budget_limit, "
        "refund_budget_window_hours, created_at, updated_at) "
        "VALUES (?, ?, ?, 1, 'support_agent', 4, 50.0, 4000.0, 24, ?, ?)",
        (agent_id, agent_id, "test", now, now),
    )
    for permission in ("read_orders", "process_refund", "queue_for_approval"):
        store.execute(
            "INSERT OR REPLACE INTO platform_agent_permissions (agent_id, permission) "
            "VALUES (?, ?)",
            (agent_id, permission),
        )
    return AgentPolicyRepository(store).get()


@pytest.fixture
def security_on(restore_settings: None) -> Iterator[None]:
    """Declare every content-level screening feature for this test."""
    _set_content_features(True)
    yield


@pytest.fixture
def security_off(restore_settings: None) -> Iterator[None]:
    """Withhold them, for the degraded-mode tests."""
    _set_content_features(False)
    yield
