"""A generated analytics world: weeks of fleet traffic, for demos of the source, Rill and MCP.

`make dummy-data analytics` simulates every agent that ships a profile
(`src/<package>/analytics/dummy.yaml`, see `dummy_profile.py`) over
`analytics.dummy.days` — sessions, tasks spanning turns, steps with retries and
backup models, tool proposals stopping at every stage, screens, grounding and
model dependencies, Gatekeeper decisions — and writes them with the platform's
own fact models into a DuckDB of its own (never the live one), then exports its
source. The schema is the live schema by construction: the same `Fact` classes,
the same audit ledger, the same exporter.

The world is deterministic (`analytics.dummy.seed`), grows over the window,
is quieter at weekends, and has one bad day (`analytics.dummy.incident`) so a
dashboard has something to show. Volume and shape come from configuration and
the agents' profiles; nothing here names an agent.
"""

from __future__ import annotations

import importlib
import math
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from agent_platform.analytics.dummy_profile import (
    CheckSpec,
    DummyProfile,
    RoleModels,
    TaskMix,
    load_profiles,
)
from agent_platform.audit import STANDING_FIELD, AuditEntry, DuckDbAuditLedger
from agent_platform.audit.types import Decision
from agent_platform.config import get_settings
from agent_platform.config.settings import DummyAnalyticsSettings
from agent_platform.observability import checks
from agent_platform.observability.facts import (
    CLOSED,
    OPENED,
    CheckFact,
    DuckDbFactStore,
    Fact,
    ModelCallFact,
    TaskEventFact,
    ToolCallFact,
    TurnFact,
)
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.units import CANCELLED, COMPLETED, CRASHED
from agent_platform.tools.path import DUPLICATE, EXECUTED, INVALID, NOT_OFFERED, REFUSED, UNKNOWN

T = TypeVar("T")
# Spread of a simulated latency around its mean (log-normal sigma).
_LATENCY_SIGMA = 0.45
_MS = 1000.0
_SECONDS_PER_HOUR = 3600
_WEEKEND = (5, 6)
_MAX_ATTEMPTS = 3
_STANDING_VALID, _STANDING_VOIDABLE = "VALID", "VOIDABLE"
_STANDING_VOID, _STANDING_RATIFIABLE, _STANDING_UNGOVERNED = "VOID", "RATIFIABLE", "UNGOVERNED"
_ALLOW: Decision = "allow"
_DENY: Decision = "deny"
_ESCALATE: Decision = "escalate"

Tier2 = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], random.Random], list[Fact]]


@dataclass(frozen=True)
class DayConditions:
    """How the platform behaved on one day: everyday rates, or the incident's."""

    crash_rate: float
    retry_rate: float
    backup_rate: float
    latency_factor: float
    guard_failure_rate: float


@dataclass
class TurnRecord:
    """What one simulated turn produced."""

    facts: list[Fact] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    status: str = COMPLETED
    ended_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class WorldSummary:
    """What was generated, per agent."""

    days: int
    facts: dict[str, int] = field(default_factory=dict)
    audit: dict[str, int] = field(default_factory=dict)


class DummyWorld:
    """Simulates every profiled agent and writes the result (Builder over the fact store)."""

    def __init__(
        self,
        settings: DummyAnalyticsSettings | None = None,
        profiles: list[DummyProfile] | None = None,
        now: datetime | None = None,
    ) -> None:
        self._settings = settings or get_settings().analytics.dummy
        self._profiles = profiles if profiles is not None else load_profiles()
        self._now = now or datetime.now(UTC)
        self._rng = random.Random(self._settings.seed)
        self._eval_channel = get_settings().evals.channel
        self._runtime_channel = get_settings().tui.channel
        self._reserved = get_settings().analytics.tasks.reserved
        self._implicit = get_settings().analytics.tasks.implicit_kind

    def generate(self, store: DuckDbObservabilityStore) -> WorldSummary:
        """Every agent's traffic over the window, appended to `store`."""
        summary = WorldSummary(days=self._settings.days)
        for profile in self._profiles:
            facts, audit = self._agent(profile)
            facts.extend(self._tier2(profile, facts))
            DuckDbFactStore(store, agent_id=profile.agent_id).append_many(facts)
            DuckDbAuditLedger(store, agent_id=profile.agent_id).record_many(audit)
            summary.facts[profile.agent_id] = len(facts)
            summary.audit[profile.agent_id] = len(audit)
        return summary

    # -- days and sessions ------------------------------------------------------------------

    def _agent(self, profile: DummyProfile) -> tuple[list[Fact], list[AuditEntry]]:
        facts: list[Fact] = []
        audit: list[AuditEntry] = []
        for day in range(self._settings.days):
            date = (self._now - timedelta(days=self._settings.days - 1 - day)).date()
            conditions = self._conditions(day)
            for _ in range(self._sessions(profile, day, date.weekday())):
                start = self._session_start(date)
                if start > self._now:
                    continue
                session_facts, session_audit = self._session(profile, start, conditions)
                facts.extend(session_facts)
                audit.extend(session_audit)
        return facts, audit

    def _sessions(self, profile: DummyProfile, day: int, weekday: int) -> int:
        progress = day / max(self._settings.days - 1, 1)
        expected = profile.sessions_per_day * (1 + self._settings.growth * progress)
        if weekday in _WEEKEND:
            expected *= self._settings.weekend_factor
        return max(0, round(self._rng.gauss(expected, math.sqrt(expected))))

    def _conditions(self, day: int) -> DayConditions:
        incident = self._settings.incident
        if self._settings.days - 1 - day == incident.days_ago:
            return DayConditions(
                incident.crash_rate,
                incident.retry_rate,
                incident.backup_rate,
                incident.latency_factor,
                incident.guard_failure_rate,
            )
        return DayConditions(
            self._settings.crash_rate,
            self._settings.retry_rate,
            self._settings.backup_rate,
            1.0,
            0.0,
        )

    def _session_start(self, date: Any) -> datetime:
        first, last = self._settings.active_hours
        seconds = self._rng.uniform(first * _SECONDS_PER_HOUR, last * _SECONDS_PER_HOUR)
        midnight = datetime(date.year, date.month, date.day, tzinfo=UTC)
        return midnight + timedelta(seconds=seconds)

    def _session(
        self, profile: DummyProfile, start: datetime, conditions: DayConditions
    ) -> tuple[list[Fact], list[AuditEntry]]:
        session = uuid.uuid4().hex
        channel = (
            self._eval_channel
            if self._rng.random() < self._settings.eval_share
            else self._runtime_channel
        )
        keys: dict[str, Any] = {
            "agent_id": profile.agent_id,
            "session_id": session,
            "channel": channel,
        }
        facts: list[Fact] = []
        audit: list[AuditEntry] = []
        clock = start
        for _ in range(self._rng.randint(1, 3)):
            kind = _weighted(self._rng, {k: m.weight for k, m in profile.tasks.items()})
            clock = self._task(profile, kind, keys, clock, conditions, facts, audit)
        return facts, audit

    # -- tasks ------------------------------------------------------------------------------

    def _task(
        self,
        profile: DummyProfile,
        kind: str,
        keys: dict[str, Any],
        clock: datetime,
        conditions: DayConditions,
        facts: list[Fact],
        audit: list[AuditEntry],
    ) -> datetime:
        """One task's turns; returns when the next one starts."""
        mix = profile.tasks[kind]
        implicit = kind == self._implicit
        task_id = uuid.uuid4().hex
        outcome = _weighted(self._rng, mix.outcomes)
        for index in range(self._rng.randint(*mix.turns)):
            if implicit and index:
                task_id, outcome = uuid.uuid4().hex, _weighted(self._rng, mix.outcomes)
            turn = self._turn(profile, {**keys, "task_id": task_id}, clock, conditions)
            facts.extend(turn.facts)
            audit.extend(turn.audit)
            if implicit or index == 0:
                facts.append(self._event(keys, task_id, kind, OPENED, clock, turn))
            if implicit:
                ended = outcome if turn.status == COMPLETED else turn.status
                facts.append(self._closed(profile, keys, task_id, kind, ended, turn))
            clock = turn.ended_at + timedelta(
                seconds=self._rng.randint(*self._settings.turn_gap_seconds)
            )
        if not implicit:
            facts.extend(self._settle(profile, mix, keys, task_id, kind, outcome, clock))
        return clock

    def _settle(
        self,
        profile: DummyProfile,
        mix: TaskMix,
        keys: dict[str, Any],
        task_id: str,
        kind: str,
        outcome: str,
        clock: datetime,
    ) -> list[Fact]:
        """An explicit task closes with its outcome, unless it was left open (then maybe abandoned)."""
        record = TurnRecord(ended_at=clock)
        if self._rng.random() >= mix.open_rate:
            return [self._closed(profile, keys, task_id, kind, outcome, record)]
        if self._rng.random() < 1 / 2:
            return [self._closed(profile, keys, task_id, kind, self._reserved.abandoned, record)]
        return []

    def _event(
        self,
        keys: Mapping[str, Any],
        task_id: str,
        kind: str,
        event: str,
        at: datetime,
        turn: TurnRecord,
    ) -> TaskEventFact:
        turn_id = turn.facts[0].turn_id if turn.facts else ""
        return TaskEventFact(
            **keys, task_id=task_id, turn_id=turn_id, event=event, kind=kind, recorded_at=at
        )

    def _closed(
        self,
        profile: DummyProfile,
        keys: Mapping[str, Any],
        task_id: str,
        kind: str,
        outcome: str,
        turn: TurnRecord,
    ) -> TaskEventFact:
        flags = profile.outcome_flags[outcome]
        closed = self._event(keys, task_id, kind, CLOSED, turn.ended_at, turn)
        return closed.model_copy(
            update={"outcome": outcome, "success": flags.success, "terminal": flags.terminal}
        )

    # -- turns ------------------------------------------------------------------------------

    def _turn(
        self,
        profile: DummyProfile,
        keys: dict[str, Any],
        start: datetime,
        conditions: DayConditions,
    ) -> TurnRecord:
        """One turn: screens, steps (model calls, dependencies, tools), grounding, the turn row."""
        turn = TurnRecord()
        stamp: dict[str, Any] = {**keys, "turn_id": uuid.uuid4().hex}
        clock = _Clock(start)
        screened, blocked = self._screens(profile, stamp, clock, conditions, turn)
        status = self._status(conditions)
        steps = 0 if blocked else self._rng.randint(*profile.steps)
        if profile.step_cap is not None:
            steps = min(steps, profile.step_cap)
        cap_hit = (
            profile.step_cap is not None
            and steps == profile.step_cap
            and self._rng.random() < profile.cap_abort_rate
        )
        for step in range(1, steps + 1):
            self._step(profile, {**stamp, "step": step}, clock, conditions, turn)
        claims = self._grounding(profile, {**stamp, "step": steps or None}, clock, turn, steps)
        clock.advance(profile.turn_overhead_ms * conditions.latency_factor)
        turn.status, turn.ended_at = status, clock.now
        turn.facts.insert(
            0,
            TurnFact(
                **stamp,
                recorded_at=clock.now,
                status=status,
                latency_ms=(clock.now - start).total_seconds() * _MS,
                steps=steps,
                step_cap=profile.step_cap,
                step_cap_hit=cap_hit if profile.step_cap is not None else None,
                claims_checked=claims[0] if claims else None,
                claims_supported=claims[1] if claims else None,
                claims_unsupported=claims[2] if claims else None,
                screens=list(profile.screens) if profile.screens else None,
                error=self._error(status),
            ),
        )
        return turn

    def _status(self, conditions: DayConditions) -> str:
        draw = self._rng.random()
        if draw < conditions.crash_rate:
            return CRASHED
        if draw < conditions.crash_rate + self._settings.cancel_rate:
            return CANCELLED
        return COMPLETED

    @staticmethod
    def _error(status: str) -> str:
        return f"{RuntimeError.__name__}: provider unavailable" if status == CRASHED else ""

    def _screens(
        self,
        profile: DummyProfile,
        stamp: dict[str, Any],
        clock: _Clock,
        conditions: DayConditions,
        turn: TurnRecord,
    ) -> tuple[bool, bool]:
        """(screened, blocked): each declared screen, unless this turn skipped screening."""
        if not profile.screens or self._rng.random() < profile.unscreened_rate:
            return False, False
        blocked = False
        for name, spec in profile.screens.items():
            failure = max(spec.failure_rate, conditions.guard_failure_rate * _hosted(spec))
            verdict = self._verdict(spec, failure)
            blocked = blocked or verdict == checks.BLOCK
            turn.facts.append(self._check(stamp, name, checks.SCREEN, spec, verdict, clock))
        return True, blocked

    def _verdict(self, spec: CheckSpec, failure: float) -> str:
        draw = self._rng.random()
        for verdict, rate in (
            (checks.FAIL_OPEN, failure),
            (checks.BLOCK, spec.block_rate),
            (checks.FLAG, spec.flag_rate),
        ):
            if draw < rate:
                return verdict
            draw -= rate
        return checks.PASS

    def _check(
        self,
        stamp: Mapping[str, Any],
        name: str,
        purpose: str,
        spec: CheckSpec,
        verdict: str,
        clock: _Clock,
        items: int | None = None,
        flagged: int | None = None,
    ) -> CheckFact:
        latency = self._latency(spec.latency_ms)
        clock.advance(latency)
        return CheckFact(
            **stamp,
            recorded_at=clock.now,
            check_name=name,
            purpose=purpose,
            backend=spec.backend,
            model=spec.model,
            verdict=verdict,
            latency_ms=latency,
            items=items,
            flagged=flagged,
        )

    # -- steps ------------------------------------------------------------------------------

    def _step(
        self,
        profile: DummyProfile,
        stamp: dict[str, Any],
        clock: _Clock,
        conditions: DayConditions,
        turn: TurnRecord,
    ) -> None:
        role = _weighted(self._rng, {name: spec.weight for name, spec in profile.roles.items()})
        turn.facts.extend(self._attempts(role, profile.roles[role], stamp, clock, conditions))
        for name, spec in profile.dependencies.items():
            for _ in range(_count(self._rng, spec.per_step)):
                verdict = checks.ERROR if self._rng.random() < spec.failure_rate else checks.PASS
                turn.facts.append(self._check(stamp, name, checks.DEPENDENCY, spec, verdict, clock))
        if profile.tools and self._rng.random() < profile.tool_step_rate:
            for _ in range(self._rng.randint(*profile.tools_per_step)):
                self._proposal(profile, stamp, clock, turn)

    def _attempts(
        self,
        role: str,
        spec: RoleModels,
        stamp: Mapping[str, Any],
        clock: _Clock,
        conditions: DayConditions,
    ) -> list[ModelCallFact]:
        """One step's attempts: a failed primary is retried, or handed to a backup model."""
        calls: list[ModelCallFact] = []
        model, backup = spec.primary, False
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            last = attempt == _MAX_ATTEMPTS
            fails = not last and self._rng.random() < conditions.retry_rate
            if attempt == 1 and spec.backups and self._rng.random() < conditions.backup_rate:
                fails = True
            calls.append(
                self._call(role, spec, model, backup, attempt, fails, stamp, clock, conditions)
            )
            if not fails:
                return calls
            if spec.backups and self._rng.random() < conditions.backup_rate + 1 / 2:
                model, backup = self._rng.choice(spec.backups), True
        return calls

    def _call(
        self,
        role: str,
        spec: RoleModels,
        model: str,
        backup: bool,
        attempt: int,
        fails: bool,
        stamp: Mapping[str, Any],
        clock: _Clock,
        conditions: DayConditions,
    ) -> ModelCallFact:
        latency = self._latency(spec.latency_ms * conditions.latency_factor)
        clock.advance(latency)
        prompt = 0 if fails else self._rng.randint(*spec.prompt_tokens)
        completion = 0 if fails else self._rng.randint(*spec.completion_tokens)
        return ModelCallFact(
            **stamp,
            recorded_at=clock.now,
            role=role,
            provider=spec.provider,
            model=model,
            attempt=attempt,
            is_backup=backup,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            latency_ms=latency,
            finish_reason="" if fails else self._finish(),
            succeeded=not fails,
            error=f"{TimeoutError.__name__}()" if fails else "",
        )

    def _finish(self) -> str:
        return "tool_calls" if self._rng.random() < 1 / 2 else "stop"

    # -- tool proposals ---------------------------------------------------------------------

    def _proposal(
        self, profile: DummyProfile, stamp: Mapping[str, Any], clock: _Clock, turn: TurnRecord
    ) -> None:
        name = _weighted(self._rng, {tool: mix.weight for tool, mix in profile.tools.items()})
        tool = profile.tools[name]
        stage = _weighted(
            self._rng,
            {**profile.stage_rates, EXECUTED: max(1 - sum(profile.stage_rates.values()), 0.0)},
        )
        proposed = name if stage != UNKNOWN else name.rsplit(".", 1)[-1]
        fact = ToolCallFact(
            **stamp,
            recorded_at=clock.now,
            tool=proposed,
            stage=stage,
            side_effecting=tool.side_effecting if stage not in (INVALID, UNKNOWN) else None,
            reason=stage if stage != EXECUTED else "",
        )
        if stage == EXECUTED:
            fact = self._executed(
                fact, tool.failure_rate, tool.timeout_rate, tool.duration_ms, clock
            )
        turn.facts.append(fact)
        entry = self._decision(profile, stamp, name, stage, tool.side_effecting, clock)
        if entry is not None:
            turn.audit.append(entry)

    def _executed(
        self, fact: ToolCallFact, failure: float, timeout: float, duration: float, clock: _Clock
    ) -> ToolCallFact:
        timed_out = self._rng.random() < timeout
        failed = timed_out or self._rng.random() < failure
        elapsed = self._latency(duration)
        clock.advance(elapsed)
        return fact.model_copy(
            update={
                "recorded_at": clock.now,
                "succeeded": not failed,
                "timed_out": timed_out,
                "duration_ms": elapsed,
                "reason": f"{TimeoutError.__name__}()"
                if timed_out
                else ("failed" if failed else ""),
            }
        )

    def _decision(
        self,
        profile: DummyProfile,
        stamp: Mapping[str, Any],
        tool: str,
        stage: str,
        side_effecting: bool,
        clock: _Clock,
    ) -> AuditEntry | None:
        """The audit entry a governed (or ungoverned) call leaves; None where none is written."""
        if stage in (INVALID, UNKNOWN, DUPLICATE):
            return None
        if stage == EXECUTED:
            standing = (
                _STANDING_UNGOVERNED
                if not profile.governed
                else _STANDING_VOIDABLE
                if side_effecting
                else _STANDING_VALID
            )
            decision = _ALLOW
        elif stage == REFUSED and self._rng.random() < profile.escalate_share:
            decision, standing = _ESCALATE, _STANDING_RATIFIABLE
        else:
            decision, standing = _DENY, _STANDING_VOID
        return AuditEntry(
            agent_id=profile.agent_id,
            trace_id=str(stamp["turn_id"]),
            actor=profile.agent_id,
            action=tool,
            authority_context={"session_id": stamp["session_id"], STANDING_FIELD: standing},
            decision=decision,
            rationale=stage if stage in (REFUSED, NOT_OFFERED) else "",
            timestamp=clock.now,
        )

    # -- grounding and the agent's own facts ------------------------------------------------

    def _grounding(
        self,
        profile: DummyProfile,
        stamp: Mapping[str, Any],
        clock: _Clock,
        turn: TurnRecord,
        steps: int,
    ) -> tuple[int, int, int] | None:
        spec = profile.grounding
        if spec is None or steps == 0:
            return None
        claims = self._rng.randint(*spec.claims)
        unsupported = sum(self._rng.random() < spec.unsupported_rate for _ in range(claims))
        if spec.low_confidence_rate is not None:
            low = sum(self._rng.random() < spec.low_confidence_rate for _ in range(claims))
            scored = CheckSpec(backend=checks.LLM, model=spec.writer_model, latency_ms=1.0)
            verdict = checks.FLAG if low else checks.PASS
            turn.facts.append(
                self._check(
                    stamp, checks.LOGPROB, checks.GROUNDING_PURPOSE, scored, verdict, clock,
                    items=claims, flagged=low,
                )
            )  # fmt: skip
        check = CheckSpec(backend=spec.backend, model=spec.model, latency_ms=spec.latency_ms)
        verdict = checks.FLAG if unsupported else checks.PASS
        turn.facts.append(
            self._check(
                stamp, checks.GROUNDING, checks.GROUNDING_PURPOSE, check, verdict, clock,
                items=claims, flagged=unsupported,
            )
        )  # fmt: skip
        return claims, claims - unsupported, unsupported

    def _tier2(self, profile: DummyProfile, facts: list[Fact]) -> list[Fact]:
        """The agent's own facts for the tasks that ended the way its hook asks for."""
        if profile.tier2 is None:
            return []
        module, _, name = profile.tier2.hook.partition(":")
        hook: Tier2 = getattr(importlib.import_module(module), name)
        wanted = set(profile.tier2.outcomes)
        tasks = [
            fact.model_dump()
            for fact in facts
            if isinstance(fact, TaskEventFact) and fact.event == CLOSED and fact.outcome in wanted
        ]
        return hook(tasks, profile.tier2.params, self._rng)

    def _latency(self, mean_ms: float) -> float:
        return self._rng.lognormvariate(math.log(mean_ms), _LATENCY_SIGMA)


class _Clock:
    """A turn's running time, advanced by each thing that takes some."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def advance(self, milliseconds: float) -> None:
        self.now += timedelta(milliseconds=milliseconds)


def _weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    names = [name for name, weight in weights.items() if weight > 0]
    return rng.choices(names, weights=[weights[name] for name in names])[0]


def _count(rng: random.Random, mean: float) -> int:
    """A whole number of events with the given mean (its fractional part as a chance)."""
    whole = int(mean)
    return whole + (1 if rng.random() < mean - whole else 0)


def _hosted(spec: CheckSpec) -> float:
    return 1.0 if spec.backend == checks.HOSTED_MODEL else 0.0


def generate_world(settings: DummyAnalyticsSettings | None = None) -> WorldSummary:
    """Write a fresh world to the demo source's DuckDB (replacing any earlier one)."""
    resolved = settings or get_settings().analytics.dummy
    path = get_settings().analytics.sources[resolved.source].resolved_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (path, path.with_name(path.name + ".wal")):
        stale.unlink(missing_ok=True)
    store = DuckDbObservabilityStore(path)
    try:
        return DummyWorld(resolved).generate(store)
    finally:
        store.close()


__all__ = ["DayConditions", "DummyWorld", "WorldSummary", "generate_world"]
