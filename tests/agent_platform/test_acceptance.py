"""Acceptance-check registry: named checks by kind, evaluated against a context."""

from agent_platform.verification.acceptance import (
    AcceptanceCheckRegistry,
    CheckOutcome,
    CheckSpec,
)


async def _length(spec: CheckSpec, context: str) -> CheckOutcome:
    return CheckOutcome(score=float(len(context)), detail="characters", evidence=[context])


async def _always(spec: CheckSpec, context: str) -> CheckOutcome:
    return CheckOutcome(score=1.0)


def _registry() -> AcceptanceCheckRegistry[str]:
    registry: AcceptanceCheckRegistry[str] = AcceptanceCheckRegistry()
    registry.register("deterministic", "length", _length)
    registry.register("judge", "always", _always)
    return registry


class TestAcceptanceCheckRegistry:
    async def test_a_score_at_or_above_threshold_passes(self) -> None:
        spec = CheckSpec(item_id="R1", kind="deterministic", name="length", threshold=3)
        [verdict] = await _registry().evaluate([spec], "abc")
        assert verdict.passed and verdict.score == 3 and verdict.evidence == ["abc"]

    async def test_below_threshold_fails(self) -> None:
        spec = CheckSpec(item_id="R1", kind="deterministic", name="length", threshold=5)
        [verdict] = await _registry().evaluate([spec], "abc")
        assert not verdict.passed and verdict.must_pass

    async def test_an_unregistered_check_fails_rather_than_passes(self) -> None:
        spec = CheckSpec(item_id="R2", kind="nli", name="missing", threshold=0)
        [verdict] = await _registry().evaluate([spec], "abc")
        assert not verdict.passed and "no check registered" in verdict.detail

    async def test_kinds_filter_what_runs(self) -> None:
        specs = [
            CheckSpec(item_id="R1", kind="deterministic", name="length", threshold=1),
            CheckSpec(item_id="R2", kind="judge", name="always", threshold=1),
        ]
        verdicts = await _registry().evaluate(specs, "abc", kinds=["deterministic"])
        assert [verdict.item_id for verdict in verdicts] == ["R1"]

    def test_names_and_has(self) -> None:
        registry = _registry()
        assert registry.names() == ["always", "length"]
        assert registry.names("judge") == ["always"]
        assert registry.has("deterministic", "length")
        assert not registry.has("nli", "length")
