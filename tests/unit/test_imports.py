"""Test that all modules can be imported and instantiated."""

from agentic_commons.resilience.circuit_breaker.types import CircuitBreakerResult


def test_circuit_breaker_result_instantiation() -> None:
    """Test CircuitBreakerResult can be instantiated."""
    result = CircuitBreakerResult(success=True, result=42)
    assert result.success is True
    assert result.result == 42


def test_circuit_breaker_result_default() -> None:
    """Test CircuitBreakerResult default values."""
    result = CircuitBreakerResult()
    assert result.success is False
    assert result.result is None
