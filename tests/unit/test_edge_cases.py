"""Test edge cases and exception paths."""

from dataclasses import dataclass

from agentic_commons.logging.stub import _placeholder_for


def test_placeholder_with_exception_in_construction() -> None:
    """Test that a construction exception falls back to None, not a crash."""

    class BadClass:
        """A class that cannot be constructed without args."""

        def __init__(self, required_param: str) -> None:
            self.param = required_param

    def returns_bad_class() -> BadClass:
        pass

    assert _placeholder_for(returns_bad_class) is None


def test_placeholder_with_no_annotation() -> None:
    """Test placeholder with missing return annotation."""

    def func() -> None:  # type: ignore
        pass

    result = _placeholder_for(func)
    assert result is None


def test_placeholder_for_dataclass() -> None:
    """Test placeholder creation for dataclass."""

    @dataclass
    class DataClassExample:
        name: str = "default"
        value: int = 0

    def func() -> DataClassExample:
        pass

    result = _placeholder_for(func)
    assert isinstance(result, DataClassExample)
