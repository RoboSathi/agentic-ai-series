"""Structured-output guarantees (E28/E29, LLD §9).

Two arms, and they are not the same guarantee:

  reject / auto_repair — validate *after* generation. The model can still
                         produce anything; you find out afterwards, and a
                         repair loop costs a round trip each time.
  constrained          — constrain the decode itself. Invalid output is
                         not unlikely, it is unrepresentable — and the
                         tail latency of the retry loop disappears with it.

Validation is what you do when you cannot constrain the decoder.
Constrained decoding is what you do when you can. Only the local
llama.cpp backend honours a real grammar here, which is why the mode is
configuration rather than an assumption.
"""

import json
from functools import wraps
from typing import Any, Awaitable, Callable, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from agentic_commons.config import get_settings
from agentic_commons.guardrails.types import (
    GuardrailMode,
    GuardrailViolationError,
    ValidationResult,
)
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest, Message
from agentic_commons.logging import get_logger, logged

T = TypeVar("T")


@runtime_checkable
class OutputValidator(Protocol):
    """Checks raw model output against a schema."""

    def validate(self, raw: str, schema: type[BaseModel]) -> ValidationResult: ...


@runtime_checkable
class ConstrainedGenerator(Protocol):
    """Generates output that cannot violate its schema, by construction."""

    async def generate_constrained(self, prompt: str, schema: type[BaseModel]) -> BaseModel: ...


class PydanticValidator:
    """Validate, and in repair mode try a bounded set of mechanical fixes.

    The repairs are deliberately mechanical — strip a code fence, cut to
    the outermost braces, drop a trailing comma. Anything cleverer is
    guessing at what the model meant, which is the failure this class
    exists to prevent.
    """

    def __init__(self, mode: GuardrailMode | None = None) -> None:
        settings = get_settings().guardrails
        self._mode: GuardrailMode = mode or settings.mode
        self._max_attempts = settings.max_repair_attempts
        self._enabled = settings.enabled
        self._logger = get_logger("guardrails")

    @property
    def mode(self) -> GuardrailMode:
        return self._mode

    @logged("guardrails")
    def validate(self, raw: str, schema: type[BaseModel]) -> ValidationResult:
        """Validate raw output, repairing it first if the mode allows."""
        if not self._enabled:
            return ValidationResult(valid=True, value=None, mode=self._mode, raw=raw)
        attempt = self._attempt(raw, schema)
        if attempt.valid or self._mode != "auto_repair":
            return attempt
        return self._repair(raw, schema)

    def validate_or_raise(self, raw: str, schema: type[BaseModel]) -> dict[str, Any]:
        """Validate and fail loudly — never return a coerced approximation."""
        result = self.validate(raw, schema)
        if not result.valid or result.value is None:
            raise GuardrailViolationError(
                f"Output failed {schema.__name__} validation: {'; '.join(result.errors)}"
            )
        return result.value

    def _attempt(self, raw: str, schema: type[BaseModel], attempts: int = 1) -> ValidationResult:
        """One validation pass over a candidate string."""
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            return ValidationResult(
                valid=False,
                errors=[error["msg"] for error in exc.errors()],
                attempts=attempts,
                mode=self._mode,
                raw=raw,
            )
        except ValueError as exc:
            return ValidationResult(
                valid=False, errors=[str(exc)], attempts=attempts, mode=self._mode, raw=raw
            )
        return ValidationResult(
            valid=True,
            value=parsed.model_dump(),
            attempts=attempts,
            mode=self._mode,
            raw=raw,
        )

    def _repair(self, raw: str, schema: type[BaseModel]) -> ValidationResult:
        """Apply mechanical repairs in order until one validates."""
        for attempt, candidate in enumerate(_repair_candidates(raw), start=2):
            if attempt > self._max_attempts + 1:
                break
            result = self._attempt(candidate, schema, attempts=attempt)
            if result.valid:
                self._logger.info("Repaired malformed output", attempts=attempt)
                return result.model_copy(update={"repaired": True})
        failed = self._attempt(raw, schema, attempts=self._max_attempts + 1)
        self._logger.warning("Output could not be repaired", errors=failed.errors)
        return failed


class LlmConstrainedGenerator:
    """Constrained decoding via the LLM client's structured-output hook.

    The client decides how to express the constraint per provider — a
    real grammar on llama.cpp, a best-effort JSON hint on NIM. This class
    validates the result regardless, because a best-effort hint is not a
    guarantee and must not be treated as one.
    """

    def __init__(self, client: LLMClient, validator: PydanticValidator | None = None) -> None:
        self._client = client
        self._validator = validator or PydanticValidator(mode="auto_repair")
        self._backend = get_settings().guardrails.constrained_backend
        self._logger = get_logger("guardrails")

    @property
    def backend(self) -> str:
        """Which constrained-decoding backend is configured."""
        return self._backend

    async def generate_constrained(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Generate against a schema, validating whatever comes back."""
        result = await self._client.complete(
            CompletionRequest(
                messages=[Message(role="user", content=prompt)],
                json_schema=schema.model_json_schema(),
            )
        )
        value = self._validator.validate_or_raise(result.content, schema)
        return schema.model_validate(value)


def fail_loud(schema: type[BaseModel]) -> Callable[..., Any]:
    """Decorator: validate an async function's string output, or raise.

    Wrapping the executor entry point rather than trusting each call site
    is what makes the guarantee hold everywhere instead of wherever
    someone remembered to check.
    """

    def decorator(func: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[dict[str, Any]]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = await func(*args, **kwargs)
            return PydanticValidator().validate_or_raise(raw, schema)

        return wrapper

    return decorator


def _repair_candidates(raw: str) -> list[str]:
    """Mechanical repair candidates, cheapest and most literal first."""
    stripped = raw.strip()
    candidates = [_strip_fence(stripped), _outermost_object(stripped)]
    candidates.append(_drop_trailing_commas(_outermost_object(stripped)))
    return [candidate for candidate in candidates if candidate and candidate != raw]


def _strip_fence(text: str) -> str:
    """Remove a markdown code fence the model wrapped its JSON in."""
    if not text.startswith("```"):
        return text
    body = text.split("```")
    return body[1].partition("\n")[2].strip() if len(body) > 1 else text


def _outermost_object(text: str) -> str:
    """The substring from the first brace to the last — drops prose."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _drop_trailing_commas(text: str) -> str:
    """Remove trailing commas, the single most common JSON syntax error."""
    try:
        json.loads(text)
        return text
    except ValueError:
        return text.replace(",}", "}").replace(",]", "]")
