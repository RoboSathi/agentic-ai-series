"""Money as integer minor units — never a float.

`0.1 + 0.2 != 0.3` in binary floating point, and a refund policy that says
"the amount must equal the order total" cannot be enforced with a
tolerance: `math.isclose(..., abs_tol=0.01)` quietly accepts a one-cent
difference, which is a different refund. So an amount is an integer count
of minor units (cents), and every comparison is exact.

Three rules keep it that way:

  * **Parse exactly, once, at the boundary.** Text and numbers from a
    customer, a model or a fixture go through `Decimal(str(value))` — never
    `float` arithmetic — straight into cents.
  * **Refuse, never round.** "$0.005" is not a request for a cent; it is an
    invalid amount. So are negatives, NaN, infinity and booleans.
  * **Dollars only for people.** `str(money)` renders `$1,234.56` for cards,
    replies and tables; storage, logs and comparisons use `cents`.
"""

from decimal import Decimal, InvalidOperation
from functools import total_ordering
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from agent_platform.config import get_settings


class InvalidAmountError(ValueError):
    """An amount that cannot be represented exactly in minor units."""


@total_ordering
class Money(BaseModel):
    """An exact amount of one currency, in minor units."""

    model_config = ConfigDict(frozen=True)

    cents: int
    currency: str

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a number or numeric string wherever a `Money` is expected."""
        if isinstance(value, (Money, dict)):
            return value
        parsed = cls.parse(value)
        return {"cents": parsed.cents, "currency": parsed.currency}

    @classmethod
    def parse(cls, value: Any) -> "Money":
        """An amount in major units (e.g. `"32.50"`, `32.5`, `"$1,200"`), exactly."""
        if isinstance(value, Money):
            return value
        if isinstance(value, bool) or value is None:
            raise InvalidAmountError(f"{value!r} is not an amount")
        settings = get_settings().money
        text = str(value).strip().replace(settings.symbol, "").replace(",", "")
        try:
            amount = Decimal(text)
        except InvalidOperation as exc:
            raise InvalidAmountError(f"'{value}' is not an amount") from exc
        return cls._from_decimal(amount, str(value))

    @classmethod
    def from_cents(cls, cents: int) -> "Money":
        """An amount already held in minor units (storage, arithmetic)."""
        if isinstance(cents, bool) or not isinstance(cents, int):
            raise InvalidAmountError(f"{cents!r} is not a whole number of minor units")
        return cls(cents=cents, currency=get_settings().money.currency)

    @classmethod
    def zero(cls) -> "Money":
        return cls.from_cents(0)

    @classmethod
    def _from_decimal(cls, amount: Decimal, shown: str) -> "Money":
        settings = get_settings().money
        if not amount.is_finite():
            raise InvalidAmountError(f"'{shown}' is not a finite amount")
        if amount < 0:
            raise InvalidAmountError(f"'{shown}' is negative")
        scaled = amount.scaleb(settings.minor_units)
        if scaled != scaled.to_integral_value():
            raise InvalidAmountError(
                f"'{shown}' has more than {settings.minor_units} decimal places"
            )
        return cls(cents=int(scaled), currency=settings.currency)

    # ── arithmetic and comparison: integers, same currency only ──────

    def __add__(self, other: "Money") -> "Money":
        return self._with(self.cents + self._same(other).cents)

    def __sub__(self, other: "Money") -> "Money":
        return self._with(self.cents - self._same(other).cents)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.cents < self._same(other).cents

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return (self.cents, self.currency) == (other.cents, other.currency)

    def __hash__(self) -> int:
        return hash((self.cents, self.currency))

    def __bool__(self) -> bool:
        return self.cents != 0

    def _same(self, other: "Money") -> "Money":
        if other.currency != self.currency:
            raise InvalidAmountError(f"cannot combine {self.currency} and {other.currency}")
        return other

    def _with(self, cents: int) -> "Money":
        return Money(cents=cents, currency=self.currency)

    # ── for people ───────────────────────────────────────────────────

    def __str__(self) -> str:
        settings = get_settings().money
        major = Decimal(self.cents).scaleb(-settings.minor_units)
        sign = "-" if major < 0 else ""
        return f"{sign}{settings.symbol}{abs(major):,.{settings.minor_units}f}"

    def format(self) -> str:
        """The human-readable form: `$1,234.56`."""
        return str(self)


def money_or_none(value: Any) -> Money | None:
    """Parse an amount, or None if the value is not one (for lenient readers)."""
    try:
        return Money.parse(value)
    except InvalidAmountError:
        return None
