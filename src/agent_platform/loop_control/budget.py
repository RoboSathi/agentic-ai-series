"""Hard budgets: named counters you reserve against before you spend.

The loop control already stops a trajectory on an iteration ceiling and on
a repeated action. What it did not have is a *resource* budget — so many
searches, so many tokens, so many seconds — that concurrent workers draw
on together. This is that, and it reports in the loop's own vocabulary
(`ControlCheck`, `LoopVerdict`) so a budget stop reads like any
other loop-control stop in the feed and the audit trail.

The discipline is **reserve → commit / release**, and reservation is
synchronous: check-and-decrement happens before the caller awaits
anything. On a single event loop that is atomic without a lock, which is
what stops four parallel workers each seeing "one search left" and
spending four. A reservation that turns out to cost a different amount
(a completion's tokens are known only afterwards) is committed at its
actual size; one that was never used is released.

Exhaustion is an event, raised once per counter, so the caller can show
it the moment it happens rather than discovering it on the next reserve.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import count

from pydantic import BaseModel

from agent_platform.logging import get_logger
from agent_platform.loop_control.types import BudgetLine, ControlCheck, LoopVerdict, WatchdogLayer
from agent_platform.observability.units import report_loop

Clock = Callable[[], float]
ExhaustionHandler = Callable[[LoopVerdict], None]

# The counter name wall-clock time is reported under.
SECONDS = "seconds"
_EXHAUSTED_CODE = "BUDGET_EXHAUSTED"


class BudgetExhaustedError(RuntimeError):
    """A reservation was refused: the counter is at its cap."""

    def __init__(self, counter: str, cap: float, requested: float) -> None:
        super().__init__(f"budget '{counter}' exhausted (cap {cap}, requested {requested})")
        self.counter = counter
        self.cap = cap
        self.requested = requested


@dataclass(frozen=True)
class Reservation:
    """A claim on part of one counter, not yet spent."""

    reservation_id: int
    counter: str
    amount: float


class CounterState(BaseModel):
    """One counter as it stands: its cap, what is held, what is spent."""

    name: str
    cap: float
    reserved: float = 0.0
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return max(self.cap - self.reserved - self.spent, 0.0)

    @property
    def exhausted(self) -> bool:
        """Used up. A zero cap means "not allowed" — nothing was spent, nothing ran out."""
        return self.cap > 0 and self.remaining <= 0


class BudgetGuard:
    """Named hard caps shared by every worker of one run."""

    def __init__(
        self,
        caps: Mapping[str, float],
        *,
        max_seconds: float | None = None,
        on_exhausted: ExhaustionHandler | None = None,
        clock: Clock = time.monotonic,
        layer: WatchdogLayer = "token_budget",
        step_counter: str | None = None,
    ) -> None:
        self._counters = {name: CounterState(name=name, cap=cap) for name, cap in caps.items()}
        self._max_seconds = max_seconds
        self._clock = clock
        self._started = clock()
        self._on_exhausted = on_exhausted
        self._layer = layer
        self._held: dict[int, Reservation] = {}
        self._ids = count(1)
        self._announced: set[str] = set()
        # The counter that is the turn's step cap, if this guard bounds a model loop:
        # its cap and whether it ran out are reported to the turn bracket.
        self._step_counter = step_counter
        self._logger = get_logger("budget")

    # -- reserve / commit / release ------------------------------------------------

    def try_reserve(self, counter: str, amount: float = 1) -> Reservation | None:
        """Claim `amount` now, or None if the cap (or the clock) forbids it."""
        state = self._state(counter)
        if self.expired() or state.remaining < amount:
            self._announce(counter)
            return None
        state.reserved += amount
        reservation = Reservation(next(self._ids), counter, amount)
        self._held[reservation.reservation_id] = reservation
        return reservation

    def reserve(self, counter: str, amount: float = 1) -> Reservation:
        """Claim `amount` now, raising `BudgetExhaustedError` if it cannot be had."""
        reservation = self.try_reserve(counter, amount)
        if reservation is None:
            raise BudgetExhaustedError(counter, self._state(counter).cap, amount)
        return reservation

    def commit(self, reservation: Reservation, actual: float | None = None) -> None:
        """Spend a reservation, at its actual size when that differs."""
        held = self._held.pop(reservation.reservation_id, None)
        if held is None:
            return
        state = self._state(held.counter)
        state.reserved -= held.amount
        state.spent += held.amount if actual is None else actual
        if state.exhausted:
            self._announce(held.counter)

    def release(self, reservation: Reservation) -> None:
        """Give back a reservation that was never used."""
        held = self._held.pop(reservation.reservation_id, None)
        if held is not None:
            self._state(held.counter).reserved -= held.amount

    def charge(self, counter: str, amount: float) -> bool:
        """Record spend known only afterwards; False once the counter is exhausted."""
        state = self._state(counter)
        state.spent += amount
        if counter == self._step_counter:
            report_loop(int(state.cap), state.exhausted)
        if state.exhausted:
            self._announce(counter)
            return False
        return True

    def extend(self, counter: str, amount: float) -> None:
        """Raise a counter's cap by `amount`: a grant the caller decided this run may have.

        An exhausted counter that the grant revives may be announced again when it runs out.
        """
        state = self._state(counter)
        state.cap += amount
        if not state.exhausted:
            self._announced.discard(counter)

    # -- reading -------------------------------------------------------------------

    def remaining(self, counter: str) -> float:
        return self._state(counter).remaining

    def exhausted(self, counter: str) -> bool:
        return self._state(counter).exhausted

    def elapsed(self) -> float:
        return self._clock() - self._started

    def expired(self) -> bool:
        """Whether the wall-clock budget, if any, has run out."""
        return self._max_seconds is not None and self.elapsed() >= self._max_seconds

    def exhausted_counters(self) -> list[str]:
        """Every counter at its cap, plus the clock when it has run out."""
        names = [name for name, state in self._counters.items() if state.exhausted]
        return [*names, SECONDS] if self.expired() else names

    def snapshot(self) -> dict[str, CounterState]:
        """Copies of every counter, for a progress card or a checkpoint."""
        counters = {name: state.model_copy() for name, state in self._counters.items()}
        if self._max_seconds is not None:
            counters[SECONDS] = CounterState(
                name=SECONDS, cap=self._max_seconds, spent=round(self.elapsed(), 3)
            )
        return counters

    def restore(self, spent: Mapping[str, float], *, elapsed: float | None = None) -> None:
        """Carry spend over from a checkpoint, so a resumed run cannot start fresh — the
        wall clock included: `elapsed` seconds already used are taken off what is left."""
        for name, amount in spent.items():
            if name in self._counters:
                self._counters[name].spent = amount
        seconds = spent.get(SECONDS) if elapsed is None else elapsed
        if seconds:
            self._started = self._clock() - seconds

    def report(self, counter: str) -> LoopVerdict:
        """The loop-control report for one counter's current state."""
        state = self._state(counter) if counter != SECONDS else self.snapshot()[SECONDS]
        stopped = state.exhausted or (counter == SECONDS and self.expired())
        check = ControlCheck(
            check_id=f"loop_control.budget.{counter}",
            name=f"Budget: {counter}",
            status="FAIL" if stopped else "PASS",
            reason_code=_EXHAUSTED_CODE if stopped else "BUDGET_AVAILABLE",
            reason=f"{counter}: {state.spent:g} spent of {state.cap:g}",
            observed={
                "counter": counter,
                "cap": state.cap,
                "spent": state.spent,
                "reserved": state.reserved,
            },
        )
        return LoopVerdict(
            # A spent budget stops the loop acting; what it has is still answered from.
            decision="finalize" if stopped else "continue",
            layer=("wall_clock" if counter == SECONDS else self._layer) if stopped else None,
            reason_code=check.reason_code,
            reason=check.reason,
            checks=[check],
        )

    def verdict(self, round_number: int, max_rounds: int, *, stopping: bool) -> LoopVerdict:
        """The loop-control verdict for a loop this guard budgets, every counter on it — so a
        loop that runs on reservations still reports in the one shape every agent's card reads."""
        exhausted = self.exhausted_counters()
        counter = exhausted[0] if exhausted else ""
        report = self.report(counter) if counter else None
        return LoopVerdict(
            decision="finalize" if stopping else "continue",
            layer=report.layer if report is not None and stopping else None,
            reason_code=report.reason_code
            if report is not None and stopping
            else "LOOP_CONTROL_CONTINUE",
            reason=report.reason if report is not None and stopping else "",
            checks=report.checks if report is not None else [],
            round_number=round_number,
            max_rounds=max_rounds,
            budgets={
                name: BudgetLine(spent=round(state.spent, 6), cap=state.cap)
                for name, state in self.snapshot().items()
            },
        )

    def _state(self, counter: str) -> CounterState:
        try:
            return self._counters[counter]
        except KeyError:
            raise KeyError(f"no budget counter named '{counter}'") from None

    def _announce(self, counter: str) -> None:
        """Raise the exhaustion event once per counter."""
        name = SECONDS if self.expired() else counter
        if name in self._announced:
            return
        self._announced.add(name)
        report = self.report(name)
        self._logger.warning(
            "Budget exhausted", counter=name, layer=self._layer, reason=report.reason
        )
        if self._on_exhausted is not None:
            self._on_exhausted(report)


__all__ = [
    "SECONDS",
    "BudgetExhaustedError",
    "BudgetGuard",
    "CounterState",
    "Reservation",
]
