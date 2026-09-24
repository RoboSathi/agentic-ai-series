"""What the person actually asked for (LLD §11.1).

The Gatekeeper's other checks all ask whether an action is *legal*: is
this actor allowed to do it, does their authority outrank what it
requires, is it within budget. None of them asks whether it is the action
that was *requested*, and those are different questions. A refund for the
right customer, on an order they own, within every limit, is still the
wrong refund if they asked about a different order.

Before this, the answer to "was this authorized?" was implicit in tool
availability: classifying an intent as a refund request put the refund
tool on the menu, and the Gatekeeper's `offered` check only confirmed the
menu had been respected. That makes exposure and authorization the same
act, so any path that widens exposure — a mis-parse, an injected
instruction that survives screening, a reused classification on a
follow-up turn — silently widens authority too.

A mandate separates them. It is a small, deterministic record saying
*this* subject asked for *this* action on *this* target, up to *this*
much, on *this* turn. It is issued from evidence, never from model
output: an order the customer named or picked from a menu, an amount read
from the authoritative record. The Gatekeeper then has something to check
a proposal against, and a write with no covering mandate is denied — not
because it was illegal, but because nobody asked for it.

The platform owns the shape and the check. What counts as evidence is the
agent's: `agentic_commons` never learns what an order is.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.tools.types import ToolCall

# A mandate that names no ceiling does not bound the amount. Whether the
# action is affordable is the budget check's question, not this one's.
UNBOUNDED = float("inf")


class ActionMandate(BaseModel):
    """Permission to take one action, granted by one request.

    `subject_id` is opaque, like `BasicSanityRequest`'s: an agent built on
    this platform need not have customers, or number them.

    `turn` is what stops a mandate outliving the request that created it.
    Authority granted by "refund order 10042" is authority for that turn,
    and a later turn that says something else does not inherit it.
    """

    model_config = {"frozen": True}

    subject_id: str
    action: str
    target: str
    max_amount: float = UNBOUNDED
    granted_by: str = "user_task"
    turn: int = 0
    evidence: str = ""

    def covers(self, call: ToolCall, target: str, amount: float, turn: int) -> bool:
        """Whether this mandate permits exactly this proposal."""
        return (
            _bare(self.action) == _bare(call.action)
            and self.target == target
            and self.turn == turn
            and amount <= self.max_amount
        )


class MandateSet(BaseModel):
    """Every mandate in force for one turn."""

    mandates: list[ActionMandate] = Field(default_factory=list)

    def covering(
        self, call: ToolCall, target: str, amount: float, turn: int
    ) -> ActionMandate | None:
        """The mandate that permits this proposal, if any does."""
        return next(
            (m for m in self.mandates if m.covers(call, target, amount, turn)),
            None,
        )


@runtime_checkable
class MandateIssuer(Protocol):
    """Turns a request into the mandates it grants.

    Implemented by the agent, because what counts as evidence that
    something was asked for is the agent's vocabulary. The only rule the
    platform imposes is the one in this docstring: issue from what the
    person said and what the record holds, never from what the model
    proposed. A mandate derived from the model's own suggestion authorizes
    the model to authorize itself.
    """

    def mandates_for(self, subject_id: str, query: str, turn: int) -> list[ActionMandate]: ...


def _bare(action: str) -> str:
    """A tool's own name, without its namespace."""
    return action.rsplit(".", maxsplit=1)[-1]


__all__ = ["UNBOUNDED", "ActionMandate", "MandateIssuer", "MandateSet"]
