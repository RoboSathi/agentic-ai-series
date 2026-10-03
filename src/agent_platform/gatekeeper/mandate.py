"""What a write is allowed to act on (LLD §11.1).

The Gatekeeper's other checks all ask whether an action is *legal*: is
this actor allowed to do it, does their authority outrank what it
requires, is it within budget. None of them asks *what it acts on*, and
that is a different question. A refund for the right customer, within
every limit, is still the wrong refund if it names an order the
conversation was never about.

A mandate answers it. It is a small, deterministic record saying *this*
subject may take *this* action on *this* target, up to *this* much, in
*this* round. It is issued from what the person supplied and what the
record holds — an identifier they typed or picked from a menu, an amount
read from the authoritative row — never from model output. A proposal
that supplies its own target authorizes itself, which is the thing this
exists to prevent.

**What a mandate is not.** It does not certify that the person *asked*
for the action. Deciding whether a sentence is a request is language
work, and an agent that does it with a keyword list does it badly in both
directions: one typo denies a legitimate request, while a bare word match
turns "do not refund order 10042" into authorization to refund it. That
reading belongs to the model, and whether the action may proceed belongs
to the agent's deterministic business rules — ownership, eligibility,
amount, limits. This is the narrow, boring middle: *which* record.

The platform owns the shape and the check. What counts as an established
target is the agent's: `agent_platform` never learns what an order is.
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.primitives.money import Money
from agent_platform.tools.types import ToolCall

# A mandate that names no ceiling does not bound the amount. Whether the
# action is affordable is the budget check's question, not this one's.
UNBOUNDED: Money | None = None


class ActionMandate(BaseModel):
    """Permission to take one action, granted by one request.

    `subject_id` is opaque, like `BasicSanityRequest`'s: an agent built on
    this platform need not have customers, or number them.

    `turn` is what stops a mandate outliving the proposal it covers: it
    is issued for the round it is checked in, and a later round asks
    again rather than inheriting.

    A conversation names its subject once and then stops naming it —
    "order 10042?" / "yes, refund it" is two turns about one order — so
    an issuer may carry forward a target the person themselves
    established earlier, and must never carry forward one the model
    proposed. `evidence` is where it records which it did.
    """

    model_config = {"frozen": True}

    subject_id: str
    action: str
    target: str
    max_amount: Money | None = UNBOUNDED
    granted_by: str = "user_task"
    turn: int = 0
    evidence: str = ""

    def covers(self, call: ToolCall, target: str, amount: Money | None, turn: int) -> bool:
        """Whether this mandate permits exactly this proposal."""
        within = self.max_amount is None or amount is None or amount <= self.max_amount
        return (
            _bare(self.action) == _bare(call.action)
            and self.target == target
            and self.turn == turn
            and within
        )


class MandateSet(BaseModel):
    """Every mandate in force for one turn."""

    mandates: list[ActionMandate] = Field(default_factory=list)

    def covering(
        self, call: ToolCall, target: str, amount: Money | None, turn: int
    ) -> ActionMandate | None:
        """The mandate that permits this proposal, if any does."""
        return next(
            (m for m in self.mandates if m.covers(call, target, amount, turn)),
            None,
        )


@runtime_checkable
class MandateIssuer(Protocol):
    """Turns a turn's established state into the writes it permits.

    Implemented by the agent, because what counts as an established
    target is the agent's vocabulary. The only rule the platform imposes
    is the one in this docstring: issue from what the person supplied and
    what the record holds, never from what the model proposed. A mandate
    derived from the model's own suggestion authorizes the model to
    authorize itself.

    `context` is what this turn established before the model was
    consulted — the same kind of material `AuthorityContext.request_context`
    already carries across this boundary. The platform neither reads it
    nor knows what is in it; it hands it over so that an issuer whose
    subject was named a turn ago is not reduced to re-reading one
    sentence. An issuer that trusts a key the model can write has broken
    the rule above, and the platform cannot tell that it has.

    `query` is passed for issuers that need it. An issuer that reaches
    for it to decide *whether* the person wanted something is doing
    language work in the authorization path — see the module docstring.
    """

    def mandates_for(
        self,
        subject_id: str,
        query: str,
        turn: int,
        context: Mapping[str, object] | None = None,
    ) -> list[ActionMandate]: ...


def _bare(action: str) -> str:
    """A tool's own name, without its namespace."""
    return action.rsplit(".", maxsplit=1)[-1]


__all__ = ["UNBOUNDED", "ActionMandate", "MandateIssuer", "MandateSet"]
