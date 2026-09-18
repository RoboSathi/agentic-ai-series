"""Interim loop control for the refund agent (LLD §21).

This is the tracked, explicit interim state the LLD calls for:
`agentic_commons.control_loop` is parked (interface-only), so the thin
controls the running product needs today live here and migrate into that
package when its phase opens. That migration is a stated follow-up, not
a silent duplication.

What is here today is the duplicate-action layer of the six-layer
watchdog stack. It earns its place: a weak model asked for the status of
an order and then, unprompted, asked to queue a refund for it — twice.
Repeating an identical call is never useful, and when the call has a side
effect, repeating it is how one request becomes two refunds.
"""

from agentic_commons.logging import get_logger
from agentic_commons.tools import ToolCall


def action_fingerprint(call: ToolCall) -> str:
    """A stable identity for "the same call again".

    Action plus sorted arguments: argument *order* varies between model
    turns and means nothing, so a fingerprint that depended on it would
    call two identical requests different.
    """
    arguments = ",".join(f"{key}={call.params[key]}" for key in sorted(call.params))
    return f"{call.action}({arguments})"


class DuplicateActionDetector:
    """Refuses a tool call the agent has already made this run.

    Deliberately exact-match rather than semantic. A near-miss judgement
    about whether two calls "mean the same thing" belongs to the semantic
    stagnation layer, which is advisory; this layer is deterministic and
    blocking, and the two must not be confused.
    """

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}
        self._logger = get_logger("customer_support_agent")

    def seen_before(self, call: ToolCall) -> str | None:
        """The previous result for this exact call, if it was already made."""
        return self._seen.get(action_fingerprint(call))

    def record(self, call: ToolCall, output: str) -> None:
        """Remember what this call returned, so a repeat can be answered."""
        self._seen[action_fingerprint(call)] = output

    def reject(self, call: ToolCall) -> str:
        """The message the model sees when it repeats itself.

        It is given the earlier result rather than a bare refusal: the
        model repeated the call because it wanted that value, and telling
        it "no" without the value invites a third attempt.
        """
        previous = self.seen_before(call) or ""
        self._logger.warning("Suppressed a repeated tool call", action=call.action)
        return (
            f"You already called this and it returned: {previous} "
            f"Use that result rather than calling it again."
        )

    def reset(self) -> None:
        """Forget everything — one detector per run, cleared between runs."""
        self._seen.clear()

    @property
    def call_count(self) -> int:
        """Distinct calls made this run."""
        return len(self._seen)


__all__ = ["DuplicateActionDetector", "action_fingerprint"]
