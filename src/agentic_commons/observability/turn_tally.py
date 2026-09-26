"""The turn summary: one card that says, in words, what a turn did.

Stage cards answer "what did this step do?". Verifying that a turn
behaved — did it need the model twice, which tools ran, was anything
refused, what did memory keep — means reading all of them. The tally
reads them for you, from the same events, so the summary can never
disagree with the cards above it. Operational numbers (thread ids,
checkpoints) go to the log, not here.
"""

from collections import Counter
from typing import Any

from agentic_commons.observability.types import StatusEvent

# Stages whose completion is one model call / one tool execution.
_LLM_STAGE = "llm_response_received"
_TOOL_STAGE = "tool_result"
# Stages that belong to memory, whichever operation they record.
_MEMORY_PREFIX = "memory_"
_RECALL_STAGE = "memory_recall"
_TERMINAL = ("completed", "failed", "warning", "blocked", "waiting")

# How a memory outcome reads to a person.
_MEMORY_WORDS = {
    "auto_commit": "saved",
    "curator_queue": "queued for a curator",
    "refused": "refused",
    "discard": "discarded",
    "completed": "event recorded",
}


class TurnTally:
    """Counts one turn's events; renders them as a readable summary."""

    def __init__(self) -> None:
        self._statuses: Counter[str] = Counter()
        self._memory: Counter[str] = Counter()
        self._recall: Counter[str] = Counter()
        self._tools: list[str] = []
        self._refused = 0
        self._llm_calls = 0
        self._tokens = 0

    def observe(self, event: StatusEvent) -> None:
        """Account for one event of the turn."""
        if event.status not in _TERMINAL:
            return
        self._statuses[event.status] += 1
        if event.stage == _LLM_STAGE and event.status == "completed":
            self._llm_calls += 1
            self._tokens += _int(event.fields.get("total_tokens"))
        if event.stage == _TOOL_STAGE:
            tool = str(event.fields.get("tool") or "tool")
            refused = event.status == "blocked"
            self._refused += refused
            self._tools.append(f"{tool} (refused)" if refused else tool)
        if event.stage == _RECALL_STAGE:
            self._recall[str(event.fields.get("answered_by") or "nothing")] += 1
        elif event.stage.startswith(_MEMORY_PREFIX):
            route = str(event.fields.get("route") or event.status)
            self._memory[_MEMORY_WORDS.get(route, route)] += 1

    def fields(self, outcome: str, seconds: float) -> dict[str, Any]:
        """The summary card's fields — only what a person checks."""
        return {
            "outcome": outcome,
            "model_calls": self._llm_calls,
            "tools": ", ".join(self._tools) or "none",
            # Refused tool calls — one per call, however many cards said so.
            "refused": self._refused,
            "failures": self._statuses["failed"],
            "warnings": self._statuses["warning"],
            "memory": " · ".join(f"{n} {what}" for what, n in self._memory.items()) or "nothing",
            "recall": ", ".join(
                "no memory matched" if tier == "nothing" else f"answered by {tier} match"
                for tier in self._recall
            ) or "not used",
            "tokens": self._tokens,
            "duration": f"{seconds:.1f} s",
        }

    def headline(self, outcome: str, seconds: float) -> str:
        """One line: the numbers that tell you whether the turn was normal."""
        calls, tools = self._llm_calls, len(self._tools)
        parts = [
            outcome[:1].upper() + outcome[1:],
            f"{calls} model call{'' if calls == 1 else 's'}",
            f"{tools} tool{'' if tools == 1 else 's'}",
        ]
        if self._refused:
            parts.append(f"{self._refused} refused")
        if self._statuses["failed"]:
            parts.append(f"{self._statuses['failed']} failed")
        parts.append(f"{seconds:.1f} s")
        return " · ".join(parts)


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
