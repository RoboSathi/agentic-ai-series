"""The turn summary: one card that says, in words, what a turn did.

Stage cards answer "what did this step do?". Verifying that a turn
behaved — did it need the model twice, which tools ran, was anything
refused, what did memory keep — means reading all of them. The tally
reads them for you, from the same events, so the summary can never
disagree with the cards above it. Operational numbers (thread ids,
checkpoints) go to the log, not here.
"""

from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentic_commons.observability.types import StatusEvent

# Stages whose completion is one model call / one tool execution.
_LLM_STAGE = "llm_response_received"
_TOOL_STAGE = "tool_result"
# Stages that belong to memory, whichever operation they record.
_MEMORY_PREFIX = "memory_"
_RECALL_STAGE = "memory_recall"
# Bookkeeping stages the summary reads (they draw no card of their own).
_WORKING_MEMORY_STAGE = "working_memory_write"
_INBOX_STAGE = "dream_inbox"
_TERMINAL = ("completed", "failed", "warning", "blocked", "waiting")

# How a memory outcome reads to a person.
_MEMORY_WORDS = {
    "auto_commit": "saved",
    "curator_queue": "queued for a curator",
    "refused": "refused",
    "discard": "discarded",
}


class TurnTally:
    """Counts one turn's events; renders them as a readable summary."""

    def __init__(self) -> None:
        self._statuses: Counter[str] = Counter()
        self._memory: Counter[str] = Counter()
        self._recall: Counter[str] = Counter()
        self._recalled = 0
        self._embed_calls = 0
        self._added: dict[str, str] = {}
        self._inbox = 0
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
        if event.stage == _WORKING_MEMORY_STAGE:
            self._added[str(event.fields.get("key"))] = str(event.fields.get("memory_value"))
        elif event.stage == _INBOX_STAGE:
            self._inbox += 1
        elif event.stage == _RECALL_STAGE:
            self._recall[str(event.fields.get("answered_by") or "nothing")] += 1
            self._recalled += len(event.fields.get("recalled") or [])
            self._embed_calls += _int(event.fields.get("embed_calls"))
        elif event.stage.startswith(_MEMORY_PREFIX):
            # A write with no route is bookkeeping (the per-turn episodic line,
            # which the dream later replaces); only a governed outcome is news.
            route = str(event.fields.get("route") or "")
            if route:
                self._memory[_MEMORY_WORDS.get(route, route)] += 1

    def fields(
        self,
        outcome: str,
        seconds: float,
        working_memory: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The summary card's fields: only what a person checks, and only when there is something to say."""
        return {
            "outcome": outcome,
            "model_calls": self._llm_calls,
            "tools": ", ".join(self._tools),
            # Refused tool calls — one per call, however many cards said so.
            "refused": self._refused or None,
            "failures": self._statuses["failed"] or None,
            "warnings": self._statuses["warning"] or None,
            "working_memory": _lines(working_memory or {}),
            "added": _lines(self._added),
            "recall": self._recall_line(),
            "dream_inbox": self._inbox_line(),
            "memory": " · ".join(f"{n} {what}" for what, n in self._memory.items()),
            "tokens": self._tokens,
            "duration": f"{seconds:.1f} s",
        }

    def _recall_line(self) -> str:
        """`2 recalled · lexical · 0 embedding calls`, or nothing when recall did not run."""
        if not self._recall:
            return ""
        if not self._recalled:
            return "nothing recalled"
        tiers = ", ".join(tier for tier in self._recall if tier != "nothing")
        calls = self._embed_calls
        return (
            f"{self._recalled} recalled · {tiers} · "
            f"{calls} embedding call{'' if calls == 1 else 's'}"
        )

    def _inbox_line(self) -> str:
        """What the dream will find waiting: the turns this one queued."""
        return f"{self._inbox} turn{'' if self._inbox == 1 else 's'} queued" if self._inbox else ""

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


def _lines(pairs: Mapping[str, Any]) -> str:
    """`key: value`, one per line."""
    return "\n".join(f"{key}: {value}" for key, value in pairs.items())
