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

from agent_platform.config import get_settings
from agent_platform.llm.types import TokenUsage
from agent_platform.observability.types import StatusEvent

# Model calls are counted where every one is made (the central meter, see
# `model_call`); a tool result is any card of this kind, whatever the agent
# names its stage.
_TOOL_KIND = "tool_result"
# Stages that belong to memory, whichever operation they record.
_MEMORY_PREFIX = "memory_"
_RECALL_STAGE = "memory_recall"
# Bookkeeping stages the summary reads (they draw no card of their own).
_WORKING_MEMORY_STAGE = "working_memory_write"
_INBOX_STAGE = "dream_inbox"
_TERMINAL = ("completed", "failed", "warning", "blocked", "waiting")
# The loop governor's cards: each that intervened (any decision but continue) is counted.
_LOOP_KIND = "loop_control"
_CONTINUE = "continue"

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
        self._inbox_actions = 0
        self._inbox_observations = 0
        self._inbox_waiting: int | None = None
        self._tools: list[str] = []
        self._refused = 0
        self._llm_calls = 0
        self._tokens = 0
        self._prompt_tokens = 0
        self._cached_tokens = 0
        self._cache_hits = 0
        self._interventions: list[str] = []

    def observe(self, event: StatusEvent) -> None:
        """Account for one event of the turn."""
        if event.status not in _TERMINAL:
            return
        self._statuses[event.status] += 1
        if event.event_kind == _TOOL_KIND:
            tool = str(event.fields.get("tool") or "tool")
            refused = event.status == "blocked"
            self._refused += refused
            self._tools.append(f"{tool} (refused)" if refused else tool)
        if event.event_kind == _LOOP_KIND:
            self._loop(event)
        if event.stage == _WORKING_MEMORY_STAGE:
            self._added[str(event.fields.get("key"))] = str(event.fields.get("memory_value"))
        elif event.stage == _INBOX_STAGE:
            self._inbox += 1
            self._inbox_actions += _int(event.fields.get("actions"))
            self._inbox_observations += _int(event.fields.get("observations"))
            waiting = event.fields.get("waiting")
            self._inbox_waiting = int(waiting) if waiting is not None else self._inbox_waiting
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

    def _loop(self, event: StatusEvent) -> None:
        decision = str(event.fields.get("decision") or "")
        if decision and decision != _CONTINUE:
            layer = str(event.fields.get("layer") or "")
            self._interventions.append(" · ".join(p for p in (decision.upper(), layer) if p))

    def _loop_line(self) -> str:
        """`2 · SUPPRESS · duplicate_action, FINALIZE · step_count`, or nothing."""
        if not self._interventions:
            return ""
        return f"{len(self._interventions)} · {', '.join(self._interventions)}"

    def model_call(self, usage: TokenUsage) -> None:
        """One provider call of this turn, as the central meter reports it."""
        self._llm_calls += 1
        self._tokens += usage.total_tokens
        self._prompt_tokens += usage.prompt_tokens
        self._cached_tokens += usage.cached_prompt_tokens
        self._cache_hits += usage.cache_hit

    def _cache_line(self) -> str:
        """How much of the turn's input the providers served from their prefix caches."""
        if not self._llm_calls:
            return ""
        return get_settings().observability.prompt_cache_line.format(
            hits=self._cache_hits,
            calls=self._llm_calls,
            cached=self._cached_tokens,
            prompt=self._prompt_tokens,
        )

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
            # What the loop governor did that changed the turn (suppressed, stopped).
            "loop_control": self._loop_line(),
            "failures": self._statuses["failed"] or None,
            "warnings": self._statuses["warning"] or None,
            "working_memory": _lines(working_memory or {}),
            "added": _lines(self._added),
            "recall": self._recall_line(),
            "dream_inbox": self._inbox_line(),
            "memory": " · ".join(f"{n} {what}" for what, n in self._memory.items()),
            "tokens": self._tokens,
            "prompt_cache": self._cache_line(),
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
        """What this turn left for the dream, and how much the dream will find waiting:
        `1 turn queued · 2 actions, 3 observations · 7 waiting for the next dream`."""
        if not self._inbox:
            return ""
        words = get_settings().observability.dream_inbox
        line = words.queued.format(
            turns=self._inbox, actions=self._inbox_actions, observations=self._inbox_observations
        )
        if self._inbox_waiting is not None:
            line += words.waiting.format(waiting=self._inbox_waiting)
        return line

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
