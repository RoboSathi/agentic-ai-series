"""What a dream extracts from events — one consolidator per long-term store.

A dream reads pending events and asks three questions, one per CoALA
store. Each question is a small protocol so an agent can answer it in its
own vocabulary — and each has a default, so an agent that customises
nothing still gets memory that works:

  * **Episodic** — what happened in this session? Default: the template
    summary of exact tool outcomes (`SessionSummarizer` with no generator).
  * **Semantic** — what is now known? Default: claims the deterministic
    pattern tier recognises in what the person said (`DeterministicFacts`).
  * **Procedural** — what keeps happening? Default: a signal recurring in
    enough sessions becomes a skill candidate (`RecurringSignalSkills`).

Consolidators only *propose*. Every write still goes through governance,
in the dream service: a proposed skill is never active until a curator
approves it.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import yaml
from pydantic import BaseModel

from agent_platform.config.settings import DreamSettings, SkillSettings
from agent_platform.memory.deterministic import DeterministicClaims
from agent_platform.memory.dream.events import DreamEvent, DreamSignal
from agent_platform.memory.dream.summaries import SessionSummary, tool_name
from agent_platform.memory.types import PrincipalRole

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
# Numbers — order ids, amounts, dates — differ between two instances of the
# same finding. Masked, "Order 10515 is already shipped" and "Order 10593 is
# already shipped" are one recurring pattern rather than two one-offs.
_SPECIFIC = re.compile(r"\$?\d[\d,.:/-]*")
_MASK = "N"
_NOT_SLUG = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z]+")


@runtime_checkable
class EpisodicConsolidator(Protocol):
    """What happened in one session, as the summary that will stand for it."""

    async def summarize(self, events: list[DreamEvent]) -> SessionSummary | None: ...


class FactCandidate(BaseModel):
    """A fact a dream proposes: who asserts it, about whom, and what it says.

    `text` is offered to the pattern tier (a person's own words);
    `key`/`value` state the fact outright (an agent's own finding).
    """

    role: PrincipalRole
    subject_id: str = ""
    key: str
    value: str
    text: str = ""
    category: str = ""
    reason: str = ""


@runtime_checkable
class SemanticConsolidator(Protocol):
    """What one session's events make known."""

    def facts(
        self, session: str, subject: str, events: list[DreamEvent]
    ) -> list[FactCandidate]: ...


class SkillCandidate(BaseModel):
    """A skill a dream proposes, written as the SKILL.md it would be."""

    name: str
    document: str
    signal: DreamSignal
    occurrences: int


@runtime_checkable
class SkillConsolidator(Protocol):
    """What keeps happening across sessions, as skills worth having."""

    def candidates(self, sessions: dict[str, list[DreamEvent]]) -> list[SkillCandidate]: ...


class DeterministicFacts:
    """Claims the pattern tier recognises in what a subject said ("my X is Y")."""

    def __init__(self, claims: DeterministicClaims | None = None) -> None:
        self._claims = claims or DeterministicClaims()

    def facts(self, session: str, subject: str, events: list[DreamEvent]) -> list[FactCandidate]:
        if not subject:
            return []
        found = []
        for event in events:
            claim = self._claims.extract(event.user_input)
            if claim is not None:
                found.append(
                    FactCandidate(
                        role="customer",
                        subject_id=subject,
                        key=claim.key,
                        value=claim.value,
                        text=event.user_input,
                    )
                )
        return found


class CombinedFacts:
    """Several semantic consolidators asked in turn — an agent's own beside the default."""

    def __init__(self, *consolidators: SemanticConsolidator) -> None:
        self._consolidators = consolidators

    def facts(self, session: str, subject: str, events: list[DreamEvent]) -> list[FactCandidate]:
        return [f for c in self._consolidators for f in c.facts(session, subject, events)]


class RecurringSignalSkills:
    """A signal seen in enough distinct sessions becomes a skill candidate.

    Generative Agents reflect once enough has accumulated; this is the
    same idea without a model. Signals are the ones an event carries, plus
    the refusals found in its tool results (`dream.refusal_markers`). The
    skill is written from configured wording, so it never claims more
    than the signal recorded.
    """

    def __init__(self, dream: DreamSettings, skills: SkillSettings) -> None:
        self._dream = dream
        self._words = dream.learned_skill
        self._skills = skills

    def candidates(self, sessions: dict[str, list[DreamEvent]]) -> list[SkillCandidate]:
        seen: dict[tuple[str, str], set[str]] = {}
        scopes: dict[tuple[str, str], Counter[str]] = {}
        for session, events in sessions.items():
            for event in events:
                for signal in self.signals(event):
                    key = (signal.kind, _SPECIFIC.sub(_MASK, signal.value))
                    seen.setdefault(key, set()).add(session)
                    scopes.setdefault(key, Counter()).update([signal.scope] if signal.scope else [])
        return [
            self._candidate(kind, value, len(where), scopes[(kind, value)])
            for (kind, value), where in sorted(seen.items())
            if len(where) >= self._dream.skill_threshold
        ]

    def signals(self, event: DreamEvent) -> list[DreamSignal]:
        """The event's own signals, plus a `refused` one per refused action."""
        found = list(event.signals)
        scope = event.scope[0] if event.scope else ""
        for action in event.actions:
            reason = _refusal_reason(str(action.result or ""), self._dream.refusal_markers)
            if reason:
                value = f"{tool_name(action.name)}: {reason}"
                found.append(
                    DreamSignal(kind=self._dream.refusal_signal_kind, value=value, scope=scope)
                )
        return found

    def _candidate(self, kind: str, value: str, count: int, scopes: Counter[str]) -> SkillCandidate:
        words = self._words
        scope = scopes.most_common(1)[0][0] if scopes else ""
        fill = {"kind": kind, "value": value, "count": count, "unit": words.unit, "scope": scope}
        lesson = words.lessons.get(kind, words.default_lesson).format(**fill)
        name = self._name(kind, value)
        precondition = (
            words.precondition_scoped.format(scope=scope) if scope else words.precondition_any
        )
        description = words.description.format(**fill, lesson=lesson)
        description = description[: self._skills.max_description_chars]
        body = words.body.format(**fill, lesson=lesson, precondition=precondition)
        meta = {
            "name": name,
            "description": description,
            "triggers": [scope] if scope else [],
            "keywords": self._keywords(value),
        }
        document = _frontmatter(meta, self._skills.frontmatter_delimiter)
        return SkillCandidate(
            name=name,
            document=f"{document}\n{body.rstrip()}\n",
            signal=DreamSignal(kind=kind, value=value, scope=scope),
            occurrences=count,
        )

    def _name(self, kind: str, value: str) -> str:
        raw = self._words.name.format(kind=kind, value=value).casefold()
        slug = _NOT_SLUG.sub("-", raw).strip("-")
        return slug[: self._skills.max_name_chars].rstrip("-")

    def _keywords(self, value: str) -> list[str]:
        words = [
            w for w in _WORD.findall(value.casefold()) if len(w) >= self._words.keyword_min_chars
        ]
        return list(dict.fromkeys(words))[: self._words.max_keywords]


def _frontmatter(meta: Mapping[str, object], delimiter: str) -> str:
    """The frontmatter of a learned skill, in the shape an authored one has."""
    dumped = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"{delimiter}\n{dumped}\n{delimiter}"


def _refusal_reason(result: str, markers: Sequence[str]) -> str:
    """The first sentence after a refusal marker, or "" if this is not a refusal."""
    for marker in markers:
        if result.startswith(marker):
            rest = result[len(marker) :].strip()
            return _SENTENCE_END.split(rest, maxsplit=1)[0]
    return ""
