"""What memory holds, as one read-only picture — `make show-memory` and `/memory`.

Two steps, kept apart: `take_snapshot` reads every long-term store through
its own read API (nothing here writes), and `snapshot_tree` turns the
picture into a tree grouped the way the agent recalls — by subject, then
session. The make target and the TUI command print the same lines.

Generated ids never appear: an episodic event is shown as when it
happened and what happened, not as `event_<session>_<uuid>`.
"""

from collections import defaultdict

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.memory.dream.events import InboxCounts
from agent_platform.memory.dream.inbox import FileDreamInbox
from agent_platform.memory.dream.summaries import SUMMARY_CATEGORY
from agent_platform.memory.episodic.memory import EpisodicStore
from agent_platform.memory.governance import CuratorQueue
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.memory.semantic.memory import SemanticStore
from agent_platform.memory.types import MemoryFact, WriteResult
from agent_platform.memory.write_pipeline import store_type
from agent_platform.persistence import PlatformStore
from agent_platform.presentation.text_tree import TreeNode

_SHARED = ""


class MemorySnapshot(BaseModel):
    """Every long-term store's contents at one moment, grouped for reading."""

    agent_id: str
    # The agent's memory directory, where every store below lives.
    root: str
    # subject → session → events, newest first.
    episodic: dict[str, dict[str, list[MemoryFact]]] = Field(default_factory=dict)
    semantic: dict[str, list[MemoryFact]] = Field(default_factory=dict)
    skills: list[MemoryFact] = Field(default_factory=list)
    curator: list[tuple[int, WriteResult]] = Field(default_factory=list)
    inbox: InboxCounts = Field(default_factory=InboxCounts)

    @property
    def headline(self) -> str:
        """One line of totals, for a title."""
        sessions = sum(len(by_session) for by_session in self.episodic.values())
        facts = sum(len(items) for items in self.semantic.values())
        return (
            f"{_count(sessions, 'session')} · {_count(facts, 'fact')} · "
            f"{_count(len(self.skills), 'skill')} · "
            f"{len(self.curator)} awaiting curator · {self.inbox.unconsolidated} to dream"
        )


def take_snapshot(
    store: PlatformStore,
    subject: str | None = None,
    term: str | None = None,
) -> MemorySnapshot:
    """Read every store; optionally narrow to one subject and to entries containing `term`."""
    keep = _Filter(subject, term)
    return MemorySnapshot(
        agent_id=store.agent_id,
        root=str(get_settings().memory.files.resolved_root),
        episodic=_episodic(EpisodicStore(), keep),
        semantic=_semantic(SemanticStore(), keep),
        skills=_skills(SkillStore(), keep),
        curator=[
            (queue_id, result)
            for queue_id, result in CuratorQueue(store).pending()
            if result.fact is None or keep.fact(result.fact)
        ],
        inbox=FileDreamInbox().counts(),
    )


def snapshot_tree(snapshot: MemorySnapshot, events_per_session: int, preview: int) -> TreeNode:
    """The snapshot as a tree: store → subject → session → entry."""
    root = TreeNode(f"Memory · {snapshot.agent_id} · {snapshot.root}")
    _episodic_branch(root, snapshot, events_per_session, preview)
    _semantic_branch(root, snapshot, preview)
    skills = root.add(f"Procedural — {len(snapshot.skills)} active")
    for fact in snapshot.skills:
        skills.add(f"{fact.key}  v{fact.version}: {_cut(fact.value, preview)}")
    _empty(skills)
    curator = root.add(f"Curator queue — {len(snapshot.curator)} pending")
    for queue_id, result in snapshot.curator:
        curator.add(f"#{queue_id}  {_candidate(result, preview)}  ({result.reason})")
    _empty(curator)
    root.add(
        f"Dream inbox — {snapshot.inbox.unconsolidated} unconsolidated · "
        f"{snapshot.inbox.consolidated} consolidated"
    )
    return root


def fact_label(fact: MemoryFact, preview: int) -> str:
    """One semantic fact: what it says, and how far to trust it."""
    return (
        f"{fact.key} = {_cut(fact.value, preview)}  "
        f"({fact.provenance.kind} · {fact.confidence:.2f} · v{fact.version})"
    )


def event_label(fact: MemoryFact, preview: int) -> str:
    """One episodic entry: when, and what — a summary marked as such."""
    marker = "summary: " if fact.category == SUMMARY_CATEGORY else ""
    return f"{fact.event_time:%H:%M}  {marker}{_cut(fact.value, preview)}"


class _Filter:
    """Which subject and which text a snapshot is narrowed to."""

    def __init__(self, subject: str | None, term: str | None) -> None:
        self._subject = subject
        self._term = term.casefold() if term else None

    def fact(self, fact: MemoryFact) -> bool:
        """Whether a fact belongs in the view; agent-wide facts (no subject) always do."""
        if self._subject is not None and fact.subject_id and fact.subject_id != self._subject:
            return False
        return self._term is None or self._term in f"{fact.key} {fact.value}".casefold()


def _episodic(episodic: EpisodicStore, keep: _Filter) -> dict[str, dict[str, list[MemoryFact]]]:
    grouped: dict[str, dict[str, list[MemoryFact]]] = defaultdict(dict)
    for session in episodic.sessions():
        events = [fact for fact in episodic.events(session) if keep.fact(fact)]
        if events:
            grouped[events[0].subject_id or _SHARED][session] = events
    return dict(grouped)


def _skills(skills: SkillStore, keep: _Filter) -> list[MemoryFact]:
    """The active version of every skill (agent-wide, so no subject filter)."""
    active = [skills.get(name) for name in skills.names()]
    return [fact for fact in active if fact is not None and keep.fact(fact)]


def _semantic(semantic: SemanticStore, keep: _Filter) -> dict[str, list[MemoryFact]]:
    grouped: dict[str, list[MemoryFact]] = defaultdict(list)
    for key in semantic.keys():
        for fact in semantic.get(key):
            if keep.fact(fact):
                grouped[fact.subject_id or _SHARED].append(fact)
    return dict(grouped)


def _episodic_branch(
    root: TreeNode, snapshot: MemorySnapshot, events_per_session: int, preview: int
) -> None:
    sessions = sum(len(by_session) for by_session in snapshot.episodic.values())
    events = sum(len(items) for by in snapshot.episodic.values() for items in by.values())
    branch = root.add(f"Episodic — {_count(sessions, 'session')}, {_count(events, 'event')}")
    for subject, by_session in sorted(snapshot.episodic.items()):
        person = branch.add(_subject(subject))
        newest_first = sorted(
            by_session.items(), key=lambda item: item[1][0].event_time, reverse=True
        )
        for session, items in newest_first:
            node = person.add(
                f"session {session} ({items[0].event_time:%Y-%m-%d}) — {_count(len(items), 'event')}"
            )
            for fact in items[:events_per_session]:
                node.add(event_label(fact, preview))
            if len(items) > events_per_session:
                node.add(f"… {len(items) - events_per_session} more")
    _empty(branch)


def _semantic_branch(root: TreeNode, snapshot: MemorySnapshot, preview: int) -> None:
    facts = sum(len(items) for items in snapshot.semantic.values())
    branch = root.add(f"Semantic — {_count(facts, 'fact')}")
    for subject, items in sorted(snapshot.semantic.items()):
        person = branch.add(_subject(subject))
        for fact in items:
            person.add(fact_label(fact, preview))
    _empty(branch)


def _candidate(result: WriteResult, preview: int) -> str:
    if result.fact is None:
        return "(no fact)"
    fact = result.fact
    # Routed by category, as the pipeline routes it — not by the default type.
    return f"{store_type(fact.category)} {fact.key} = {_cut(fact.value, preview)}"


def _subject(subject: str) -> str:
    memory = get_settings().memory
    return memory.subject_label.format(subject=subject) if subject else memory.no_subject_label


def _empty(branch: TreeNode) -> None:
    if not branch.children:
        branch.add("nothing stored")


def _cut(text: str, preview: int) -> str:
    return text if len(text) <= preview else text[: preview - 1] + "…"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")
