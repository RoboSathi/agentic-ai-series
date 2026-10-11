"""What memory holds, as tables — `make show-memory`, `/memory` and `make dream SHOW=1`.

Two steps, kept apart: `take_snapshot` reads every long-term store through
its own read API (nothing here writes), and the table builders turn the
picture into one table per store, in the order a reader wants them:
episodic memory one table per subject, then the semantic facts and the
skills dreams consolidated, then the dream events still waiting (and those
already read) at the end. A console scoped to one verified subject shows
only `working_table` and that subject's episodic tables: facts and skills
are the agent's, common to every subject.

Generated ids never appear: an episode is shown as when it happened and
what happened, not as `event_<session>_<uuid>`.
"""

from collections import defaultdict

from pydantic import BaseModel, Field

from agent_platform.commands.types import CommandTable
from agent_platform.config import get_settings
from agent_platform.memory.dream.events import DreamEvent
from agent_platform.memory.dream.inbox import FileDreamInbox
from agent_platform.memory.dream.summaries import SUMMARY_CATEGORY
from agent_platform.memory.episodic.memory import EpisodicStore
from agent_platform.memory.procedural.skills.document import parse_stored
from agent_platform.memory.procedural.skills.types import AUTHORED, LEARNED
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.memory.semantic.memory import SemanticStore
from agent_platform.memory.types import MemoryFact
from agent_platform.memory.working.memory import WorkingMemory

_SHARED = ""


class MemorySnapshot(BaseModel):
    """Every long-term store's contents at one moment, grouped for reading."""

    agent_id: str
    # The agent's memory directory, where every store below lives.
    root: str
    # subject → session → events, newest first.
    episodic: dict[str, dict[str, list[MemoryFact]]] = Field(default_factory=dict)
    semantic: list[MemoryFact] = Field(default_factory=list)
    skills: list[MemoryFact] = Field(default_factory=list)
    # The dream inbox, newest first: pending ones carry no `consolidated_at`.
    events: list[DreamEvent] = Field(default_factory=list)


def take_snapshot(subject: str | None = None, term: str | None = None) -> MemorySnapshot:
    """Read every store; optionally narrow to one subject and to entries containing `term`."""
    settings = get_settings()
    keep = _Filter(subject, term)
    limit = settings.commands.memory_tables.inbox_events
    return MemorySnapshot(
        agent_id=settings.agent.id,
        root=str(settings.memory.files.resolved_root),
        episodic=_episodic(EpisodicStore(), keep),
        semantic=[fact for fact in _current_facts(SemanticStore()) if keep.fact(fact)],
        skills=_skills(SkillStore(), keep),
        events=[event for event in FileDreamInbox().events(limit) if keep.event(event)],
    )


def snapshot_tables(snapshot: MemorySnapshot, per_session: int) -> list[CommandTable]:
    """Episodic per subject, then semantic, then procedural, then the dream inbox last."""
    return [
        *episodic_tables(snapshot, per_session),
        semantic_table(snapshot),
        skills_table(snapshot),
        inbox_table(snapshot),
    ]


def working_table(memory: WorkingMemory) -> CommandTable:
    """This conversation's working memory: what it has established, and how much it holds."""
    words = get_settings().commands.memory_tables
    rows = [[str(key), str(value)] for key, value in memory.state.items()]
    turns = len(memory.history) // 2
    rows.append([words.history_key, words.history_value.format(n=turns)])
    return CommandTable(title=words.working_title, headers=words.working_headers, rows=rows)


def episodic_tables(snapshot: MemorySnapshot, per_session: int) -> list[CommandTable]:
    """One table per subject: each session's episodes, newest session first."""
    words = get_settings().commands.memory_tables
    if not snapshot.episodic:
        return [CommandTable(headers=words.episodic_headers, empty=words.episodic_empty)]
    tables = []
    for subject, by_session in sorted(snapshot.episodic.items()):
        sessions = sorted(by_session.items(), key=lambda item: item[1][0].event_time, reverse=True)
        rows = [
            [_when(fact), session, _kind(fact), fact.value]
            for session, facts in sessions
            for fact in facts[:per_session]
        ]
        title = words.episodic_title.format(subject=_subject(subject), n=len(by_session))
        tables.append(CommandTable(title=title, headers=words.episodic_headers, rows=rows))
    return tables


def semantic_table(snapshot: MemorySnapshot) -> CommandTable:
    words = get_settings().commands.memory_tables
    rows = [
        [_subject(fact.subject_id), fact.key, fact.value, fact.provenance.kind, f"v{fact.version}"]
        for fact in sorted(snapshot.semantic, key=lambda f: (f.subject_id, f.key))
    ]
    title = words.semantic_title.format(n=len(rows))
    return CommandTable(title=title, headers=words.semantic_headers, rows=rows, empty=words.empty)


def skills_table(snapshot: MemorySnapshot) -> CommandTable:
    words = get_settings().commands.memory_tables
    rows = [_skill_row(fact) for fact in snapshot.skills]
    title = words.skills_title.format(n=len(rows))
    return CommandTable(title=title, headers=words.skills_headers, rows=rows, empty=words.empty)


def inbox_table(snapshot: MemorySnapshot) -> CommandTable:
    """The dream events, newest first: what is waiting for `make dream`, and what it read."""
    words = get_settings().commands.memory_tables
    pending = sum(1 for event in snapshot.events if event.consolidated_at is None)
    rows = [
        [
            words.pending if event.consolidated_at is None else words.consolidated,
            event.timestamp.astimezone().strftime(words.time_format),
            event.session_id,
            _subject(event.subject_id),
            event.kind,
            event.user_input,
            str(len(event.actions)),
            ", ".join(f"{s.kind}: {s.value}" for s in event.signals) or "—",
        ]
        for event in snapshot.events
    ]
    title = words.inbox_title.format(pending=pending, consolidated=len(rows) - pending)
    return CommandTable(title=title, headers=words.inbox_headers, rows=rows, empty=words.empty)


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

    def event(self, event: DreamEvent) -> bool:
        """Whether a dream event belongs in the view: the subject's own, or anyone's."""
        if self._subject is not None and event.subject_id != self._subject:
            return False
        text = f"{event.user_input} {event.assistant_output}".casefold()
        return self._term is None or self._term in text


def _episodic(episodic: EpisodicStore, keep: _Filter) -> dict[str, dict[str, list[MemoryFact]]]:
    grouped: dict[str, dict[str, list[MemoryFact]]] = defaultdict(dict)
    for session in episodic.sessions():
        events = [fact for fact in episodic.events(session) if keep.fact(fact)]
        if events:
            grouped[events[0].subject_id or _SHARED][session] = events
    return dict(grouped)


def _current_facts(semantic: SemanticStore) -> list[MemoryFact]:
    """Each current fact once (the latest version of each subject's key)."""
    latest: dict[tuple[str, str], MemoryFact] = {}
    for key in semantic.keys():
        for fact in semantic.get(key):
            latest.setdefault((fact.subject_id, fact.key), fact)
    return list(latest.values())


def _skills(skills: SkillStore, keep: _Filter) -> list[MemoryFact]:
    """The active version of every skill (agent-wide, so no subject filter)."""
    active = [skills.get(name) for name in skills.names()]
    return [fact for fact in active if fact is not None and keep.fact(fact)]


def _skill_row(fact: MemoryFact) -> list[str]:
    """A skill: its name, version, where it came from, when it applies, what it is for."""
    settings = get_settings().skills
    document = parse_stored(fact.value, fact.key, settings)
    learned = fact.category != settings.authored_artifact_type
    return [
        fact.key,
        f"v{fact.metadata.get('version', fact.version)}",
        LEARNED if learned else AUTHORED,
        ", ".join(document.triggers) or "—",
        document.description,
    ]


def _when(fact: MemoryFact) -> str:
    return fact.event_time.astimezone().strftime(get_settings().commands.memory_tables.time_format)


def _kind(fact: MemoryFact) -> str:
    words = get_settings().commands.memory_tables
    return words.summary_kind if fact.category == SUMMARY_CATEGORY else words.event_kind


def _subject(subject: str) -> str:
    memory = get_settings().memory
    return memory.subject_label.format(subject=subject) if subject else memory.no_subject_label
