"""Procedural memory — skills, versioned (E14, LLD §8, ADR 0010, ADR 0012).

Files, one directory per agent: `procedural/<name>/SKILL.md` is the active
version of each skill, verbatim — the same file an author writes — and
`MEMORY.md` lists them. Every version ever recorded is a numbered record
under `.history/<name>/`, with where it came from and whether it is the
active one in its frontmatter; exactly one version per name is active.
"""

from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.logging import get_logger, logged
from agent_platform.memory.files import (
    IndexEntry,
    MemoryDirectory,
    MemoryDocument,
    Record,
    RecordLog,
    safe_name,
    write_index,
)
from agent_platform.memory.types import MemoryFact, MemoryQuery, Provenance, WriteResult
from agent_platform.observability.types import utc_now

# The store's name on its MEMORY.md title, and the artifact type a fact gets when it names none.
_STORE_TITLE = "Procedural"
_DEFAULT_ARTIFACT = "how_to"


class SkillStore:
    """Versioned skills, with exactly one active version per name."""

    def __init__(self, root: Path | None = None, *, agent_id: str | None = None) -> None:
        settings = get_settings()
        self._files = settings.memory.files
        self._skill_file = settings.skills.skill_file
        self._agent_id = agent_id or settings.agent.id
        self._dir = MemoryDirectory(root or self._files.procedural_root, self._files.lock_file)
        self._log = RecordLog(self._dir, self._files)
        self._logger = get_logger("memory")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def root(self) -> Path:
        return self._dir.root

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Record a new version of a skill and make it the active one."""
        with self._dir.locked():
            version = self.latest_version(fact.key) + 1
            self._deactivate(fact.key)
            self._log.append(
                [fact.key],
                MemoryDocument(
                    meta={
                        "name": fact.key,
                        "version": version,
                        "artifact_type": str(fact.metadata.get("artifact_type", _DEFAULT_ARTIFACT)),
                        "source": provenance.source,
                        "provenance": provenance.kind,
                        "rationale": str(fact.metadata.get("rationale", "")),
                        "active": True,
                        "recorded_at": utc_now().isoformat(),
                    },
                    body=fact.value,
                ),
            )
            self._refresh(fact.key)
        stored = fact.model_copy(
            update={
                "memory_type": "procedural",
                "provenance": provenance,
                "metadata": {**fact.metadata, "version": version},
            }
        )
        return WriteResult(
            route="auto_commit", written=True, fact=stored, reason=f"version {version}"
        )

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Active skills matching the query."""
        if query.key:
            found = self.get(query.key)
            return [found] if found else []
        needle = query.text.casefold()
        active = [skill for skill in map(self.get, self.names()) if skill is not None]
        matching = [
            skill
            for skill in active
            if not needle or needle in skill.key.casefold() or needle in skill.value.casefold()
        ]
        return matching[: query.limit]

    def get(self, name: str, version: int | None = None) -> MemoryFact | None:
        """The active version of a skill, or a specific historical one."""
        for record in reversed(self._log.records([name])):
            meta = record.document.meta
            if (version is None and meta.get("active")) or meta.get("version") == version:
                return _to_fact(record)
        return None

    def history(self, name: str) -> list[MemoryFact]:
        """Every version of a skill, newest first."""
        records = sorted(self._log.records([name]), key=_version, reverse=True)
        return [_to_fact(record) for record in records]

    def rollback(self, name: str, version: int) -> MemoryFact | None:
        """Reactivate an earlier version — the point of keeping them all."""
        with self._dir.locked():
            target = next((r for r in self._log.records([name]) if _version(r) == version), None)
            if target is None:
                self._logger.warning("Rollback target not found", skill=name, version=version)
                return None
            self._deactivate(name)
            self._log.rewrite(target, {"active": True})
            self._refresh(name)
        self._logger.info("Rolled back skill", skill=name, version=version)
        return self.get(name)

    def retire(self, name: str) -> int:
        """Deactivate every version of a skill, keeping its history.

        A skill whose authored source was removed must stop being
        selectable without losing the record of what it used to say.
        """
        with self._dir.locked():
            self._deactivate(name)
            retired = len(self._log.records([name]))
            self._refresh(name)
        self._logger.info("Retired skill", skill=name, versions=retired)
        return retired

    def latest_version(self, name: str) -> int:
        """Highest version number recorded for a skill (0 if none)."""
        return max((_version(record) for record in self._log.records([name])), default=0)

    def purge(self) -> int:
        """Remove every skill and version. Operator reset only (`make clean-memory`)."""
        with self._dir.locked():
            removed = len(self._log.records())
            for child in self._dir.children():
                self._dir.remove_tree(child.name)
            self._dir.delete(self._files.index_file)
        return removed

    def names(self) -> list[str]:
        """Every skill with at least one version."""
        return sorted({str(record.document.meta.get("name", "")) for record in self._log.records()})

    def _deactivate(self, name: str) -> int:
        """Mark every active version of a skill inactive; returns how many. Under the lock."""
        active = [r for r in self._log.records([name]) if r.document.meta.get("active")]
        for record in active:
            self._log.rewrite(record, {"active": False})
        return len(active)

    def _refresh(self, name: str) -> None:
        """The active version as `<name>/SKILL.md` (or no file), then MEMORY.md. Under the lock."""
        view = f"{safe_name(name, self._files.name_digest_chars)}/{self._skill_file}"
        active = self.get(name)
        if active is None:
            self._dir.delete(view)
        else:
            self._dir.write(view, active.value.rstrip("\n") + "\n")
        entries = [
            IndexEntry(
                title=f"{skill.key} v{skill.metadata['version']}",
                path=f"{safe_name(skill.key, self._files.name_digest_chars)}/{self._skill_file}",
                description=str(skill.metadata.get("rationale") or skill.category),
            )
            for skill in (self.get(each) for each in self.names())
            if skill is not None
        ]
        write_index(
            self._dir,
            self._files.index_file,
            _STORE_TITLE,
            self._agent_id,
            entries,
            self._files.index,
        )


def _version(record: Record) -> int:
    value = record.document.meta.get("version", 0)
    return value if isinstance(value, int) else 0


def _to_fact(record: Record) -> MemoryFact:
    """A stored version as a fact, the shape every store returns."""
    meta: dict[str, Any] = record.document.meta
    artifact = str(meta.get("artifact_type", _DEFAULT_ARTIFACT))
    return MemoryFact(
        key=str(meta.get("name", "")),
        value=record.document.body,
        memory_type="procedural",
        category=artifact,
        provenance=Provenance.model_validate(
            {"source": str(meta.get("source", "")), "kind": meta.get("provenance")}
        ),
        metadata={
            "version": _version(record),
            "artifact_type": artifact,
            "rationale": str(meta.get("rationale", "")),
            "active": bool(meta.get("active")),
        },
    )
