"""`/skills` — an agent's procedural memory.

Platform-owned and agent-agnostic: the host hands it the agent's skill
catalog and, when the agent has one, its curator queue. Listing shows the
active skills and the candidates a dream proposed; a candidate becomes a
skill only when a curator approves it in `/queue` (ADR 0012), with a
comment, through the same governed approval every curated memory takes.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from agent_platform.commands.types import CommandError, CommandResult
from agent_platform.config import get_settings
from agent_platform.config.settings import SkillsCommandWords
from agent_platform.memory.procedural.skills.catalog import SkillCatalog
from agent_platform.memory.procedural.skills.types import Skill
from agent_platform.memory.types import WriteResult
from agent_platform.memory.write_pipeline import store_type

_LIST = "list"
_SHOW = "show"
_HISTORY = "history"
_PROCEDURAL = "procedural"


@runtime_checkable
class CuratorPipeline(Protocol):
    """The part of the memory write pipeline a curator acts through."""

    def pending_with_ids(self) -> list[tuple[int, WriteResult]]: ...


class SkillsCommand:
    """List, show and history over the agent's skills."""

    name = "skills"

    def __init__(
        self,
        catalog: Callable[[], SkillCatalog | None],
        curator: Callable[[], CuratorPipeline | None],
    ) -> None:
        self._catalog = catalog
        self._curator = curator
        self._words = get_settings().commands.skills
        self.summary = self._words.summary
        self.usage = self._words.usage
        self.group = self._words.group

    async def run(self, args: list[str]) -> CommandResult:
        action = args[0].casefold() if args else _LIST
        catalog = self._require_catalog()
        if action == _LIST:
            return self._list(catalog)
        if action in (_SHOW, _HISTORY):
            name = " ".join(args[1:]).strip()
            if not name:
                raise CommandError(self._words.needs_name.format(action=action, usage=self.usage))
            return self._show(catalog, name) if action == _SHOW else self._history(catalog, name)
        raise CommandError(self.usage)

    def _require_catalog(self) -> SkillCatalog:
        catalog = self._catalog()
        if catalog is None:
            raise CommandError(self._words.no_catalog)
        return catalog

    def _list(self, catalog: SkillCatalog) -> CommandResult:
        skills = catalog.active()
        result = CommandResult(title=self._words.list_title.format(n=len(skills)))
        result.headers = list(self._words.list_headers)
        for skill in skills:
            result.row(*_summary_cells(skill, self._words))
        if not skills:
            result.line(self._words.empty)
        self._pending_lines(result)
        return result

    def _pending_lines(self, result: CommandResult) -> None:
        pending = self._pending()
        if not pending:
            return
        result.line(self._words.pending_title.format(n=len(pending)))
        for queue_id, entry in pending:
            fact = entry.fact
            result.line(f"{queue_id} · {fact.key} · {fact.value}" if fact else str(queue_id))

    def _pending(self) -> list[tuple[int, WriteResult]]:
        curator = self._curator()
        if curator is None:
            return []
        return [(i, entry) for i, entry in curator.pending_with_ids() if _is_skill(entry)]

    def _show(self, catalog: SkillCatalog, name: str) -> CommandResult:
        skill = catalog.show(name)
        if skill is None:
            raise CommandError(self._words.unknown.format(name=name, usage=self.usage))
        result = CommandResult(title=skill.ref(get_settings().skills.render.ref))
        result.lines = [skill.document.description, *skill.document.body.splitlines()]
        return result

    def _history(self, catalog: SkillCatalog, name: str) -> CommandResult:
        versions = catalog.history(name)
        if not versions:
            raise CommandError(self._words.unknown.format(name=name, usage=self.usage))
        active = catalog.show(name)
        result = CommandResult(title=self._words.history_title.format(name=name, n=len(versions)))
        result.headers = list(self._words.history_headers)
        for skill in versions:
            is_active = active is not None and active.version == skill.version
            result.row(str(skill.version), self._yes_no(is_active), skill.source, skill.name)
        return result

    def _yes_no(self, value: bool) -> str:
        return self._words.active if value else self._words.inactive


def _is_skill(entry: WriteResult) -> bool:
    return entry.fact is not None and store_type(entry.fact.category) == _PROCEDURAL


def _summary_cells(skill: Skill, words: SkillsCommandWords) -> list[str]:
    description = skill.document.description
    if len(description) > words.description_chars:
        description = description[: words.description_chars].rstrip() + "…"
    triggers = ", ".join(skill.document.triggers) or words.none
    return [skill.name, str(skill.version), skill.source, triggers, description]
