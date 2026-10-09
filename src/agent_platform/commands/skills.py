"""`/skills` — an agent's procedural memory, and the curator's gate into it.

Platform-owned and agent-agnostic: the host hands it the agent's skill
catalog and, when the agent has one, its curator queue. Listing shows the
active procedures and the lessons a dream proposed; `approve` is the only
way a proposed lesson becomes a procedure, and it goes through the same
governed approval every other curated memory does.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from agent_platform.commands.types import CommandError, CommandResult
from agent_platform.config import get_settings
from agent_platform.config.settings import SkillsCommandWords
from agent_platform.memory.types import MemoryPrincipal, WriteResult
from agent_platform.memory.write_pipeline import store_type
from agent_platform.skills.catalog import SkillCatalog
from agent_platform.skills.types import Skill

_LIST = "list"
_SHOW = "show"
_HISTORY = "history"
_APPROVE = "approve"
_PROCEDURAL = "procedural"


@runtime_checkable
class CuratorPipeline(Protocol):
    """The part of the memory write pipeline a curator acts through."""

    def pending_with_ids(self) -> list[tuple[int, WriteResult]]: ...

    def approve_queued(self, queue_id: int, principal: MemoryPrincipal) -> WriteResult: ...


class SkillsCommand:
    """List, show, history and approve over the agent's procedures."""

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
        if action == _APPROVE:
            return self._approve(args[1:])
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
        return [(i, entry) for i, entry in curator.pending_with_ids() if _is_procedure(entry)]

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

    def _approve(self, args: list[str]) -> CommandResult:
        curator = self._curator()
        if curator is None:
            raise CommandError(self._words.no_curator)
        if not args or not args[0].isdigit():
            raise CommandError(self._words.needs_id.format(usage=self.usage))
        queue_id = int(args[0])
        if queue_id not in dict(self._pending()):
            raise CommandError(self._words.not_a_procedure.format(id=queue_id))
        principal = MemoryPrincipal(actor=self._words.curator_actor, role="curator")
        return self._approved(curator.approve_queued(queue_id, principal))

    def _approved(self, outcome: WriteResult) -> CommandResult:
        if not outcome.written or outcome.fact is None:
            raise CommandError(self._words.refused.format(reason=outcome.reason))
        version = outcome.fact.metadata.get("version", "")
        return CommandResult(
            title=self._words.approved.format(name=outcome.fact.key, version=version)
        )

    def _yes_no(self, value: bool) -> str:
        return self._words.active if value else self._words.inactive


def _is_procedure(entry: WriteResult) -> bool:
    return entry.fact is not None and store_type(entry.fact.category) == _PROCEDURAL


def _summary_cells(skill: Skill, words: SkillsCommandWords) -> list[str]:
    description = skill.document.description
    if len(description) > words.description_chars:
        description = description[: words.description_chars].rstrip() + "…"
    triggers = ", ".join(skill.document.triggers) or words.none
    return [skill.name, str(skill.version), skill.source, triggers, description]
