"""`/kb` — inspect and maintain an agent's knowledge index (LLD §20).

Platform-owned and agent-agnostic: it knows only the `KnowledgeBase`
contract. The running agent's host hands it the knowledge base it built,
so the command serves any agent with a corpus, and says so plainly when
the agent has none.
"""

from collections.abc import Callable

from agentic_commons.commands.types import CommandError, CommandResult
from agentic_commons.memory.semantic.rag import Hit, KnowledgeBase, SyncReport

KnowledgeBaseProvider = Callable[[], KnowledgeBase | None]

_SYNC = "sync"
_REBUILD = "rebuild"
_SEARCH = "search"


class KnowledgeBaseCommand:
    """Status, sync, rebuild and search over the agent's knowledge index."""

    group = "Agent"
    name = "kb"
    summary = "Knowledge index: status, sync, rebuild, search"
    usage = "/kb [sync | rebuild | search <query>]   e.g. /kb search return window"

    def __init__(self, provider: KnowledgeBaseProvider) -> None:
        self._provider = provider

    async def run(self, args: list[str]) -> CommandResult:
        knowledge_base = self._provider()
        if knowledge_base is None:
            raise CommandError("This agent has no knowledge base wired")
        action = args[0].casefold() if args else ""
        if action in (_SYNC, _REBUILD):
            return _synced(await knowledge_base.sync(rebuild=action == _REBUILD))
        if action == _SEARCH:
            query = " ".join(args[1:]).strip()
            if not query:
                raise CommandError(f"Usage: {self.usage}")
            return _searched(query, await knowledge_base.search(query))
        if action:
            raise CommandError(f"Unknown /kb action '{args[0]}'. Usage: {self.usage}")
        return _status(knowledge_base)


def _status(knowledge_base: KnowledgeBase) -> CommandResult:
    status = knowledge_base.status()
    fingerprint = status.fingerprint
    result = CommandResult(title="Knowledge index")
    result.headers = ["field", "value"]
    result.row("index", status.db_path)
    result.row("corpus", status.corpus_dir)
    result.row("documents", str(status.documents))
    result.row("passages", str(status.passages))
    result.row(
        "embedder", f"{fingerprint.model_id} ({fingerprint.dimensions}d)" if fingerprint else "—"
    )
    result.row("last synced", status.last_synced or "never")
    return result


def _synced(report: SyncReport) -> CommandResult:
    result = CommandResult(
        title="Knowledge index rebuilt" if report.rebuilt else "Knowledge index synced"
    )
    result.headers = ["change", "documents"]
    for label, paths in (
        ("added", report.added),
        ("updated", report.updated),
        ("removed", report.removed),
        ("unchanged", report.unchanged),
    ):
        result.row(label, ", ".join(paths) or "—")
    result.line(f"{report.passages_written} passages written")
    if report.halted:
        result.ok = False
        result.line(f"Halted: {report.halted}")
    return result


def _searched(query: str, hits: list[Hit]) -> CommandResult:
    result = CommandResult(title=f"Knowledge search — {query}")
    if not hits:
        return result.line("No passages found. Is the index synced? Try /kb sync.")
    result.headers = ["#", "score", "bm25", "dense", "rrf", "rerank", "passage"]
    for position, hit in enumerate(hits, start=1):
        result.row(
            str(position),
            f"{hit.score:.3f}",
            _rank(hit.sparse_rank),
            _rank(hit.dense_rank),
            f"{hit.fused_score:.4f}",
            f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "—",
            f"{hit.passage.passage_id}  {hit.passage.title}",
        )
    return result


def _rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "—"
