"""The dream inbox: one JSON file per event, under the agent's `dream_events/`.

A completed turn writes its event to `pending/`. A dream reads everything
pending, and when it has consolidated an event moves the file to
`consolidated/` with the time it did so — so a re-run never reads it
twice, an operator can still see what a dream read, and old events are
pruned after the retention window. An event no dream has read is never
pruned.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.memory.dream.events import DreamEvent, InboxCounts
from agent_platform.memory.files import MemoryDirectory


class FileDreamInbox:
    """Durable, file-per-event inbox for one agent."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings().memory
        self._files = settings.files
        self._time_format = settings.dream.event_file_time_format
        self._dir = MemoryDirectory(root or self._files.dream_events_root, self._files.lock_file)
        self._logger = get_logger("memory")

    @property
    def root(self) -> Path:
        return self._dir.root

    def emit(self, event: DreamEvent) -> None:
        """Persist one event; an event id already in the inbox is ignored."""
        if self._find(event.event_id) is not None:
            return
        name = f"{event.timestamp:{self._time_format}}-{event.event_id}{self._files.event_suffix}"
        self._dir.write(f"{self._files.pending_dir}/{name}", event.model_dump_json(indent=2))

    def counts(self) -> InboxCounts:
        """Events waiting for a dream, and events a dream has read."""
        return InboxCounts(
            unconsolidated=len(self._paths(self._files.pending_dir)),
            consolidated=len(self._paths(self._files.consolidated_dir)),
        )

    def unconsolidated(self) -> dict[str, list[DreamEvent]]:
        """Pending events by session, oldest first within each."""
        sessions: dict[str, list[DreamEvent]] = defaultdict(list)
        events = [self._read(path) for path in self._paths(self._files.pending_dir)]
        for event in sorted((e for e in events if e is not None), key=lambda e: e.timestamp):
            sessions[event.session_id].append(event)
        return dict(sessions)

    def mark_consolidated(self, event_ids: list[str], at: datetime) -> None:
        """Move these events to `consolidated/`, stamped; a later dream skips them."""
        wanted = set(event_ids)
        for path in self._paths(self._files.pending_dir):
            event = self._read(path)
            if event is None or event.event_id not in wanted:
                continue
            done = event.model_copy(update={"consolidated_at": at})
            self._dir.write(
                f"{self._files.consolidated_dir}/{path.name}", done.model_dump_json(indent=2)
            )
            path.unlink()

    def prune(self, consolidated_before: datetime) -> int:
        """Delete events consolidated before a cutoff; returns how many."""
        pruned = 0
        for path in self._paths(self._files.consolidated_dir):
            event = self._read(path)
            if event is not None and event.consolidated_at is not None:
                if event.consolidated_at < consolidated_before:
                    path.unlink()
                    pruned += 1
        return pruned

    def purge(self) -> int:
        """Remove every event, read or not. Operator reset only (`make clean-memory`)."""
        removed = len(self._paths(self._files.pending_dir)) + len(
            self._paths(self._files.consolidated_dir)
        )
        for folder in (self._files.pending_dir, self._files.consolidated_dir):
            self._dir.remove_tree(folder)
        return removed

    def _paths(self, folder: str) -> list[Path]:
        return self._dir.files(folder, f"*{self._files.event_suffix}")

    def _find(self, event_id: str) -> Path | None:
        suffix = f"-{event_id}{self._files.event_suffix}"
        for folder in (self._files.pending_dir, self._files.consolidated_dir):
            for path in self._paths(folder):
                if path.name.endswith(suffix):
                    return path
        return None

    def _read(self, path: Path) -> DreamEvent | None:
        """One event file; an unreadable one is logged and left where it is."""
        try:
            return DreamEvent.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self._logger.warning(
                "Skipping unreadable dream event", file=str(path), error=str(error)
            )
            return None
