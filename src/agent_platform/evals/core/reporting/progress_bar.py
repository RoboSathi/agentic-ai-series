"""The live progress display `make eval` shows: one bar per section.

Each section (agent runs, oracle, DeepEval, Promptfoo) gets its own line — what it is,
a bar, the percentage, done/total and the time it has taken — and keeps it when it
finishes, so the screen ends with where the time went. A section whose size is not
known shows activity rather than an invented percentage.

It draws on the console the log handlers write to (`logging.shared_console`), so a
warning — a 429 and the model it hit, a model or guard fallback — prints above the
bars instead of through them.
"""

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from agent_platform.config import get_settings
from agent_platform.logging import shared_console


class ProgressBar:
    """An `EvalProgress` drawn with Rich; start it, report through it, stop it."""

    def __init__(self, console: Console | None = None) -> None:
        self._names = get_settings().evals.reporting.overview.sections
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console or shared_console(),
        )
        self._tasks: dict[str, TaskID] = {}

    def __enter__(self) -> "ProgressBar":
        self._progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._progress.stop()

    def start(self, section: str, total: int | None) -> None:
        name = self._names.get(section, section)
        if section in self._tasks:
            self._progress.reset(self._tasks[section], total=total, description=name)
            return
        self._tasks[section] = self._progress.add_task(name, total=total)

    def advance(self, section: str, completed: int) -> None:
        task = self._tasks.get(section)
        if task is not None:
            self._progress.update(task, completed=completed)

    def finish(self, section: str) -> None:
        task = self._tasks.get(section)
        if task is None:
            return
        total = self._progress.tasks[self._index(task)].total
        if total is None:
            self._progress.update(task, total=1, completed=1)
        else:
            self._progress.update(task, completed=total)
        self._progress.stop_task(task)

    def _index(self, task: TaskID) -> int:
        return next(i for i, item in enumerate(self._progress.tasks) if item.id == task)
