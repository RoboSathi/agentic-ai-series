"""`/resume` and `/abort`: continue a stopped run from its last completed step, or end it.

Any durable agent's run that was cancelled or failed part-way keeps its
checkpoints. Resuming re-runs only the node that did not finish; a side
effect that node had already committed replays from its idempotency record
rather than happening twice. A run waiting on a person resumes with their
decision instead (an agent's own command, such as `/review approve`).
"""

from collections.abc import Callable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.durability.protocols import DurableAgent
from agent_platform.durability.types import ABORTABLE_STATUSES, RunRecord
from agent_platform.interface.commands.scope import ConsoleScope
from agent_platform.interface.commands.types import CommandResult, ConsoleEffect

# Runs `/resume` continues on its own; an interrupted run needs a decision.
_RESUMABLE_ALONE = ("cancelled", "failed_resumable")


class ResumeCommand:
    """Resume this session's last stopped run, or one named by thread prefix."""

    group = "Runs"
    name = "resume"
    summary = "Continue a cancelled or failed run from its last completed step"
    usage = "/resume [thread-prefix]"
    # The resumed turn opens with its own banner; echoing `/resume` above it says it twice.
    forwards_turn = True

    def __init__(self, agent: Callable[[], Any]) -> None:
        self._agent = agent
        self._scope: ConsoleScope | None = None

    def bind_scope(self, scope: ConsoleScope) -> None:
        """Default to the run of the session the console is on."""
        self._scope = scope

    async def run(self, args: list[str]) -> CommandResult:
        agent = self._agent()
        if not isinstance(agent, DurableAgent):
            return CommandResult(title="/resume", ok=False).line("This agent does not run durably.")
        record = self._target(agent, args)
        if record is None or record.status not in _RESUMABLE_ALONE:
            return CommandResult(title="/resume").line("Nothing here is waiting to be resumed.")
        # The console runs the resume as a turn, so it is shown as one; the thread id is the
        # run registry's business (and the log's), not the operator's.
        return CommandResult(effect=ConsoleEffect(resume_thread=record.thread_id))

    def _target(self, agent: DurableAgent, args: list[str]) -> RunRecord | None:
        return _target(agent, args, self._scope)


def _target(agent: DurableAgent, args: list[str], scope: ConsoleScope | None) -> RunRecord | None:
    """The run named by a thread prefix, or else the console session's latest."""
    if args:
        return agent.runs.find(args[0])
    session = scope.session_id if scope is not None else ""
    return agent.runs.latest_in_session(session) if session else None


class AbortCommand:
    """End this session's stopped run for good; the chat prompt is the operator's again."""

    group = "Runs"
    name = "abort"
    summary = "End a stopped run for good (its record is kept; it never resumes)"
    usage = "/abort [thread-prefix]"

    def __init__(self, agent: Callable[[], Any]) -> None:
        self._agent = agent
        self._scope: ConsoleScope | None = None

    def bind_scope(self, scope: ConsoleScope) -> None:
        """Default to the run of the session the console is on."""
        self._scope = scope

    async def run(self, args: list[str]) -> CommandResult:
        words = get_settings().durability.stop_report
        agent = self._agent()
        if not isinstance(agent, DurableAgent):
            return CommandResult(ok=False).line(words.nothing_to_abort)
        record = _target(agent, args, self._scope)
        if record is None or record.status not in ABORTABLE_STATUSES:
            return CommandResult().line(words.nothing_to_abort)
        await agent.abort(record.thread_id)
        return CommandResult().line(words.aborted)


class CancelCommand:
    """Stop the run in progress; it keeps its checkpoints and `/resume` continues it."""

    group = "Runs"
    name = "cancel"
    summary = "Stop the run in progress (its progress is kept; /resume continues it)"
    usage = "/cancel"

    def __init__(self, agent: Callable[[], Any]) -> None:
        self._agent = agent

    async def run(self, args: list[str]) -> CommandResult:
        agent = self._agent()
        stopped = isinstance(agent, DurableAgent) and agent.cancel()
        line = (
            "Cancelling — the run stops now and keeps its progress."
            if stopped
            else "Nothing is running."
        )
        return CommandResult(title="/cancel").line(line)


__all__ = ["CancelCommand", "ResumeCommand"]
