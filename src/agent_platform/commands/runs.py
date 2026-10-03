"""`/resume`: continue a stopped run from its last completed step.

Any durable agent's run that was cancelled or failed part-way keeps its
checkpoints. Resuming re-runs only the node that did not finish; a side
effect that node had already committed replays from its idempotency record
rather than happening twice. A run waiting on a person resumes with their
decision instead (an agent's own command, such as `/queue approve`).
"""

from collections.abc import Callable
from typing import Any

from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.types import CommandResult
from agent_platform.durability.protocols import DurableAgent
from agent_platform.durability.types import RunRecord

# Runs `/resume` continues on its own; an interrupted run needs a decision.
_RESUMABLE_ALONE = ("cancelled", "failed_resumable")


class ResumeCommand:
    """Resume this session's last stopped run, or one named by thread prefix."""

    group = "Runs"
    name = "resume"
    summary = "Continue a cancelled or failed run from its last completed step"
    usage = "/resume [thread-prefix]"

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
        resumed = await agent.resume(record.thread_id)
        result = CommandResult(title=f"Resumed {record.thread_id} → {resumed.status}")
        return result.line(f"ASSISTANT → {resumed.response}")

    def _target(self, agent: DurableAgent, args: list[str]) -> RunRecord | None:
        if args:
            return agent.runs.find(args[0])
        session = self._scope.session_id if self._scope is not None else ""
        return agent.runs.latest_in_session(session) if session else None


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
