"""The run buttons every agent's console shows: Stop while running; Resume or Abort once stopped.

They used to be one agent's own, so another agent's Stop had no way back. Which to offer is
read from the run registry alone, which every durable agent has.
"""

from collections.abc import Sequence
from typing import Any

from agent_platform.config import get_settings
from agent_platform.durability.stop_report import stage_name
from agent_platform.durability.types import ABORTABLE_STATUSES
from agent_platform.interface.host.protocols import HostAction

STOP, RESUME, ABORT = "stop", "resume", "abort"
CANCEL_COMMAND, RESUME_COMMAND, ABORT_COMMAND = "/cancel", "/resume", "/abort"


def run_actions(agent: Any, session_id: str) -> Sequence[HostAction]:
    """Stop while a run is in flight; Resume and Abort for this session's stopped run."""
    controls = get_settings().tui.run_controls
    if bool(getattr(agent, "running", False)):
        stop = controls[STOP]
        return [HostAction(stop.label, CANCEL_COMMAND, True, stop.variant)]
    last = agent.runs.latest_in_session(session_id)
    if last is None or last.status not in ABORTABLE_STATUSES:
        return []
    node = stage_name(last.next_node.split(",")[0])
    resume, abort = controls[RESUME], controls[ABORT]
    return [
        HostAction(
            resume.label.format(node=node),
            f"{RESUME_COMMAND} {last.thread_id}",
            False,
            resume.variant,
        ),
        HostAction(abort.label, f"{ABORT_COMMAND} {last.thread_id}", False, abort.variant),
    ]


__all__ = ["run_actions"]
