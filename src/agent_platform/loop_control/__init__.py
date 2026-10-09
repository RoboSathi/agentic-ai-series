"""Public API for deterministic agent loop controls.

Implementation is intentionally split into modules:

* ``types`` — data contracts and state models.
* ``watchdogs`` — deterministic watchdog logic.
* ``protocols`` — orchestration interfaces.

This module is an import surface only; it contains no runtime logic.
"""

from agent_platform.loop_control.budget import (
    SECONDS,
    BudgetExhaustedError,
    BudgetGuard,
    CounterState,
    Reservation,
)
from agent_platform.loop_control.protocols import (
    CheckpointStore,
    LoopController,
    ReactStep,
    StateMachine,
)
from agent_platform.loop_control.types import (
    AgentState,
    Checkpoint,
    CheckStatus,
    ControlCheck,
    IllegalTransitionError,
    LoopControlReport,
    LoopDecision,
    LoopVerdict,
    StepResult,
    Trajectory,
    Transition,
    WatchdogLayer,
)
from agent_platform.loop_control.watchdogs import (
    DuplicateActionDetector,
    ToolLoopControl,
    action_fingerprint,
)

__all__ = [
    "SECONDS",
    "AgentState",
    "BudgetExhaustedError",
    "BudgetGuard",
    "CounterState",
    "Reservation",
    "Checkpoint",
    "CheckpointStore",
    "CheckStatus",
    "ControlCheck",
    "LoopControlReport",
    "DuplicateActionDetector",
    "IllegalTransitionError",
    "LoopController",
    "LoopDecision",
    "LoopVerdict",
    "ReactStep",
    "StateMachine",
    "StepResult",
    "ToolLoopControl",
    "Trajectory",
    "Transition",
    "WatchdogLayer",
    "action_fingerprint",
]
