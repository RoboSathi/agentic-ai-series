"""Public API for deterministic agent control loops.

Implementation is intentionally split into modules:

* ``types`` — data contracts and state models.
* ``watchdogs`` — deterministic watchdog logic.
* ``protocols`` — orchestration interfaces.

This module is an import surface only; it contains no runtime logic.
"""

from agentic_commons.control_loop.budget import (
    SECONDS,
    BudgetExhaustedError,
    BudgetGuard,
    CounterState,
    Reservation,
)
from agentic_commons.control_loop.protocols import (
    CheckpointStore,
    LoopController,
    ReactStep,
    StateMachine,
)
from agentic_commons.control_loop.types import (
    AgentState,
    Checkpoint,
    CheckStatus,
    ControlCheck,
    ControlLoopReport,
    IllegalTransitionError,
    LoopDecision,
    LoopVerdict,
    StepResult,
    Trajectory,
    Transition,
    WatchdogLayer,
)
from agentic_commons.control_loop.watchdogs import (
    DuplicateActionDetector,
    ToolControlLoop,
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
    "ControlLoopReport",
    "DuplicateActionDetector",
    "IllegalTransitionError",
    "LoopController",
    "LoopDecision",
    "LoopVerdict",
    "ReactStep",
    "StateMachine",
    "StepResult",
    "ToolControlLoop",
    "Trajectory",
    "Transition",
    "WatchdogLayer",
    "action_fingerprint",
]
