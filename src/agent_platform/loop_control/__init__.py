"""Public API for loop control: one governor, one verdict, one state.

* ``types``        the contracts: `LoopVerdict`, `TerminalOutcome`, phases, layers
* ``state``        `LoopState` (checkpointable) and the phase machine
* ``budget``       named hard caps (reserve → commit / release), shared by workers
* ``stagnation``   exact, canonical and semantic repeats; cycles; the historical baseline
* ``convergence``  progress per step, and an agent's own convergence signal
* ``governor``     `LoopGovernor`: the single authority, and `route_on_verdict`
* ``factory``      building a governor in a composition root

This module is an import surface only; it contains no runtime logic.
"""

from agent_platform.loop_control.budget import (
    SECONDS,
    BudgetExhaustedError,
    BudgetGuard,
    CounterState,
    Reservation,
)
from agent_platform.loop_control.convergence import ConvergenceMonitor, ConvergenceSignal
from agent_platform.loop_control.factory import build_loop_governor
from agent_platform.loop_control.governor import (
    COST,
    STEPS,
    TOKENS,
    TOOL_CALLS,
    CallLimits,
    LoopBudget,
    LoopGovernor,
    route_on_verdict,
)
from agent_platform.loop_control.stagnation import (
    CycleDetector,
    FactStepHistory,
    HistoricalBaseline,
    SemanticTier,
    StepHistory,
    action_fingerprint,
    canonical_fingerprint,
)
from agent_platform.loop_control.state import ActionRecord, LoopState, LoopStateMachine
from agent_platform.loop_control.types import (
    RUN_STATUS,
    TOOL_STAGE,
    BudgetLine,
    CheckStatus,
    ControlCheck,
    IllegalTransitionError,
    LoopDecision,
    LoopPhase,
    LoopVerdict,
    TerminalOutcome,
    Transition,
    WatchdogLayer,
)

__all__ = [
    "COST",
    "RUN_STATUS",
    "SECONDS",
    "STEPS",
    "TOKENS",
    "TOOL_CALLS",
    "TOOL_STAGE",
    "ActionRecord",
    "BudgetExhaustedError",
    "BudgetGuard",
    "BudgetLine",
    "CallLimits",
    "CheckStatus",
    "ControlCheck",
    "ConvergenceMonitor",
    "ConvergenceSignal",
    "CounterState",
    "CycleDetector",
    "FactStepHistory",
    "HistoricalBaseline",
    "IllegalTransitionError",
    "LoopBudget",
    "LoopDecision",
    "LoopGovernor",
    "LoopPhase",
    "LoopState",
    "LoopStateMachine",
    "LoopVerdict",
    "Reservation",
    "SemanticTier",
    "StepHistory",
    "TerminalOutcome",
    "Transition",
    "WatchdogLayer",
    "action_fingerprint",
    "build_loop_governor",
    "canonical_fingerprint",
    "route_on_verdict",
]
