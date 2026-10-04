"""Console slash commands (LLD §20).

An operator asking the harness about itself: `/audit`, `/config`,
`/harness`, `/eval`. None of them reaches a model and none appears in the
customer's transcript — inspecting the system should not consume the
customer's token budget, and a command answered by a model could be
wrong about what the system contains.

Add a *platform* command by writing a class with `name`, `summary`,
`usage` and an async `run`, then listing it in
`registry._default_commands`. A command that reads one product's tables
is not a platform command: it belongs to that agent and reaches the
console through its `AgentHost`.
"""

from agent_platform.commands.data import MemoryCommand
from agent_platform.commands.evals import EvalCommand
from agent_platform.commands.harness import HarnessCommand
from agent_platform.commands.inspection import (
    AuditCommand,
    ConfigCommand,
)
from agent_platform.commands.knowledge import KnowledgeBaseCommand
from agent_platform.commands.registry import PREFIX, CommandRegistry
from agent_platform.commands.runs import AbortCommand, CancelCommand, ResumeCommand
from agent_platform.commands.scope import ConsoleScope, ScopedCommand
from agent_platform.commands.system import (
    ClearCommand,
    HelpCommand,
)
from agent_platform.commands.types import (
    Command,
    CommandError,
    CommandResult,
    CommandRow,
    ConsoleEffect,
)

__all__ = [
    "ScopedCommand",
    "PREFIX",
    "ResumeCommand",
    "AbortCommand",
    "AuditCommand",
    "CancelCommand",
    "ClearCommand",
    "ConsoleScope",
    "Command",
    "CommandError",
    "CommandRegistry",
    "CommandResult",
    "ConsoleEffect",
    "CommandRow",
    "ConfigCommand",
    "EvalCommand",
    "HarnessCommand",
    "HelpCommand",
    "KnowledgeBaseCommand",
    "MemoryCommand",
]
