"""Console slash commands (LLD §20).

An operator asking the harness about itself: `/audit`, `/config`,
`/orders`, `/eval`. None of them reaches a model and none appears in the
customer's transcript — inspecting the system should not consume the
customer's token budget, and a command answered by a model could be
wrong about what the system contains.

Add a command by writing a class with `name`, `summary`, `usage` and an
async `run`, then listing it in `registry._default_commands`.
"""

from agentic_commons.commands.data import MemoryCommand, OrdersCommand, UsersCommand
from agentic_commons.commands.harness import (
    CascadeCommand,
    EvalCommand,
    GatekeeperCommand,
    HarnessCommand,
    PatternsCommand,
    ToolsCommand,
)
from agentic_commons.commands.inspection import (
    AuditCommand,
    ConfigCommand,
    ModelCommand,
    QueueCommand,
    SloCommand,
)
from agentic_commons.commands.registry import PREFIX, CommandRegistry
from agentic_commons.commands.system import (
    ClearCommand,
    HelpCommand,
    LogsCommand,
    SessionCommand,
)
from agentic_commons.commands.types import (
    Command,
    CommandError,
    CommandResult,
    CommandRow,
    ConsoleEffect,
)

__all__ = [
    "PREFIX",
    "AuditCommand",
    "CascadeCommand",
    "ClearCommand",
    "Command",
    "CommandError",
    "CommandRegistry",
    "CommandResult",
    "ConsoleEffect",
    "CommandRow",
    "ConfigCommand",
    "ModelCommand",
    "EvalCommand",
    "GatekeeperCommand",
    "HarnessCommand",
    "HelpCommand",
    "LogsCommand",
    "MemoryCommand",
    "OrdersCommand",
    "PatternsCommand",
    "QueueCommand",
    "SessionCommand",
    "SloCommand",
    "ToolsCommand",
    "UsersCommand",
]
