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

from agentic_commons.commands.data import MemoryCommand
from agentic_commons.commands.evals import EvalCommand
from agentic_commons.commands.harness import HarnessCommand
from agentic_commons.commands.inspection import (
    AuditCommand,
    ConfigCommand,
)
from agentic_commons.commands.knowledge import KnowledgeBaseCommand
from agentic_commons.commands.registry import PREFIX, CommandRegistry
from agentic_commons.commands.system import (
    ClearCommand,
    HelpCommand,
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
    "ClearCommand",
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
