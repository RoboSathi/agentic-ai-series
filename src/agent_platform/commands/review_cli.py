"""`make queue <agent> [KIND=…] [ID=…] [ACTION=approve|deny ID=… COMMENT=…]`: `/queue`, from a terminal.

The same command and the same lines as the console; an operator at a
terminal sees every subject's items (there is no console session to scope
to). Only the printing is here.
"""

import asyncio
import sys

from agent_platform.commands.review import QueueCommand
from agent_platform.commands.types import CommandError
from agent_platform.config import get_settings
from agent_platform.host.loader import load_spec


def main(argv: list[str] | None = None) -> int:
    command = _command()
    try:
        result = asyncio.run(command.run(list(argv if argv is not None else sys.argv[1:])))
    except CommandError as exc:
        print(exc)
        return 1
    for line in [result.title, *result.lines]:
        if line:
            print(line)
    return 0


def _command() -> QueueCommand:
    """The agent's own `/queue` (with its kinds) when its host registers one, else the platform's."""
    spec = get_settings().agent.host
    host = load_spec(spec) if spec else None
    for command in host.commands() if host is not None else ():
        if isinstance(command, QueueCommand):
            return command
    return QueueCommand()


if __name__ == "__main__":
    sys.exit(main())
