"""`make queue <agent> [KIND=…] [ID=…] [ACTION=approve|deny ID=… COMMENT=…]`: `/queue`, from a terminal.

The same command and the same tables as the console, printed in the same
colours; an operator at a terminal sees every subject's items (there is no
console session to scope to), and the hint under the list says how to
decide from a terminal.
"""

import argparse
import asyncio
import sys

from agent_platform.commands.review import TERMINAL, QueueCommand
from agent_platform.commands.types import CommandError
from agent_platform.config import get_settings
from agent_platform.host.loader import load_spec
from agent_platform.presentation.terminal import print_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--agent", default="", help="the agent's name as make knows it")
    parser.add_argument("words", nargs="*", help="what /queue would be given")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    command = _command()
    command.surface, command.make_agent = TERMINAL, args.agent
    try:
        result = asyncio.run(command.run(args.words))
    except CommandError as exc:
        print(exc)
        return 1
    print_result(result)
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
