"""`make dlq <agent> [STATUS=…|ID=…] [ACTION=… ID=… NOTE=…]`: `/dlq`, from the terminal.

The same command and the same lines as the console; only the printing is here.
"""

import asyncio
import sys

from agent_platform.commands.quarantine import DeadLetterCommand
from agent_platform.commands.types import CommandError


def main(argv: list[str] | None = None) -> int:
    try:
        result = asyncio.run(
            DeadLetterCommand().run(list(argv if argv is not None else sys.argv[1:]))
        )
    except CommandError as exc:
        print(exc)
        return 1
    for line in [result.title, *result.lines]:
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
