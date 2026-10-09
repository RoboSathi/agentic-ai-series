"""Serving over stdio: stdout *is* the protocol, so nothing else may write to it.

A server's `__main__` calls `silence_console()` before anything loads settings;
everything is still logged to the files under `logging.dir`.
"""

import os

# The console handler's switch, as its environment variable (`logging.console_enabled`).
_CONSOLE_SWITCH = "LOGGING__CONSOLE_ENABLED"


def silence_console() -> None:
    """Keep console logging off stdout, before settings are first read."""
    os.environ[_CONSOLE_SWITCH] = "false"


__all__ = ["silence_console"]
