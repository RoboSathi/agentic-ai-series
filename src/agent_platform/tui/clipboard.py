"""Copying text out of the TUI (LLD §20).

OSC 52 is the right primitive: the terminal itself performs the copy, so
it works over SSH with no clipboard daemon on the remote host. Its limit
is size — many terminals silently truncate or ignore a large sequence,
and a copy that silently does nothing is worse than one that refuses.

So the file export is not a fallback; it always runs. The clipboard
attempt is the bonus, and the person is told the path either way.
"""

import base64
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger, log_directory
from agent_platform.observability.types import utc_now

# Terminals commonly cap an OSC 52 payload around 100KB; stay well under.
_MAX_OSC52_BYTES = 64_000


@runtime_checkable
class ClipboardService(Protocol):
    """Puts text where the person can paste it."""

    def copy(self, text: str) -> str: ...

    def copy_text(self, text: str) -> bool: ...


class Osc52ClipboardService:
    """Writes an export file, and additionally tries the terminal clipboard."""

    def __init__(self, export_dir: Path | None = None) -> None:
        settings = get_settings()
        self._method = settings.tui.copy_method
        self._commands = settings.tui.system_clipboard_commands
        self._export_dir = export_dir or log_directory()
        self._logger = get_logger("tui")

    def copy(self, text: str) -> str:
        """Export the text and return the path it was written to."""
        path = self._export(text)
        if self._method == "osc52":
            self._to_terminal(text)
        return str(path)

    def copy_text(self, text: str) -> bool:
        """Put a selection on the system clipboard; no export file.

        A mouse selection is small and deliberate, so a file per drag would
        only litter the logs. The system command is what makes ⌘V work in
        terminals that ignore OSC 52; OSC 52 is still sent for the ones
        (and the SSH sessions) where only the terminal can reach a clipboard.
        """
        if self._method == "osc52":
            self._to_terminal(text)
        return self._to_system(text)

    def _to_system(self, text: str) -> bool:
        """The first installed clipboard command, fed the text on stdin."""
        for command in self._commands:
            if not command or shutil.which(command[0]) is None:
                continue
            try:
                subprocess.run(command, input=text.encode("utf-8"), check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                self._logger.debug("Clipboard command failed", command=command[0], error=str(exc))
                continue
            return True
        return False

    def _export(self, text: str) -> Path:
        """Always write a file — the copy that cannot silently fail."""
        self._export_dir.mkdir(parents=True, exist_ok=True)
        path = self._export_dir / f"tui_export_{utc_now():%Y%m%d_%H%M%S}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def _to_terminal(self, text: str) -> None:
        """Best-effort OSC 52, skipped when the payload is too large."""
        payload = text.encode("utf-8")
        if len(payload) > _MAX_OSC52_BYTES:
            self._logger.debug("Payload too large for OSC 52; file export only")
            return
        try:
            encoded = base64.b64encode(payload).decode("ascii")
            sys.stdout.write(f"\033]52;c;{encoded}\a")
            sys.stdout.flush()
        except (OSError, ValueError) as exc:
            self._logger.debug("OSC 52 copy failed; file export still written", error=str(exc))
