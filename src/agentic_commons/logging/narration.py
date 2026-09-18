"""What a log record says when a person is reading it (LLD §2, §20).

Every log record carries two renderings of the same event. The file gets
the record: full payloads, exact arguments, the whole return value —
evidence, kept because the argument you need six hours later is the one
nobody thought to keep. The screen gets the narration: one line saying
what just happened, in words, with the two or three values that make it
meaningful.

They are deliberately not the same text. A screen filled with
`CALLED FiveStagePipeline.assemble(Task(query='where is my order',
session_id='tui-8f2a', actor='support-agent-1', kind='general',
metadata={}))` is a log file someone has pasted into a terminal: it
scrolls past faster than it can be read, and the one line that mattered
is indistinguishable from the four hundred that did not.

So this module decides what a person sees. The rules it applies:

- one line per call, at entry — the flow is the story, and a matching
  RETURNED line doubles the length of the story without advancing it;
- values that identify (an order id, an intent, a session) are shown;
  values that are *content* (an assembled window, a prompt, a memory
  fact, a list of chunks) are counted, not printed;
- anything still too long is elided to a width that keeps one event on
  one line.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# The attribute a `Narration` travels on. Named once, here, because both
# the logger that attaches it and the handler that reads it must agree
# and neither should have to know the other.
NARRATION = "narration"

NarrationKind = Literal["call", "return", "skip", "event"]


class Narration(BaseModel):
    """The human-facing half of one log record."""

    subsystem: str
    kind: NarrationKind = "event"
    # For a traced call: `CustomerAuthenticator.authenticate`.
    qualname: str = ""
    # For a plain event: the sentence the caller wrote.
    headline: str = ""
    arguments: dict[str, str] = Field(default_factory=dict)
    fields: dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""
    elapsed_ms: float = 0.0

    @property
    def component(self) -> str:
        """The class whose method this was, if it was a method."""
        return self.qualname.split(".")[0] if "." in self.qualname else ""

    @property
    def action(self) -> str:
        """The step, in words: `get_by_code` reads as "get by code"."""
        method = self.qualname.rsplit(".", maxsplit=1)[-1]
        return method.replace("_", " ") if method else self.headline
