"""The ScriptedUser: a deterministic person on the other side of the interview.

It answers whatever checklist item the agent last asked about, from a
table keyed by item id; waives the items it is told to waive; edits the
rubric as scripted; and approves. It reads where the conversation stands
from the agent's own session record — the same state a person reads off
the screen — and speaks only through `handle()`, so it drives exactly the
path the console does.
"""

from pydantic import BaseModel, Field

from research_agent.config.settings import CommandNames
from research_agent.graph.agent import ResearchAgent, TurnResult

_SCOPING_PHASES = ("SCOPING", "RUBRIC_REVIEW")


class UserScript(BaseModel):
    """What the user will say, by checklist item."""

    first_message: str
    answers: dict[str, str] = Field(default_factory=dict)
    waive: list[str] = Field(default_factory=list)
    rubric_edits: list[str] = Field(default_factory=list)
    # Stop answering after this many turns and say /done (tests early closing).
    done_after_turns: int | None = None


class ScriptedUser:
    """Plays one UserScript against a research agent."""

    def __init__(self, script: UserScript, commands: CommandNames, max_turns: int) -> None:
        self._script = script
        self._names = commands
        self._max_turns = max_turns
        self._edits = list(script.rubric_edits)
        self.transcript: list[tuple[str, str]] = []

    async def converse(self, agent: ResearchAgent, session_id: str) -> TurnResult:
        """The whole conversation, first message to delivered report."""
        reply = await self._say(agent, session_id, self._script.first_message)
        for _ in range(self._max_turns):
            if reply.phase not in _SCOPING_PHASES:
                return reply
            reply = await self._say(agent, session_id, self.next_message(agent, session_id, reply))
        return reply

    def next_message(self, agent: ResearchAgent, session_id: str, reply: TurnResult) -> str:
        if reply.phase == "RUBRIC_REVIEW":
            return self._edits.pop(0) if self._edits else f"/{self._names.approve}"
        scope = agent.session(session_id).scope
        asked = scope.asked_item_id if scope is not None else ""
        turns = scope.turn if scope is not None else 0
        limit = self._script.done_after_turns
        if (limit is not None and turns >= limit) or not asked:
            return f"/{self._names.done}"
        if asked in self._script.waive:
            return f"/{self._names.waive} {asked}"
        return self._script.answers.get(asked, f"/{self._names.done}")

    async def _say(self, agent: ResearchAgent, session_id: str, message: str) -> TurnResult:
        reply = await agent.handle(message, session_id)
        self.transcript.append((message, reply.response))
        return reply
