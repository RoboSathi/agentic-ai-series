"""How every role is prompted: fixed instructions, then one JSON payload.

Each role's prompt is its system instructions (a module in this package)
plus a user message that carries the role's inputs as one JSON object and
the schema its answer must satisfy. Keeping inputs structured, rather than
interpolated into prose, is what lets a test inspect *exactly* what a
role was shown — the check that the planner, reflector and synthesizer
are never handed raw page text reads these payloads.
"""

import json
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_platform.llm.types import Message

# The first line of every system prompt names the role, so a transcript
# (or the fixture model) can tell which role a request belongs to.
ROLE_HEADER = "ROLE: {role}"
PAYLOAD_OPEN = "INPUT (JSON):\n```json\n"
PAYLOAD_CLOSE = "\n```"
SCHEMA_INTRO = "\nAnswer with one JSON object and nothing else, matching this schema:\n"
# Every research-side role is told the run's date. Without it a model counts "the last 20 years"
# back from its training-era "now" (run-4f316d864e planned 2003-2023 on 2026-10-04).
TODAY_KEY = "today"
TODAY_RULE = (
    'Today\'s date is `today` in the input. Every relative time ("the last 20 years", "this '
    'season", "recent") counts back from it; never take the current date or year from your '
    "own knowledge."
)


def dated(payload: dict[str, Any], today: date) -> dict[str, Any]:
    """The payload with the run's date under `today` (and so the rule that goes with it)."""
    return {**payload, TODAY_KEY: today.isoformat()}


class RolePrompt(BaseModel):
    """A role's instructions plus its structured inputs."""

    model_config = ConfigDict(frozen=True)

    role: str
    instructions: str
    payload: dict[str, Any]

    def messages(self, schema: type[BaseModel]) -> list[Message]:
        rule = f"\n\n{TODAY_RULE}" if TODAY_KEY in self.payload else ""
        system = f"{ROLE_HEADER.format(role=self.role)}\n{self.instructions.strip()}{rule}"
        body = json.dumps(self.payload, indent=2, sort_keys=True, default=str)
        schema_text = json.dumps(schema.model_json_schema(), sort_keys=True)
        user = f"{PAYLOAD_OPEN}{body}{PAYLOAD_CLOSE}{SCHEMA_INTRO}{schema_text}"
        return [Message(role="system", content=system), Message(role="user", content=user)]

    def with_instruction(self, extra: str) -> "RolePrompt":
        """The same prompt with one more instruction appended (a guard's re-prompt)."""
        return self.model_copy(update={"instructions": f"{self.instructions.strip()}\n\n{extra}"})


def role_of(messages: list[Message]) -> str:
    """The role a request was built for (read back from its system message)."""
    first = messages[0].content.splitlines()[0] if messages and messages[0].content else ""
    prefix = ROLE_HEADER.format(role="")
    return first[len(prefix) :].strip() if first.startswith(prefix) else ""


def payload_of(messages: list[Message]) -> dict[str, Any]:
    """The JSON payload a request carried (read back from its user message)."""
    for message in messages:
        if message.role == "user" and PAYLOAD_OPEN in message.content:
            body = message.content.split(PAYLOAD_OPEN, 1)[1].split(PAYLOAD_CLOSE, 1)[0]
            loaded = json.loads(body)
            return loaded if isinstance(loaded, dict) else {}
    return {}


def instructions_of(messages: list[Message]) -> str:
    return messages[0].content if messages else ""
