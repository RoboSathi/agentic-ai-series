"""A transcript as DeepEval's `ConversationalTestCase`.

Every turn the customer took and every reply the agent gave, in order;
each reply carries the tools the agent called for it and the evidence it
could ground on (`retrieval_context`), so a faithfulness metric checks the
reply against what the agent was actually shown. The case's scenario and
the outcome the written policy requires travel with it, so a G-Eval judges
against the expected outcome rather than its own idea of a good answer.
"""

import json
from typing import Any

from agent_platform.evals.core.contracts import (
    ConversationFrame,
    GoldenCase,
    ToolCallRecord,
    Transcript,
)

_USER, _ASSISTANT = "user", "assistant"


def conversation(
    deepeval: Any,
    case: GoldenCase,
    transcript: Transcript,
    frame: ConversationFrame,
    evidence_cap: int,
) -> Any:
    """The case and its transcript, as one multi-turn test case."""
    test_case = deepeval.test_case
    evidence = capped(transcript.evidence, evidence_cap)
    turns: list[Any] = []
    for index, message in enumerate(case.turns):
        turns.append(test_case.Turn(role=_USER, content=message))
        if index < len(transcript.replies):
            turns.append(
                test_case.Turn(
                    role=_ASSISTANT,
                    content=transcript.replies[index],
                    tools_called=_tools(test_case, transcript.tool_calls, index) or None,
                    retrieval_context=evidence or None,
                )
            )
    return test_case.ConversationalTestCase(
        name=case.case_id,
        turns=turns,
        scenario=frame.scenario or None,
        expected_outcome=frame.expected_outcome or None,
        chatbot_role=frame.chatbot_role or None,
        tags=case.tags or None,
    )


def capped(evidence: list[str], cap: int) -> list[str]:
    """The evidence a judge reads, oldest first, up to `cap` characters (0 = all)."""
    if cap <= 0:
        return evidence
    kept, used = [], 0
    for item in evidence:
        if used + len(item) > cap:
            break
        kept.append(item)
        used += len(item)
    return kept


def _tools(test_case: Any, calls: list[ToolCallRecord], turn: int) -> list[Any]:
    return [
        test_case.ToolCall(
            name=call.name,
            input_parameters=call.arguments,
            output=_output(call),
        )
        for call in calls
        if call.turn == turn
    ]


def _output(call: ToolCallRecord) -> dict[str, Any]:
    """What became of the call: the gatekeeper's ruling and the tool's result."""
    return {
        "status": call.status,
        "decision": call.decision,
        "standing": call.standing,
        "reason_code": call.reason_code,
        "result": _parsed(call.output),
    }


def _parsed(output: str) -> Any:
    try:
        return json.loads(output) if output else None
    except ValueError:
        return output
