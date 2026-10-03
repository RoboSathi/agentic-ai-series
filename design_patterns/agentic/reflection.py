"""Reflection — critique a finished attempt and revise it (E06).

Shinn et al.'s Reflexion (2023): verbal self-critique, kept and reused,
standing in for gradient updates when the weights cannot change.

The failure mode is the reason this pattern needs a warning label rather
than a recommendation. An agent asked to reflect on a wrong answer will
frequently produce a *more confident* wrong answer: nothing in the loop
is external to the model, so there is nothing to catch a mistake the
model cannot see. Self-critique raises fluency and certainty, which is
not the same as raising accuracy.

Which is why `EvaluatorOptimizerPattern` exists next door with an
independent scorer, and why M11 fixes this properly with grounding.
Reflection is worth using when an external signal is available to
critique *against* — and worth distrusting when it is not.
"""

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

_CRITIQUE = (
    "Critique the answer below against the task. Name concrete problems "
    "only — if it is correct, reply exactly: NO ISSUES.\n\n"
    "Task: {task}\n\nAnswer: {answer}"
)
_REVISE = (
    "Revise the answer to address the critique.\n\n"
    "Task: {task}\n\nAnswer: {answer}\n\nCritique: {critique}"
)

# What the critic says when it finds nothing. Matched exactly, because a
# loose match ("no problems here, although…") would stop a loop that
# should have continued.
_NO_ISSUES = "NO ISSUES"


class ReflectionPattern:
    """Generate, critique, revise — up to a bounded number of rounds."""

    name: str = "reflection"
    family: PatternFamily = "AG"

    def __init__(self, llm: LLMClient, max_rounds: int = 2) -> None:
        self._llm = llm
        self._max_rounds = max_rounds
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Produce an answer, then critique and revise it."""
        first = await self._llm.complete(
            CompletionRequest(messages=[Message(role="user", content=task)])
        )
        answer = first.content
        steps = [
            Step(kind="draft", detail="initial", output=answer, tokens=first.usage.total_tokens)
        ]

        for round_index in range(self._max_rounds):
            critique = await self._ask(_CRITIQUE.format(task=task, answer=answer))
            steps.append(Step(kind="critique", detail=f"round {round_index + 1}", output=critique))
            if _NO_ISSUES in critique.upper():
                self._logger.debug("Reflection settled", rounds=round_index + 1)
                break
            answer = await self._ask(_REVISE.format(task=task, answer=answer, critique=critique))
            steps.append(Step(kind="revise", detail=f"round {round_index + 1}", output=answer))

        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=answer,
            steps=steps,
            metadata={
                "rounds": sum(step.kind == "critique" for step in steps),
                # Recorded so a caller can weigh the answer accordingly:
                # nothing outside the model checked any of this.
                "externally_grounded": False,
            },
        )

    async def _ask(self, prompt: str) -> str:
        result = await self._llm.complete(
            CompletionRequest(messages=[Message(role="user", content=prompt)])
        )
        return result.content
