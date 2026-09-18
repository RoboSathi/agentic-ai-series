"""Evaluator-optimizer — a generator paired with an independent scorer (E06).

Generate-test-and-debug, applied to language agents. The distinction from
reflection is the whole point: the scorer must be *independent* of the
generator. A generator grading its own work is reflection wearing a hat,
and it converges on confident rather than correct.

So the scorer here is `evals.NollmScoringCascade` — the same cascade the
rest of the system scores with, which means an improvement measured here
is measured the same way as everywhere else, and `nollm_scorer_ratio`
still means something.

Reach for it when quality is measurable by something other than the thing
producing it. When it is not, no amount of looping helps.
"""

from agentic_commons.evals import Claim, NollmScoringCascadeImpl
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest, Message
from agentic_commons.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

_IMPROVE = (
    "Your previous answer scored poorly against the criteria. Produce a "
    "better one.\n\nTask: {task}\n\nPrevious: {previous}\n\nCriteria: {criteria}"
)


class EvaluatorOptimizerPattern:
    """Generates candidates until an independent scorer is satisfied."""

    name: str = "evaluator_optimizer"
    family: PatternFamily = "AG"

    def __init__(
        self,
        llm: LLMClient,
        cascade: NollmScoringCascadeImpl,
        target_score: float = 0.8,
        max_rounds: int = 3,
    ) -> None:
        self._llm = llm
        self._cascade = cascade
        self._target = target_score
        self._max_rounds = max_rounds
        self._logger = get_logger("design_patterns")

    async def run(self, task: str, criteria: list[str] | None = None) -> PatternRun:
        """Generate, score, and regenerate until the bar is met or rounds run out.

        The best candidate is kept rather than the last one: a later
        revision is not automatically better, and returning whatever came
        last would sometimes discard the answer that actually passed.
        """
        evidence = criteria or []
        steps: list[Step] = []
        best, best_score = "", -1.0

        for round_index in range(self._max_rounds):
            prompt = (
                task
                if round_index == 0
                else _IMPROVE.format(task=task, previous=best, criteria="; ".join(evidence))
            )
            candidate = await self._generate(prompt)
            verdict = await self._cascade.resolve(Claim(text=candidate, evidence=evidence))
            steps.append(
                Step(
                    kind="candidate",
                    detail=f"round {round_index + 1} scored {verdict.score:.2f} ({verdict.tier})",
                    output=candidate,
                )
            )
            if verdict.score > best_score:
                best, best_score = candidate, verdict.score
            if verdict.score >= self._target:
                self._logger.debug("Target met", rounds=round_index + 1)
                break

        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=best,
            steps=steps,
            succeeded=best_score >= self._target,
            metadata={
                "best_score": best_score,
                "target": self._target,
                "rounds": len(steps),
                "nollm_scorer_ratio": self._cascade.nollm_scorer_ratio,
            },
        )

    async def _generate(self, prompt: str) -> str:
        result = await self._llm.complete(
            CompletionRequest(messages=[Message(role="user", content=prompt)])
        )
        return result.content
