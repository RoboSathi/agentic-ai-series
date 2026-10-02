"""The anti-sycophancy guard: the interviewer must push back, at least every K turns.

Deterministic, on purpose. "Did this turn challenge anything?" is read
off the structured turn — a non-empty `challenges` list, or a follow-up
on the item just answered — not judged by another model. When K turns
pass without one, the turn is re-prompted *once* with an explicit
instruction to challenge, and the violation is an audit event.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from research_agent.prompts.interviewer import InterviewTurn


class ChallengeVerdict(BaseModel):
    violated: bool
    turns_without_challenge: int


class ChallengeGuard:
    """Counts turns since the interviewer last challenged or followed up."""

    def __init__(self, every_k_turns: int) -> None:
        self._k = every_k_turns

    @staticmethod
    def warranted(answer: str, vague_max_words: int, absolute_words: Sequence[str]) -> bool:
        """Whether an answer calls for pushback: it is vague (a few words) or absolute ("best").

        Deterministic, like the rest of the guard. An answer that is neither needs no challenge,
        and forcing one (a second model call) only slows the interview down.
        """
        words = answer.casefold().split()
        absolute = {word.casefold() for word in absolute_words}
        return 0 < len(words) <= vague_max_words or bool(absolute & set(words))

    @staticmethod
    def challenged(turn: InterviewTurn, asked_item_id: str) -> bool:
        """A challenge, or a follow-up question on the item just answered."""
        follow_up = bool(asked_item_id) and turn.next_item_id == asked_item_id
        return bool(turn.challenges) or follow_up

    def check(self, turn: InterviewTurn, asked_item_id: str, since_last: int) -> ChallengeVerdict:
        """Whether this turn breaks the every-K rule, and the new count."""
        if self.challenged(turn, asked_item_id):
            return ChallengeVerdict(violated=False, turns_without_challenge=0)
        count = since_last + 1
        return ChallengeVerdict(violated=count >= self._k, turns_without_challenge=count)
