"""The scoping graph's nodes.

Each node does one thing and reports it: the research kind, each
interview turn (with the checklist it left behind), a waiver, the rubric,
the frozen contract. What the model proposes is filtered through code —
the termination policy and the challenge guard — before it becomes state.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.config.settings import CommandNames, MessageSettings, ScopingSettings
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import GENERAL, KindVerdict, ScopeBrief, UnknownReason
from research_agent.graph import cards
from research_agent.graph.state import ScopingState
from research_agent.llm.structured import StructuredCaller
from research_agent.observability.telemetry import RunTelemetry
from research_agent.prompts import interviewer as interviewer_prompt
from research_agent.rubric.editor import RubricEditError, RubricEditor, render
from research_agent.rubric.freeze import freeze_contract
from research_agent.rubric.generator import RubricGenerator
from research_agent.scoping import checklist
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.challenge_check import ChallengeGuard
from research_agent.scoping.kind import KindClassifier
from research_agent.scoping.termination import TerminationPolicy

# A word of an answer, as the vague-answer rule reads it.
_WORD = re.compile(r"[^\W\d_]+")

Update = dict[str, Any]


@dataclass
class ScopingParts:
    """The collaborators scoping is composed of; optional ones are features."""

    # "Now" for the contract (injectable, so a test can fix the date).
    clock: Callable[[], datetime]
    classifier: KindClassifier
    caller: StructuredCaller
    checklists: Path
    termination: TerminationPolicy
    briefs: BriefBuilder
    telemetry: RunTelemetry
    settings: ScopingSettings
    messages: MessageSettings
    commands: CommandNames
    contract_prefix: str
    challenge: ChallengeGuard | None = None
    generator: RubricGenerator | None = None
    editor: RubricEditor | None = None


class ScopingNodes:
    """Research kind, interview, waivers, rubric review and freezing, as graph nodes."""

    def __init__(self, parts: ScopingParts) -> None:
        self._p = parts
        self._affirmations = [
            re.compile(pattern, re.IGNORECASE) for pattern in parts.settings.affirmations
        ]

    # -- research kind ------------------------------------------------------------------

    async def kind(self, state: ScopingState) -> Update:
        """Which kind of research the question asks for — or a refusal."""
        verdict = await self._p.classifier.classify(state.first_message)
        self._report_kind(verdict)
        if verdict.decision == "refuse" or verdict.archetype is None:
            return {"phase": "REFUSED", "reply": self._p.messages.refuse}
        items = checklist.load_checklist(self._p.checklists, verdict.archetype)
        return {
            "phase": "INTERVIEW",
            "archetype": verdict.archetype,
            "items": checklist.prefill_window(items, verdict.recent_days),
            "topic": verdict.topic or state.first_message,
            "recent_days": verdict.recent_days,
        }

    def _report_kind(self, verdict: KindVerdict) -> None:
        heading, sections = cards.kind_card(verdict)
        refused = verdict.decision == "refuse"
        self._p.telemetry.card(
            "research_kind", heading, sections, status="blocked" if refused else "completed"
        )
        self._p.telemetry.audit(
            "scope.kind",
            "deny" if refused else "allow",
            verdict.reason or verdict.decision,
            archetype=verdict.archetype,
            kind_decision=verdict.decision,
            classifier=verdict.source,
            classifier_model=verdict.source_model,
        )

    @staticmethod
    def after_kind(state: ScopingState) -> str:
        return "__end__" if state.phase == "REFUSED" else "interview_turn"

    # -- interview -------------------------------------------------------------------

    async def interview(self, state: ScopingState) -> Update:
        turn_number = state.turn + 1
        prompt = interviewer_prompt.build(
            archetype=state.archetype or "",
            request=state.first_message,
            topic=state.topic,
            items=state.items,
            asked_item_id=state.asked_item_id,
            answer=state.user_input,
            turn=turn_number,
            today=self._p.clock().date(),
            rubric_shape=self._p.settings.rubric_shapes.get(state.archetype or "", ""),
        )
        turn = await self._p.caller.call(prompt, interviewer_prompt.InterviewTurn)
        turn, since = await self._guarded(prompt, turn, state)
        items = self._apply(state.items, turn)
        decision = self._p.termination.decide(turn, items, turn_number)
        reply = "\n".join(
            part for part in (state.notice, self._reply(turn, decision.outcome, items)) if part
        )
        self._report_turn(turn_number, turn, items, decision.outcome, decision.overruled)
        return {
            "items": items,
            "turn": turn_number,
            # Only a question is "asked": a proposal to finish waits for /done, not an item.
            "asked_item_id": turn.next_item_id if decision.outcome == "ask" else "",
            "proposed": decision.outcome == "propose_done",
            "since_challenge": since,
            "draft_plan": turn.draft_plan,
            "reply": reply,
            "user_input": "",
            "notice": "",
        }

    async def _guarded(
        self, prompt: Any, turn: interviewer_prompt.InterviewTurn, state: ScopingState
    ) -> tuple[interviewer_prompt.InterviewTurn, int]:
        """The anti-sycophancy guard: re-prompt once when K turns passed unchallenged."""
        if self._p.challenge is None:
            return turn, 0
        if self._menu_answer(state) or not self._needs_challenge(state.user_input):
            # A picked choice, or an answer that is specific and measured, calls for no pushback:
            # nothing is forced and nothing is counted (unless the model challenged anyway).
            challenged = ChallengeGuard.challenged(turn, state.asked_item_id)
            return turn, 0 if challenged else state.since_challenge
        verdict = self._p.challenge.check(turn, state.asked_item_id, state.since_challenge)
        if not verdict.violated:
            return turn, verdict.turns_without_challenge
        self._p.telemetry.audit(
            "scope.challenge_guard",
            "escalate",
            "interviewer went K turns without a challenge",
            turns=verdict.turns_without_challenge,
        )
        heading, sections = cards.guard_card(verdict.turns_without_challenge)
        self._p.telemetry.card("interview_turn", heading, sections, status="warning")
        retried = await self._p.caller.call(
            prompt.with_instruction(interviewer_prompt.CHALLENGE_INSTRUCTION),
            interviewer_prompt.InterviewTurn,
        )
        return retried, 0 if ChallengeGuard.challenged(
            retried, state.asked_item_id
        ) else verdict.turns_without_challenge

    def _needs_challenge(self, answer: str) -> bool:
        settings = self._p.settings
        return ChallengeGuard.warranted(
            answer, settings.vague_answer_max_words, settings.absolute_words
        )

    def _names_nothing(self, target: Any, value: str) -> bool:
        """Whether an answer to a subject item names nothing to research ("all", "everything")."""
        settings = self._p.settings
        if target.brief_field not in settings.subject_fields:
            return False
        words = _WORD.findall(value.casefold())
        vague = {word.casefold() for word in settings.vague_subject_words}
        return bool(words) and all(word in vague for word in words)

    @staticmethod
    def _menu_answer(state: ScopingState) -> bool:
        asked = checklist.item(state.items, state.asked_item_id)
        return asked is not None and bool(asked.choices)

    def _apply(self, items: list[Any], turn: interviewer_prompt.InterviewTurn) -> list[Any]:
        for update in turn.updated_items:
            value = update.value.strip()
            target = checklist.item(items, update.item_id)
            if not value or target is None:
                continue
            if target.status != "delegated" and self._names_nothing(target, value):
                # "all" is no list of concepts to research: ask again, then propose them.
                items = checklist.vague_answer(
                    items, target.item_id, self._p.settings.vague_answers_before_delegating
                )
                continue
            derived = target.status == "delegated"
            items = checklist.covered(items, target.item_id, value, derived=derived)
        for skip in turn.skip_items:
            target = checklist.item(items, skip.item_id)
            if (
                target is not None
                and not target.required
                and target.skippable
                and target.status == "open"
            ):
                items = checklist.with_status(
                    items, skip.item_id, "covered", self._p.settings.skipped_answer
                )
        return items

    def _reply(self, turn: interviewer_prompt.InterviewTurn, outcome: str, items: list[Any]) -> str:
        challenges = [f"{self._p.messages.challenge_prefix}{c}" for c in turn.challenges]
        if outcome == "propose_done":
            steps = turn.draft_plan or self._fallback_plan(items)
            plan = "\n".join(f"  - {step}" for step in steps)
            proposal = self._p.messages.propose_done.format(plan=plan)
            return "\n".join([*challenges, proposal, "", self._p.messages.next_step_proposed])
        if outcome == "turn_cap":
            open_ids = ", ".join(
                item.item_id for item in checklist.open_items(items, required_only=True)
            )
            return "\n".join([*challenges, self._p.messages.turn_cap.format(open=open_ids)])
        return "\n".join([*challenges, turn.next_question])

    def _fallback_plan(self, items: list[Any]) -> list[str]:
        """A plan from what the interview established, when the model proposed none."""
        criteria = checklist.item(items, self._p.settings.plan_item)
        values = criteria.value if criteria is not None and criteria.value else ""
        parts = [p.strip() for p in re.split(self._p.settings.answer_split_pattern, values)]
        return [self._p.messages.plan_step.format(part=part) for part in parts if part]

    def _report_turn(
        self, number: int, turn: Any, items: list[Any], outcome: str, overruled: bool
    ) -> None:
        heading, sections = cards.interview_card(number, turn, outcome, overruled)
        self._p.telemetry.card("interview_turn", heading, sections)

    # -- waivers, closing ----------------------------------------------------------------

    async def waive(self, state: ScopingState) -> Update:
        target = (
            state.user_input.split(maxsplit=1)[1].strip() if " " in state.user_input.strip() else ""
        )
        items = checklist.waive(state.items, target, self._p.settings.waive_all_keyword)
        waived = [i.item_id for i in items if i.status == "waived"]
        heading, sections = cards.waiver_card(waived)
        self._p.telemetry.card("coverage_update", heading, sections)
        return {
            "items": items,
            "notice": self._p.messages.waived.format(items=", ".join(waived) or target),
            "user_input": "",
            "asked_item_id": "",
        }

    async def close(self, state: ScopingState) -> Update:
        reason: UnknownReason = (
            "turn_cap" if state.turn >= self._p.settings.max_turns else "closed_early"
        )
        unknowns = checklist.known_unknowns(state.items, reason)
        brief = self._p.briefs.build(
            state.archetype or GENERAL,
            state.topic,
            state.items,
            unknowns,
            today=self._p.clock().date(),
        )
        self._p.telemetry.audit(
            "scope.closed",
            "allow",
            reason,
            archetype=brief.archetype,
            known_unknowns=[u.item_id for u in unknowns],
        )
        return {"brief": brief, "phase": "RUBRIC_REVIEW", "user_input": "", "proposed": False}

    # -- rubric ---------------------------------------------------------------------

    async def generate_rubric(self, state: ScopingState) -> Update:
        assert state.brief is not None
        rubric = (
            self._p.generator.generate(state.brief) if self._p.generator is not None else Rubric()
        )
        heading, sections = cards.rubric_card(rubric, state.brief)
        self._p.telemetry.card("rubric_generate", heading, sections)
        return {"rubric": rubric, "reply": self._rubric_reply(rubric)}

    async def brief_ready(self, state: ScopingState) -> Update:
        """With the rubric withheld there is nothing to review but the Brief."""
        return {"rubric": Rubric(), "reply": self._p.messages.brief_ready}

    async def edit_rubric(self, state: ScopingState) -> Update:
        assert state.rubric is not None
        try:
            rubric = self._edit(state.rubric, state.user_input.split()[1:])
        except RubricEditError as exc:
            return {"reply": self._p.messages.rubric_error.format(error=exc), "user_input": ""}
        heading, sections = cards.rubric_card(rubric, state.brief, edited=state.user_input)
        self._p.telemetry.card("rubric_generate", heading, sections)
        return {
            "rubric": rubric,
            "reply": self._rubric_reply(rubric),
            "user_input": "",
        }

    def _rubric_reply(self, rubric: Rubric) -> str:
        """The criteria to review, then — set apart — what to do next."""
        return f"{render(rubric, self._p.messages)}\n\n{self._p.messages.next_step_rubric}"

    def _edit(self, rubric: Rubric, args: list[str]) -> Rubric:
        names, editor = self._p.commands, self._p.editor
        if editor is None or not args:
            return rubric
        verb, rest = args[0], args[1:]
        if verb == names.rubric_add:
            return editor.add(rubric, " ".join(rest))
        if verb == names.rubric_drop and rest:
            return editor.drop(rubric, rest[0])
        if verb == names.rubric_set and rest:
            return editor.set(rubric, rest[0], rest[1:])
        raise RubricEditError(self._p.messages.rubric_hint)

    async def freeze(self, state: ScopingState) -> Update:
        assert state.brief is not None and state.rubric is not None
        contract = self._freeze(state.brief, state.rubric, "user approved the research contract")
        reply = self._p.messages.approved.format(
            contract_id=contract.contract_id, hash=contract.contract_hash[:12]
        )
        return {"contract": contract, "phase": "APPROVED", "reply": reply}

    def _freeze(self, brief: ScopeBrief, rubric: Rubric, why: str) -> Contract:
        """Freeze, hash, and record the hash where the end-of-run check will find it."""
        contract = freeze_contract(brief, rubric, self._p.contract_prefix, self._p.clock())
        self._p.telemetry.audit(
            "contract.frozen",
            "allow",
            why,
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
        )
        return contract

    async def say(self, state: ScopingState) -> Update:
        hint = (
            self._p.messages.rubric_hint
            if state.phase == "RUBRIC_REVIEW"
            else self._p.messages.finish_scoping_first
        )
        return {"reply": hint, "user_input": ""}

    # -- routing ----------------------------------------------------------------------

    def route_input(self, state: ScopingState) -> str:
        verb = _verb(state.user_input)
        if verb is None and self._affirmed(state.user_input):
            # "yes" to what was just proposed: the plan closes the interview, the criteria
            # approve. Anything else is an answer, or a change to the plan.
            if state.phase == "RUBRIC_REVIEW":
                return "rubric_frozen"
            if state.proposed:
                return "close_scoping"
        if state.phase == "RUBRIC_REVIEW":
            return self._review_route(verb)
        return self._interview_route(verb)

    def _affirmed(self, reply: str) -> bool:
        text = reply.strip()
        return any(pattern.fullmatch(text) for pattern in self._affirmations)

    def _interview_route(self, verb: str | None) -> str:
        names = self._p.commands
        if verb is None:
            return "interview_turn"
        if verb == names.done:
            return "close_scoping"
        return "coverage_update" if verb == names.waive else "say"

    def _review_route(self, verb: str | None) -> str:
        names = self._p.commands
        if verb == names.approve:
            return "rubric_frozen"
        return "rubric_edit" if verb == names.rubric and self._p.editor is not None else "say"


def _verb(line: str) -> str | None:
    """The slash command a line starts with, or None for an ordinary answer."""
    stripped = line.strip()
    return (
        stripped[1:].split(maxsplit=1)[0].casefold()
        if stripped.startswith("/") and len(stripped) > 1
        else None
    )
