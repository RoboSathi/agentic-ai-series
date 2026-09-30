"""The scoping graph's nodes.

Each node does one thing and reports it: the gate's decision, each
interview turn (with the checklist it left behind), a waiver, the rubric,
the frozen contract. What the model proposes is filtered through code —
the termination policy and the challenge guard — before it becomes state.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.config.settings import CommandNames, MessageSettings, ScopingSettings
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import GENERAL, GateVerdict, Mode, ScopeBrief, UnknownReason
from research_agent.graph.state import ScopingState
from research_agent.prompts import interviewer as interviewer_prompt
from research_agent.research.telemetry import RunTelemetry
from research_agent.rubric.editor import RubricEditError, RubricEditor, render
from research_agent.rubric.freeze import freeze_contract
from research_agent.rubric.generator import RubricGenerator
from research_agent.scoping import checklist
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.challenge_check import ChallengeGuard
from research_agent.scoping.gate import (
    ArchetypeClassifier,
    render_options,
    resolve_reframe_choice,
    with_mode,
)
from research_agent.scoping.termination import TerminationPolicy
from research_agent.structured import StructuredCaller

Update = dict[str, Any]


@dataclass
class ScopingParts:
    """The collaborators scoping is composed of; optional ones are features."""

    # "Now" for the contract (real time live; the fixture world's date in the demo).
    clock: Callable[[], datetime]
    classifier: ArchetypeClassifier
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
    """Gate, interview, waivers, rubric review and freezing, as graph nodes."""

    def __init__(self, parts: ScopingParts) -> None:
        self._p = parts

    # -- gate ------------------------------------------------------------------------

    async def gate(self, state: ScopingState) -> Update:
        """Q&A, a Research kind, a reframe menu, or a refusal — for the first message."""
        choice = self._reframe_choice(state)
        forced = choice or state.forced_mode
        message = (
            state.first_message
            if state.phase == "GATE" or choice is not None
            else f"{state.first_message} {state.user_input}"
        )
        verdict = await self._p.classifier.classify(message)
        if forced is not None and verdict.decision != "refuse":
            verdict = with_mode(verdict, forced)
        self._report_gate(verdict)
        if verdict.decision == "refuse":
            return {"phase": "REFUSED", "reply": self._p.messages.refuse}
        if verdict.decision == "needs_reframe" or verdict.archetype is None:
            return {"phase": "REFRAME", "reply": self._menu(), "user_input": ""}
        found = {"topic": verdict.topic or message, "recent_days": verdict.recent_days}
        if verdict.mode == "qa":
            return {"phase": "QA", "mode": "qa", "archetype": verdict.archetype, **found}
        items = checklist.load_checklist(self._p.checklists, verdict.archetype)
        return {
            "phase": "INTERVIEW",
            "mode": "research",
            "archetype": verdict.archetype,
            "items": checklist.prefill_window(items, verdict.recent_days),
            **found,
        }

    def _reframe_choice(self, state: ScopingState) -> Mode | None:
        if state.phase != "REFRAME":
            return None
        return resolve_reframe_choice(state.user_input, self._p.settings.reframe_options)

    def _menu(self) -> str:
        options = render_options(self._p.settings.reframe_options, self._p.messages.reframe_option)
        return self._p.messages.reframe.format(options=options)

    def _report_gate(self, verdict: GateVerdict) -> None:
        kind = verdict.archetype if verdict.decision == "scopable" else None
        self._p.telemetry.stage(
            "scope_gate",
            f"{verdict.decision}" + (f" · {verdict.mode} · {kind}" if kind else ""),
            verdict.model_dump(),
            status="blocked" if verdict.decision == "refuse" else "completed",
        )
        self._p.telemetry.audit(
            "scope.gate",
            "deny" if verdict.decision == "refuse" else "allow",
            verdict.reason or verdict.decision,
            archetype=verdict.archetype,
            gate_decision=verdict.decision,
            mode=verdict.mode,
        )

    @staticmethod
    def after_gate(state: ScopingState) -> str:
        if state.phase == "REFUSED":
            return "__end__"
        if state.phase == "QA":
            return "auto_freeze"
        return "await_user" if state.phase == "REFRAME" else "interview_turn"

    async def auto_freeze(self, state: ScopingState) -> Update:
        """Q&A: the question becomes a Brief, the Brief a contract — no interview, no review."""
        brief = self._p.briefs.question(state.topic, state.recent_days)
        rubric = self._p.generator.generate(brief) if self._p.generator is not None else Rubric()
        contract = self._freeze(brief, rubric, "the Q&A question was asked")
        return {
            "brief": brief,
            "rubric": rubric,
            "contract": contract,
            "phase": "APPROVED",
            "reply": self._p.messages.qa_started,
        }

    # -- interview -------------------------------------------------------------------

    async def interview(self, state: ScopingState) -> Update:
        turn_number = state.turn + 1
        prompt = interviewer_prompt.build(
            archetype=state.archetype or "",
            topic=state.topic,
            items=state.items,
            asked_item_id=state.asked_item_id,
            answer=state.user_input,
            turn=turn_number,
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
            "asked_item_id": turn.next_item_id,
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
        verdict = self._p.challenge.check(turn, state.asked_item_id, state.since_challenge)
        if not verdict.violated:
            return turn, verdict.turns_without_challenge
        self._p.telemetry.audit(
            "scope.challenge_guard",
            "escalate",
            "interviewer went K turns without a challenge",
            turns=verdict.turns_without_challenge,
        )
        self._p.telemetry.stage(
            "interview_turn",
            "Challenge guard: re-prompting with an explicit challenge",
            {"turns_without_challenge": verdict.turns_without_challenge},
            status="warning",
        )
        retried = await self._p.caller.call(
            prompt.with_instruction(interviewer_prompt.CHALLENGE_INSTRUCTION),
            interviewer_prompt.InterviewTurn,
        )
        return retried, 0 if ChallengeGuard.challenged(
            retried, state.asked_item_id
        ) else verdict.turns_without_challenge

    @staticmethod
    def _apply(items: list[Any], turn: interviewer_prompt.InterviewTurn) -> list[Any]:
        for update in turn.updated_items:
            if update.value.strip():
                items = checklist.with_status(
                    items, update.item_id, "covered", update.value.strip()
                )
        return items

    def _reply(self, turn: interviewer_prompt.InterviewTurn, outcome: str, items: list[Any]) -> str:
        challenges = [f"{self._p.messages.challenge_prefix}{c}" for c in turn.challenges]
        if outcome == "propose_done":
            plan = "\n".join(f"  - {step}" for step in turn.draft_plan)
            return "\n".join([*challenges, self._p.messages.propose_done.format(plan=plan)])
        if outcome == "turn_cap":
            open_ids = ", ".join(
                item.item_id for item in checklist.open_items(items, required_only=True)
            )
            return "\n".join([*challenges, self._p.messages.turn_cap.format(open=open_ids)])
        return "\n".join([*challenges, turn.next_question])

    def _report_turn(
        self, number: int, turn: Any, items: list[Any], outcome: str, overruled: bool
    ) -> None:
        self._p.telemetry.stage(
            "interview_turn",
            f"Turn {number} · {outcome}" + (" · premature done overruled" if overruled else ""),
            {
                "next_item": turn.next_item_id,
                "challenges": len(turn.challenges),
                "outcome": outcome,
            },
        )
        self._p.telemetry.stage(
            "coverage_update",
            " · ".join(f"{i.item_id}:{i.status}" for i in items),
            {i.item_id: i.status for i in items},
        )

    # -- waivers, closing ----------------------------------------------------------------

    async def waive(self, state: ScopingState) -> Update:
        target = (
            state.user_input.split(maxsplit=1)[1].strip() if " " in state.user_input.strip() else ""
        )
        items = checklist.waive(state.items, target, self._p.settings.waive_all_keyword)
        waived = [i.item_id for i in items if i.status == "waived"]
        self._p.telemetry.stage(
            "coverage_update", f"Waived: {', '.join(waived) or 'nothing'}", {"waived": waived}
        )
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
        brief = self._p.briefs.build(state.archetype or GENERAL, state.topic, state.items, unknowns)
        self._p.telemetry.stage(
            "close_scoping",
            f"Scope Brief · {len(unknowns)} known unknown(s)",
            {"archetype": brief.archetype, "known_unknowns": [u.item_id for u in unknowns]},
        )
        return {"brief": brief, "phase": "RUBRIC_REVIEW", "user_input": ""}

    # -- rubric ---------------------------------------------------------------------

    async def generate_rubric(self, state: ScopingState) -> Update:
        assert state.brief is not None
        rubric = (
            self._p.generator.generate(state.brief) if self._p.generator is not None else Rubric()
        )
        must = sum(item.must_pass for item in rubric.items)
        self._p.telemetry.stage(
            "rubric_generate",
            f"{len(rubric.items)} criteria ({must} must-pass)",
            {"items": [i.item_id for i in rubric.items]},
        )
        return {"rubric": rubric, "reply": f"{render(rubric)}\n{self._p.messages.rubric_hint}"}

    async def brief_ready(self, state: ScopingState) -> Update:
        """With the rubric withheld there is nothing to review but the Brief."""
        return {"rubric": Rubric(), "reply": self._p.messages.brief_ready}

    async def edit_rubric(self, state: ScopingState) -> Update:
        assert state.rubric is not None
        try:
            rubric = self._edit(state.rubric, state.user_input.split()[1:])
        except RubricEditError as exc:
            return {"reply": self._p.messages.rubric_error.format(error=exc), "user_input": ""}
        self._p.telemetry.stage(
            "rubric_generate",
            f"Rubric edited · {len(rubric.items)} criteria",
            {"edit": state.user_input},
        )
        return {
            "rubric": rubric,
            "reply": f"{render(rubric)}\n{self._p.messages.rubric_hint}",
            "user_input": "",
        }

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
        self._p.telemetry.stage(
            "rubric_frozen",
            f"Contract {contract.contract_id} · {contract.contract_hash[:12]}",
            {"contract_id": contract.contract_id, "hash": contract.contract_hash},
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
        if state.phase == "REFRAME":
            return "scope_gate"
        verb = _verb(state.user_input)
        if state.phase == "RUBRIC_REVIEW":
            return self._review_route(verb)
        return self._interview_route(verb)

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
