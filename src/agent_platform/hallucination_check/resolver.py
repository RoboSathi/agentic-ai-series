"""Grounding: which tiers may ground a claim, and what each says.

The evidence is not one pool. A claim is classed (record, rule or
conversation) and is then read against only the tiers that may ground its
class, each with its own top-k, so a policy section can no longer crowd a
tool fact out of the ranking. That is what removed the reconciliation
machinery this module used to carry — authority ranks, support-only
tiers, supersession, one voice per observation — all of it existed to
undo the pooling.

Evidence that states one fact across several lines (a web page) needs
premises wider than a sentence, and gets two, both support-only: passage
windows (`passage_window`, built with the evidence) and, for a claim a
tier left undecided, the relevant evidence of one source joined in
source order (`joined_premise`). A paragraph that also says what is not
so reads to NLI as contradicting what is, so neither ever refutes.

A claim no tier decided may get a second opinion (`confirmer.py`): a decision model reads it
beside its best source's text. It may confirm the claim, never refute it.

Embeddings have one job here: retrieve the best evidence for a claim.
They never decide whether the claim is true. NLI makes that decision.
"""

import asyncio
import re
from typing import Literal

from agent_platform.config import get_settings
from agent_platform.config.settings import GroundingTierSettings
from agent_platform.context.retrieval import DenseRetriever, RelevanceRanker
from agent_platform.context.types import Chunk
from agent_platform.hallucination_check.claims import ClaimClassifier
from agent_platform.hallucination_check.confirmer import ClaimConfirmer
from agent_platform.hallucination_check.direct import (
    NOT_EXECUTED_MARKER,
    DirectChecker,
    DirectFinding,
    amounts,
    numbers,
)
from agent_platform.hallucination_check.ledger import (
    FACT_MISMATCH,
    FACT_MODEL,
    FACT_TIER,
    FactLedger,
    LedgerFinding,
)
from agent_platform.hallucination_check.states import StateModel
from agent_platform.hallucination_check.types import (
    ClaimClass,
    DetectionVerdict,
    EvidenceCheck,
)
from agent_platform.llm.embeddings import tokenize
from agent_platform.logging import instrument
from agent_platform.scoring.entailment import (
    EntailmentModel,
    EntailmentResult,
    create_entailment_model,
)

# `Chunk.metadata` key for the order in which an observation was made this
# turn. Only staleness reads it: a later observation of the same thing
# overtakes an earlier one, and after that the order means nothing.
OBSERVATION_SEQUENCE = "observation_sequence"
# `Chunk.metadata` key marking an observation that changed something (a
# write). A read taken before a write of the same thing describes a world
# that no longer exists.
STATE_CHANGING = "state_changing"
# `Chunk.metadata` key marking a premise wider than one sentence: a passage
# window, or a joined premise. It may support a claim, never refute one.
PASSAGE = "passage"
# The detector name of a rule that restates the claim in nearly its own words.
RESTATEMENT_MODEL = "rule-restatement"
# The label a line of the record carries when it was read directly and not
# scored by a model.
RECORD_LABEL = "record"
# What decided a claim with no evidence tier: the direct check, or nothing.
DIRECT_TIER = "deterministic"
NO_TIER = "none"
# A claim no tier decided that the second-opinion model confirmed against its source.
CONFIRM_TIER = "confirmed"
# A line that is only a link states nothing a claim could rest on.
_BARE_LINK = re.compile(r"\S+://\S+")
# A verifiable claim no evidence grounds, when the agent counts that as a hallucination.
CLAIM_UNSUPPORTED = "CLAIM_UNSUPPORTED"

ENTAILMENT_LABEL: Literal["entailment"] = "entailment"
_DECIDING_LABELS = frozenset({ENTAILMENT_LABEL, "contradiction"})


class _TierOutcome:
    """What one tier concluded about one claim."""

    def __init__(
        self,
        tier: GroundingTierSettings,
        label: str,
        result: EntailmentResult | None,
        checks: list[EvidenceCheck],
        retrieved: list[Chunk],
        note: str = "",
    ) -> None:
        self.note = note
        self.tier = tier
        self.label = label
        self.result = result
        self.checks = checks
        self.retrieved = retrieved


@instrument("hallucination_check")
class GroundingResolver:
    """Resolve one claim against the tiers that may ground it."""

    def __init__(
        self,
        ranker: RelevanceRanker,
        entailment: EntailmentModel | None = None,
        classifier: ClaimClassifier | None = None,
        confirmer: ClaimConfirmer | None = None,
    ) -> None:
        self._app_config = get_settings()
        settings = get_settings().hallucination_check
        self._confirmer = confirmer
        self._confirm_settings = settings.confirmer
        self._retriever = DenseRetriever(ranker)
        self._top_k = settings.embedding_top_k
        self._threshold = settings.nli_entailment_threshold
        self._uncalibrated_refutes = settings.uncalibrated_may_refute
        self._floor = settings.evidence_min_similarity
        self._joined = settings.joined_premise
        self._tiers = settings.tiers
        self._restatement = settings.rule_restatement_coverage
        self._identifier = (
            re.compile(settings.refute_identifier_pattern)
            if settings.refute_identifier_pattern
            else None
        )
        self._refuting = settings.refuting_premise
        self._negations = tuple(
            m.casefold() for m in get_settings().hallucination_check.scoring.nli.negation_markers
        )
        self._entailment = entailment or create_entailment_model()
        self._classifier = classifier or ClaimClassifier()
        self._direct = DirectChecker()
        self._states = StateModel()
        self._ledger = FactLedger()
        self._unsupported_blocks = settings.unsupported_is_hallucination
        # A state record ("Order N refund status: X") is read by the state
        # model. NLI cannot: it took `pending_review` to contradict "submitted".
        self._state_line = re.compile(settings.record_state_pattern, re.IGNORECASE)
        self._graded_kinds = frozenset(kind for tier in self._tiers for kind in tier.kinds)
        self._record_kinds = frozenset(
            kind for tier in self._tiers if "record" in tier.claims for kind in tier.kinds
        )
        # The rules that may refute: an amount found here is a policy amount, not an invented one.
        self._rule_kinds = frozenset(
            kind
            for tier in self._tiers
            if "rule" in tier.claims and tier.may_refute
            for kind in tier.kinds
        )

    async def resolve(
        self, claim: str, evidence: list[Chunk], claim_class: ClaimClass | None = None
    ) -> DetectionVerdict:
        """The verdict on one claim: refuted, supported, or undetermined."""
        kind = claim_class or self._classifier.classify(claim)
        if kind == "conversational":
            return self._verdict(
                claim,
                kind,
                NO_TIER,
                None,
                [],
                undetermined=True,
                rationale="conversation, not a claim",
            )
        evidence = self._current(evidence)
        kind = self._about_this_case(claim, kind, evidence)
        finding = self._direct_finding(claim, kind, evidence)
        if finding is not None:
            return self._read(
                claim,
                kind,
                finding.chunk,
                supported=False,
                rationale=finding.rationale,
                reason_code=finding.reason_code,
            )
        facts = self._ledger.judge(claim, evidence)
        if facts is not None:
            return self._from_facts(claim, kind, facts)
        record = [chunk for chunk in evidence if chunk.kind in self._record_kinds]
        settled = self._settled_by_record(claim, kind, record)
        if settled is not None:
            # The state line decided it; the lines holding the amounts it names are read too.
            also = [c for c in record if amounts(claim) & amounts(c.content)]
            return self._read(claim, kind, settled[1], supported=True, rationale=settled[0],
                              also=also)  # fmt: skip
        outcomes = [
            await self._judge(claim, tier, _tier_evidence(tier, evidence))
            for tier in self._tiers
            if kind in tier.claims
        ]
        verdict = self._settle(claim, kind, outcomes)
        if self._confirmer is None or verdict.tier != NO_TIER or verdict.supported:
            return verdict
        confirmed, asked, note = await self._second_opinion(claim, kind, evidence)
        # The card shows what NLI read beside what the second opinion read, either way.
        checks = self._ranked([*verdict.evidence_checks, *asked])
        if confirmed is not None:
            rationale = f"{verdict.rationale} · {confirmed.rationale}"
            return confirmed.model_copy(update={"evidence_checks": checks, "rationale": rationale})
        return verdict.model_copy(
            update={"rationale": f"{verdict.rationale} · {note}", "evidence_checks": checks}
        )

    @property
    def entailment(self) -> EntailmentModel:
        """The entailment model this resolver judges claims with."""
        return self._entailment

    def _about_this_case(self, claim: str, kind: ClaimClass, evidence: list[Chunk]) -> ClaimClass:
        """A rule claim the record's state decides is a fact about this case, checked as one.

        "No refund has been requested yet" names no order, so it reads as a rule, and a
        rule is checked against policy and knowledge — where an FAQ line about refunds
        "contradicted" it. It states where this refund stands, which only the record can
        show. A promise stays a rule: what will happen is the business's to state.
        """
        if kind != "rule" or self._classifier.promises(claim):
            return kind
        record = [chunk for chunk in evidence if chunk.kind in self._record_kinds]
        return "record" if self._states.judge(claim, record) is not None else kind

    def _from_facts(self, claim: str, kind: ClaimClass, finding: LedgerFinding) -> DetectionVerdict:
        """A verdict the typed facts decided, quoting every fact they read."""
        return self._verdict(
            claim,
            kind,
            FACT_TIER,
            None,
            finding.chunks,
            supported=finding.holds,
            rationale=finding.rationale,
            reason_code="" if finding.holds else FACT_MISMATCH,
            checks=[self._record_check(chunk) for chunk in finding.chunks],
            detector=FACT_MODEL,
        )

    def _settled_by_record(
        self, claim: str, kind: ClaimClass, record: list[Chunk]
    ) -> tuple[str, Chunk] | None:
        """A claim the record decides by itself: a refusal, a status, or a commitment under way."""
        if kind != "record":
            return self._direct.kept_commitment(claim, record)
        return self._direct.refusal_support(claim, record) or self._direct.state_support(
            claim, record
        )

    def _read(
        self,
        claim: str,
        kind: ClaimClass,
        chunk: Chunk | None,
        *,
        supported: bool,
        rationale: str,
        reason_code: str = "",
        also: list[Chunk] | None = None,
    ) -> DetectionVerdict:
        """A verdict settled by reading the record, quoting every line it read."""
        read = list({id(c): c for c in [*([chunk] if chunk else []), *(also or [])]}.values())
        checks = [self._record_check(c) for c in read]
        return self._verdict(
            claim,
            kind,
            DIRECT_TIER,
            None,
            read,
            supported=supported,
            rationale=rationale,
            reason_code=reason_code,
            checks=checks,
        )

    @staticmethod
    def _record_check(chunk: Chunk) -> EvidenceCheck:
        return EvidenceCheck(
            rank=1,
            similarity=1.0,
            tier=chunk.kind,
            excerpt=chunk.content,
            nli_label=RECORD_LABEL,
            nli_score=1.0,
            decisive=True,
            source=chunk.source,
            chosen=True,
        )

    def _direct_finding(
        self, claim: str, kind: ClaimClass, evidence: list[Chunk]
    ) -> DirectFinding | None:
        """What the record, or the evidence as a whole, rules out without asking a model."""
        record = [chunk for chunk in evidence if chunk.kind in self._record_kinds]
        if kind == "record":
            rules = [chunk for chunk in evidence if chunk.kind in self._rule_kinds]
            return self._direct.refutation(claim, record, rules)
        return self._direct.unsourced_detail(
            claim, [chunk for chunk in evidence if chunk.kind in self._graded_kinds]
        ) or self._direct.broken_commitment(claim, record)

    def _settle(
        self, claim: str, kind: ClaimClass, outcomes: list[_TierOutcome]
    ) -> DetectionVerdict:
        """A refutation that may stand wins; else the first support; else no verdict."""
        checks = self._ranked([check for outcome in outcomes for check in outcome.checks])
        retrieved = [chunk for outcome in outcomes for chunk in outcome.retrieved]
        for label, supported in (("contradiction", False), ("entailment", True)):
            for outcome in outcomes:
                if outcome.label == label and outcome.result is not None:
                    return self._verdict(
                        claim,
                        kind,
                        outcome.tier.name,
                        outcome.result,
                        retrieved,
                        supported=supported,
                        checks=self._chosen_by(outcome, outcomes),
                    )
        why = "no tier decided: " + " · ".join(
            f"{outcome.tier.name}: {outcome.note}" for outcome in outcomes
        )
        return self._verdict(
            claim,
            kind,
            NO_TIER,
            None,
            retrieved,
            # Made up, when the agent counts a claim nothing grounds as a hallucination.
            undetermined=not self._unsupported_blocks,
            reason_code=CLAIM_UNSUPPORTED if self._unsupported_blocks else "",
            checks=checks,
            rationale=why,
        )

    def _chosen_by(self, winner: _TierOutcome, outcomes: list[_TierOutcome]) -> list[EvidenceCheck]:
        """Every tier's lines, with only the deciding tier's choice marked as the one chosen."""
        return self._ranked(
            [
                check if outcome is winner else check.model_copy(update={"chosen": False})
                for outcome in outcomes
                for check in outcome.checks
            ]
        )

    def _current(self, evidence: list[Chunk]) -> list[Chunk]:
        """The evidence with every read the turn has since overtaken taken out.

        "No refund has been requested for order 10169" was true when it was
        read and false once the refund was referred, and nothing the later
        observation says needs to contradict it for that to be so. A read
        is overtaken by a *later* observation of the same subject that
        either changed it (a write) or looked at it again through the same
        tool (a re-read, after a settlement sweep moved it, say). A read
        about another subject, or in a turn that neither wrote nor re-read,
        stands; so does any read a refused call came after, since a refused
        call observed nothing — and what stands is one world, which `_deciding` relies on.
        """
        later = [
            (sequence, chunk.source, bool(chunk.metadata.get(STATE_CHANGING)), named)
            for chunk in evidence
            if isinstance(sequence := chunk.metadata.get(OBSERVATION_SEQUENCE), int)
            and not chunk.content.startswith(NOT_EXECUTED_MARKER)
            and (named := self._subjects(chunk.content))
        ]

        def overtaken(chunk: Chunk) -> bool:
            sequence = chunk.metadata.get(OBSERVATION_SEQUENCE)
            if chunk.metadata.get(STATE_CHANGING) or not isinstance(sequence, int):
                return False
            mine = self._subjects(chunk.content)
            return any(
                seen > sequence and (wrote or self._reread(source, chunk.source)) and named & mine
                for seen, source, wrote, named in later
            )

        return [chunk for chunk in evidence if not overtaken(chunk)]

    @staticmethod
    def _reread(later: str, earlier: str) -> bool:
        """Whether two observations came through the same tool. An unnamed source is no tool."""
        return bool(earlier) and later == earlier

    def _subjects(self, text: str) -> set[str]:
        """What a text is about: the agent's identifiers if it names a pattern, else its numbers.

        Bare numbers also catch dates and amounts, so two reads of different
        orders placed the same year would otherwise be one subject.
        """
        return set(self._identifier.findall(text)) if self._identifier else numbers(text)

    def _is_state_record(self, chunk: Chunk) -> bool:
        """Whether this is a line of the record's state, which only the state model reads."""
        return chunk.kind in self._record_kinds and bool(self._state_line.search(chunk.content))

    async def _judge(
        self, claim: str, tier: GroundingTierSettings, chunks: list[Chunk]
    ) -> _TierOutcome:
        """One tier's answer: top-k within the tier, then NLI, latest observation first."""
        chunks = [chunk for chunk in chunks if not self._is_state_record(chunk)]
        if not chunks:
            return _TierOutcome(tier, "abstain", None, [], [], note="no evidence")
        self._retriever.index(chunks)
        retrieved = await self._retriever.retrieve(claim, self._top_k)
        relevant = [chunk for chunk in retrieved if chunk.score >= self._floor]
        restated = self._restated(claim, relevant) if "rule" in tier.claims else None
        results = (
            []
            if restated
            else list(
                await asyncio.gather(
                    *(self._entailment.entails(c.content, claim) for c in relevant)
                )
            )
        )
        judged = [(restated[0], restated[1])] if restated else list(zip(results, relevant))
        chosen = self._deciding(claim, tier, judged)
        if chosen is None and self._joined.enabled:
            joined = await self._joined_support(claim, relevant, chunks)
            if joined is not None:
                chosen, judged, retrieved = joined, [*judged, joined], [*retrieved, joined[1]]
        checks = self._checks(retrieved, judged, chosen[1] if chosen else None)
        label = chosen[0].label if chosen else "abstain"
        note = "" if chosen else self._why_undecided(tier, retrieved, judged)
        return _TierOutcome(tier, label, chosen[0] if chosen else None, checks, retrieved, note)

    async def _joined_support(
        self, claim: str, relevant: list[Chunk], chunks: list[Chunk]
    ) -> tuple[EntailmentResult, Chunk] | None:
        """The surest support from one source's relevant evidence, joined in source order.

        "Created by Kumar Karan, a former Lead Instructor at Scaler Academy" is entailed by
        no single line of the page that states it ("Kumar Karan", "Lead Instructor @ Scaler
        Academy"), and by the lines together. Support only: a joined paragraph also says
        things the claim does not, which NLI reads as contradiction.
        """
        order = {(chunk.source, chunk.content): index for index, chunk in enumerate(chunks)}
        by_source: dict[str, list[Chunk]] = {}
        for chunk in relevant:
            by_source.setdefault(chunk.source, []).append(chunk)
        premises = [
            self._joined_chunk(sorted(group, key=lambda c: order.get((c.source, c.content), 0)))
            for group in by_source.values()
            if len(group) > 1
        ]
        results = await asyncio.gather(
            *(self._entailment.entails(premise.content, claim) for premise in premises)
        )
        supports = [
            (result, premise)
            for result, premise in zip(results, premises, strict=True)
            if result.label == ENTAILMENT_LABEL
            and self._is_decisive(result)
            and not self._names_only_others(claim, premise)
        ]
        return max(supports, key=self._strength, default=None)

    def _joined_chunk(self, group: list[Chunk]) -> Chunk:
        """One premise from a source's chunks: each sentence once, up to `max_chars`."""
        text = ""
        for chunk in group:
            if chunk.content in text:
                continue  # already inside a passage window joined before it
            joined = f"{text} {chunk.content}".strip()
            if len(joined) > self._joined.max_chars:
                break
            text = joined
        first = group[0]
        return first.model_copy(
            update={
                "content": text,
                "score": max(chunk.score for chunk in group),
                "metadata": {**first.metadata, PASSAGE: True},
            }
        )

    async def _second_opinion(
        self, claim: str, kind: ClaimClass, evidence: list[Chunk]
    ) -> tuple[DetectionVerdict | None, list[EvidenceCheck], str]:
        """The confirmer's verdict on a claim no tier decided, source by source, best first.

        Each source's text is the lines most like the claim, put back in source order, up to
        `max_evidence_chars`: the claim's facts, with the subject they belong to around them.
        Returns the supported verdict (or None), every answer it gave as an evidence line, and
        what it said when it did not confirm.
        """
        assert self._confirmer is not None
        kinds = {chunk_kind for t in self._tiers if kind in t.claims for chunk_kind in t.kinds}
        lines = [
            chunk
            for chunk in evidence
            if chunk.kind in kinds
            and not chunk.metadata.get(PASSAGE)
            and not self._is_state_record(chunk)
            and not _BARE_LINK.fullmatch(chunk.content.strip())
        ]
        model = self._confirmer.model_name
        if not lines:
            return None, [], f"{model}: no evidence"
        self._retriever.index(lines)
        ranked = await self._retriever.retrieve(claim, len(lines))
        notes: list[str] = []
        asked: list[EvidenceCheck] = []
        for source in self._best_sources(ranked):
            text = self._source_text(source, ranked, lines)
            answer = await self._confirmer.confirm(claim, text)
            asked.append(self._confirm_check(text, source, ranked, answer.label, answer.score,
                                             decisive=answer.confirmed))  # fmt: skip
            if answer.confirmed:
                verdict = self._confirmed(claim, kind, text, source, ranked, answer.score)
                return verdict, asked, ""
            notes.append(f"{answer.label} {answer.score:.2f} ({source or 'evidence'})")
        return None, asked, f"{model}: {', '.join(notes) or 'nothing related'}"

    def _confirm_check(
        self,
        text: str,
        source: str,
        ranked: list[Chunk],
        label: str,
        score: float,
        *,
        decisive: bool,
    ) -> EvidenceCheck:
        """The second opinion's answer on one source, as an evidence line of the card."""
        assert self._confirmer is not None
        best = next((c for c in ranked if c.source == source), None)
        return EvidenceCheck(
            rank=0,
            similarity=round(best.score, 3) if best is not None else 0.0,
            tier=best.kind if best is not None else "",
            excerpt=text,
            nli_label=label,
            nli_score=round(score, 3),
            decisive=decisive,
            source=source,
            model=self._confirmer.model_name,
            chosen=decisive,
        )

    def _best_sources(self, ranked: list[Chunk]) -> list[str]:
        """Sources holding a line related to the claim, closest first."""
        related = [chunk.source for chunk in ranked if chunk.score >= self._floor]
        return list(dict.fromkeys(related))[: self._confirm_settings.max_sources]

    def _source_text(self, source: str, ranked: list[Chunk], lines: list[Chunk]) -> str:
        """The source's lines closest to the claim, in source order, within the budget."""
        order = {(c.source, c.content): index for index, c in enumerate(lines)}
        budget = self._confirm_settings.max_evidence_chars
        chosen: list[Chunk] = []
        for chunk in (c for c in ranked if c.source == source):
            if sum(len(c.content) + 1 for c in chosen) + len(chunk.content) > budget:
                break
            chosen.append(chunk)
        chosen.sort(key=lambda c: order.get((c.source, c.content), 0))
        return "\n".join(dict.fromkeys(chunk.content for chunk in chosen))

    def _confirmed(
        self,
        claim: str,
        kind: ClaimClass,
        text: str,
        source: str,
        ranked: list[Chunk],
        score: float,
    ) -> DetectionVerdict:
        """A supported verdict the confirmer gave, quoting the text it read."""
        assert self._confirmer is not None
        model = self._confirmer.model_name
        result = EntailmentResult(
            label=ENTAILMENT_LABEL,
            entailed=True,
            score=score,
            calibrated=False,
            model=model,
            rationale=f"{model}: {self._confirm_settings.supported_label} {score:.2f}",
        )
        best = next((c for c in ranked if c.source == source), None)
        read = Chunk(
            content=text,
            kind=best.kind if best is not None else "tool_result",
            source=source,
            score=best.score if best is not None else 0.0,
            metadata={PASSAGE: True},
        )
        return self._verdict(claim, kind, CONFIRM_TIER, result, [read], supported=True)

    def _why_undecided(
        self,
        tier: GroundingTierSettings,
        retrieved: list[Chunk],
        judged: list[tuple[EntailmentResult, Chunk]],
    ) -> str:
        """In a few words, why this tier could not decide the claim."""
        if not judged:
            best = max((chunk.score for chunk in retrieved), default=0.0)
            return f"nothing related (best {best:.2f} < {self._floor})"
        result = max((item[0] for item in judged), key=lambda item: item.score)
        refutation = result.label == "contradiction" and not tier.may_refute
        if refutation:
            return f"contradiction ignored, this tier cannot refute ({result.score:.2f})"
        names = self._app_config.hallucination_check.card.label_names
        label = names.get(result.label, result.label)
        decisive = result.label in _DECIDING_LABELS
        gloss = "" if decisive else " (it neither states nor contradicts this)"
        return f"{label} {result.score:.2f}{gloss}"

    def _deciding(
        self,
        claim: str,
        tier: GroundingTierSettings,
        judged: list[tuple[EntailmentResult, Chunk]],
    ) -> tuple[EntailmentResult, Chunk] | None:
        """The one decisive result that speaks for the tier: any support, else a refutation.

        A tier's evidence, once `_current` has removed what the turn
        overtook, is one consistent body — the record as it now stands, or
        the rulebook — so it cannot both entail a claim and contradict it.
        When the model says it does, the contradiction is about another
        attribute: "the refund for order 10842 has been approved" against
        "order 10842 was cancelled". So a contradiction stands only when
        nothing the tier holds entails the claim. Neither order nor
        closeness decides it; the first rule let a later read of the refund
        overrule an earlier read of the order it said nothing about.

        A false support is no cheaper than a false refutation, so support
        must come from evidence about what the claim names, and a claim
        about a record's state has already met the deterministic state
        check before any model is asked.
        """
        decisive = [
            item
            for item in judged
            if item[0].label in _DECIDING_LABELS
            and self._is_decisive(item[0])
            and self._may_decide(claim, tier, item)
        ]
        supports = [item for item in decisive if item[0].label == ENTAILMENT_LABEL]
        return max(supports or decisive, key=self._strength, default=None)

    def _may_decide(
        self, claim: str, tier: GroundingTierSettings, item: tuple[EntailmentResult, Chunk]
    ) -> bool:
        """Whether this result may speak: support about this subject, refutation where allowed.

        A passage (several sentences as one premise) supports only.
        """
        if item[0].label == ENTAILMENT_LABEL:
            return not self._names_only_others(claim, item[1])
        passage = bool(item[1].metadata.get(PASSAGE))
        return (
            tier.may_refute
            and not passage
            and self._may_refute(claim, item[1])
            and self._is_statement(item[1].content)
        )

    def _is_statement(self, text: str) -> bool:
        """Whether a line reads as a statement, not a heading, a menu or a row of links."""
        words = re.findall(r"[^\W\d_][\w'’-]*", text)
        if len(words) < self._refuting.min_words:
            return False
        capitalised = sum(1 for word in words if word[0].isupper())
        return capitalised / len(words) <= self._refuting.max_capitalised_share

    @staticmethod
    def _strength(item: tuple[EntailmentResult, Chunk]) -> tuple[float, float]:
        """The surest result first, and the closer evidence between two equally sure."""
        return (item[0].score, item[1].score)

    def _restated(self, claim: str, chunks: list[Chunk]) -> tuple[EntailmentResult, Chunk] | None:
        """A rule that says the claim in nearly its own words.

        "Tell the customer … they'll hear back soon" is an instruction, and
        NLI on an instruction against the statement it dictates is a coin
        toss that has come up "contradiction". The rule *is* the
        grounding for the sentence it scripts, so a rule that covers the
        claim's content, with the same polarity, supports it directly.
        """
        wanted = set(tokenize(claim)) - set(self._negations)
        for chunk in chunks:
            have = set(tokenize(chunk.content))
            coverage = len(wanted & have) / len(wanted) if wanted else 0.0
            if coverage >= self._restatement and self._negated(claim) == self._negated(
                chunk.content
            ):
                return (
                    EntailmentResult(
                        label="entailment",
                        entailed=True,
                        score=coverage,
                        calibrated=False,
                        model=RESTATEMENT_MODEL,
                        rationale=f"{chunk.kind} restates {coverage:.0%} of the claim",
                    ),
                    chunk,
                )
        return None

    def _negated(self, text: str) -> bool:
        lowered = text.casefold()
        return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in self._negations)

    def _may_refute(self, claim: str, chunk: Chunk) -> bool:
        """Whether this item is about what the claim is about, so it may contradict it.

        A claim naming order 10876 is not refuted by the record of order
        10854 — NLI calls that a contradiction because the numbers differ.
        """
        if self._identifier is None:
            return True
        named = set(self._identifier.findall(claim))
        return not named or bool(named & set(self._identifier.findall(chunk.content)))

    def _names_only_others(self, claim: str, chunk: Chunk) -> bool:
        """Whether evidence names subjects, none of them the claim's.

        "Order 10843 has been cancelled" says nothing about order 10842,
        whatever a model makes of it. Evidence naming nothing (a carrier's
        note, split from the sentence that named its order) may still
        support.
        """
        if self._identifier is None:
            return False
        named = set(self._identifier.findall(claim))
        theirs = set(self._identifier.findall(chunk.content))
        return bool(named and theirs and not named & theirs)

    def _is_decisive(self, result: EntailmentResult) -> bool:
        """Whether a result settles the claim, judged on its own scale.

        `nli_entailment_threshold` is a probability, and only a calibrated
        classifier produces one. The lexical fallback scores token
        coverage instead, and a claim that restates its evidence exactly
        still covers only the words the evidence happens to contain. So
        the threshold applies only where it means something: an
        uncalibrated model's label is its decision. Whether that includes refuting is the
        agent's call (`uncalibrated_may_refute`): a word-overlap guess that two sentences
        disagree is not evidence enough to remove a sentence from an answer.
        """
        if not result.calibrated:
            return result.label == ENTAILMENT_LABEL or self._uncalibrated_refutes
        return result.score >= self._threshold

    def _checks(
        self,
        retrieved: list[Chunk],
        judged: list[tuple[EntailmentResult, Chunk]],
        chosen: Chunk | None = None,
    ) -> list[EvidenceCheck]:
        """Every retrieved item in similarity order, with the NLI result it got (if any)."""
        by_chunk = {id(chunk): result for result, chunk in judged}
        return [
            EvidenceCheck(
                rank=0,
                similarity=round(chunk.score, 3),
                tier=chunk.kind,
                excerpt=chunk.content,
                nli_label=by_chunk[id(chunk)].label
                if id(chunk) in by_chunk
                else self._unchecked(chunk),
                nli_score=round(by_chunk[id(chunk)].score, 3) if id(chunk) in by_chunk else 0.0,
                decisive=id(chunk) in by_chunk
                and by_chunk[id(chunk)].label in _DECIDING_LABELS
                and self._is_decisive(by_chunk[id(chunk)]),
                source=chunk.source,
                model=by_chunk[id(chunk)].model if id(chunk) in by_chunk else "",
                chosen=chunk is chosen,
            )
            for chunk in retrieved
        ]

    def _unchecked(self, chunk: Chunk) -> str:
        """Why a retrieved item got no NLI call: unrelated, or the claim was already decided."""
        if chunk.score < self._floor:
            return f"not checked (below {self._floor})"
        return f"not checked ({self._app_config.hallucination_check.methods.restated})"

    @staticmethod
    def _ranked(checks: list[EvidenceCheck]) -> list[EvidenceCheck]:
        return [
            check.model_copy(update={"rank": rank}) for rank, check in enumerate(checks, start=1)
        ]

    def _verdict(
        self,
        claim: str,
        kind: ClaimClass,
        tier: str,
        result: EntailmentResult | None,
        retrieved: list[Chunk],
        *,
        supported: bool = False,
        undetermined: bool = False,
        rationale: str = "",
        reason_code: str = "",
        checks: list[EvidenceCheck] | None = None,
        detector: str = "",
    ) -> DetectionVerdict:
        """An auditable verdict: who decided, on what, and everything retrieved."""
        return DetectionVerdict(
            claim=claim,
            claim_class=kind,
            supported=supported,
            undetermined=undetermined,
            confidence=result.score if result else 0.0,
            tier=tier,
            rationale=rationale or (result.rationale if result else ""),
            reason_code=reason_code,
            detector_model=detector
            or (result.model if result else "")
            or (DIRECT_TIER if tier == DIRECT_TIER else self._detector_model),
            retrieval_top_k=self._top_k,
            retrieval_method=self._retrieval_method,
            retrieved_evidence=[chunk.content for chunk in retrieved],
            evidence_checks=checks or [],
            threshold=self._threshold if result is not None and result.calibrated else None,
        )

    @property
    def _retrieval_method(self) -> str:
        """Report the actual retrieval path used by the injected ranker."""
        cascade = getattr(self._retriever, "_ranker", None)
        enabled = self._app_config.hallucination_check.scoring.tiers.embedding_similarity
        return "embedding" if enabled and cascade is not None else "lexical_fallback"

    @property
    def _detector_model(self) -> str:
        """Expose the concrete NLI model, including its deterministic fallback."""
        return str(getattr(self._entailment, "model_name", "")) or "lexical-entailment-fallback"


def _tier_evidence(tier: GroundingTierSettings, evidence: list[Chunk]) -> list[Chunk]:
    """The evidence a tier reads: its kinds, with its passages, without them, or only them."""
    if tier.passages == "include":
        return [chunk for chunk in evidence if chunk.kind in tier.kinds]
    wanted = tier.passages == "only"
    return [
        chunk
        for chunk in evidence
        if chunk.kind in tier.kinds and bool(chunk.metadata.get(PASSAGE)) == wanted
    ]
