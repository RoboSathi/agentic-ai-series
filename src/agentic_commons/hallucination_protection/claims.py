"""Claims: what a reply asserts, in the smallest checkable pieces, and what kind each is.

A claim is not always a fact. "Thanks for reaching out" asserts nothing;
"you'll hear back soon" promises, and a promise is grounded by the rule
that makes it, not by a record; "$18.70 has been refunded" reports what
happened to this case and only a record can show it. Verifying all three
the same way — retrieve, then ask NLI — is how a courteous reply came to
be flagged as a hallucination, so the class is decided first and the
class decides what may ground the claim.
"""

import re

from agentic_commons.config import get_settings
from agentic_commons.hallucination_protection.types import AtomicClaim, ClaimClass
from agentic_commons.llm.embeddings import tokenize

# Sentence boundaries, and the connectives that join two claims into one
# sentence — both produce claims that must be checked separately.
#
# The semicolon belongs here for the same reason the full stop does. Left
# out, "the refund has not been initiated; the status shows none" stayed a
# single claim, and its two halves were scored against evidence that
# matched only one of them — so a sentence half of which was verbatim
# supported scored as entirely unsupported.
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")
# Causal connectives split too. "Queued for review because the amount
# exceeds the limit" is two claims from two tiers — the outcome from a tool
# result, the reason from the policy — and no single premise entails both.
# "Totals $117.23, so I cannot issue a $1 refund" likewise: joined, NLI
# read the pair as contradicting "a refund is always for the full total".
_CONJUNCTION = re.compile(r",?\s+\b(?:and|but|however|although|while|because|since|so)\b\s+")
# A clause that opens on one of these has lost its subject to the split:
# "Your order has been cancelled and is awaiting review" leaves "is awaiting
# review", which no premise can entail — it names nothing. It borrows the
# subject of the clause before it.
_AUXILIARY = re.compile(
    r"(?:is|are|was|were|has|have|had|will|would|can|could|should|may|might|must|"
    r"does|do|did)\b",
    re.IGNORECASE,
)
# A clause that opens on a past participle ("…has been cancelled and refunded
# in full") has lost its subject *and* its auxiliaries: it borrows both, so
# the claim reads "Order 10988 has already been refunded in full". Words that
# only look like participles are excluded.
_PARTICIPLE = re.compile(
    r"(?:then\s+)?(?!(?:then|when|even|often|seven|eleven)\b)[a-z]+(?:ed|en)\b",
    re.IGNORECASE,
)
# What may follow the first auxiliary and still belong to the verb group.
_VERB_GROUP = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "does",
        "do",
        "did",
        "be",
        "been",
        "being",
        "already",
        "also",
        "now",
        "just",
        "since",
        "then",
    }
)
# A clause with no auxiliary ("I cancelled it and will refund you") still
# names its subject when it opens on a pronoun.
_PRONOUN = re.compile(r"(?:I|we|you|he|she|it|they)\b", re.IGNORECASE)
# Brackets hold a list or an aside, never a second claim.
_OPENERS = "(["
_CLOSERS = ")]"
# Stripped from a word before asking whether it is a bare specific.
_ITEM_PUNCTUATION = "()[],;:"
_MIN_CONTENT_TOKENS = 3

# The concrete details a reply can only get from the record: identifiers,
# amounts, dates, weekdays. They are what separates a claim the detector
# merely could not score from one that asserts something the evidence
# does not contain — "the refund status shows none" against a line that
# says exactly that, versus "your package will arrive on Tuesday" against
# a line that mentions no day at all. The first is an abstention worth
# tolerating; the second is a fabrication wearing the same score.
_SPECIFIC = re.compile(
    r"\d[\d,.:/-]*"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)


# An amount in prose: "$210", "$1,234.56", or a bare "210.00". Money is a
# shape, not vocabulary, so it is code; what counts as an outcome is not.
MONEY = re.compile(r"(?<!\w)(?:\$\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*\.\d{2})(?!\w)")


def specifics(text: str) -> set[str]:
    """The identifiers, amounts, dates and weekdays a text names, normalised for comparison."""
    return {value.casefold().rstrip(".,:") for value in _SPECIFIC.findall(text)} - {""}


class ClaimClassifier:
    """Decides whether a claim reports a fact about this case, states a rule, or is only manner.

    A fact about the case names an order or an amount, or states an
    outcome as done. Anything that promises ("will", "once", "'ll") states
    a rule instead: it needs no tool to have run, and no tool result could
    show it. What opens as courtesy and states no outcome is conversation.
    """

    def __init__(self) -> None:
        settings = get_settings().hallucination_protection
        self._identifier = re.compile(settings.record_identifier_pattern, re.IGNORECASE)
        self._outcome = re.compile(settings.outcome_pattern, re.IGNORECASE)
        self._promise = re.compile(settings.promissory_pattern, re.IGNORECASE)
        self._openers = [
            re.compile(pattern, re.IGNORECASE) for pattern in settings.conversational_openers
        ]

    def classify(self, claim: str) -> ClaimClass:
        """The class that decides which tiers may ground this claim."""
        if self._is_record(claim):
            return "record"
        if self._is_conversation(claim):
            return "conversational"
        return "rule"

    def _is_record(self, claim: str) -> bool:
        if self._identifier.search(claim) or MONEY.search(claim):
            return True
        return bool(self._outcome.search(claim)) and not self._promise.search(claim)

    def _is_conversation(self, claim: str) -> bool:
        text = claim.strip().casefold()
        opens = any(opener.match(text) for opener in self._openers)
        return (opens or text.endswith("?")) and not self._outcome.search(text)


class ClaimDecomposer:
    """Breaks a generation into atomic, separately checkable claims."""

    def __init__(self, classifier: ClaimClassifier | None = None) -> None:
        self._classifier = classifier or ClaimClassifier()
        self._rule_phrases = [
            (re.compile(phrase.pattern, re.IGNORECASE), phrase)
            for phrase in get_settings().hallucination_protection.rule_phrases
        ]

    def decompose(self, generation: str) -> list[AtomicClaim]:
        """Split on sentence boundaries, then on joining conjunctions, then by class."""
        pieces = [
            (index, text, forced)
            for index, sentence in enumerate(split_sentences(generation))
            for part in _with_subjects(_clauses(sentence))
            for text, forced in self._carve(part.strip().rstrip(".;"))
            if text
        ]
        return [
            self._claim(text, position, forced, index)
            for position, (index, text, forced) in enumerate(pieces)
        ]

    def _claim(
        self, text: str, position: int, forced: ClaimClass | None, sentence: int
    ) -> AtomicClaim:
        claim_class = forced or self._classifier.classify(text)
        return AtomicClaim(
            text=text,
            position=position,
            sentence=sentence,
            claim_class=claim_class,
            is_verifiable=claim_class != "conversational"
            and len(tokenize(text)) >= _MIN_CONTENT_TOKENS,
        )

    def _carve(self, text: str) -> list[tuple[str, ClaimClass | None]]:
        """The claim, then each rule it states in passing.

        "$18.70 has been refunded to your original payment method" is an
        outcome and a rule. Checked whole, the outcome is neutral against
        the rule's evidence and the rule against the outcome's, so
        neither is ever supported. Split, each meets what can ground it.
        """
        pieces: list[tuple[str, ClaimClass | None]] = []
        for pattern, phrase in self._rule_phrases:
            if pattern.search(text) and pattern.sub("", text).strip():
                text = pattern.sub("", text).strip(" ,")
                pieces.append((phrase.statement, "rule"))
        return [(text, None), *pieces]


def split_sentences(text: str) -> list[str]:
    """A text's sentences, split exactly as claims are split into them."""
    return _SENTENCE.split(text.strip())


def _clauses(sentence: str) -> list[str]:
    """Split one sentence at the conjunctions that join clauses, and only those.

    "and" also joins the items of a list. Split there and "orders 10876
    and 10854" becomes two fragments, each checked on its own — the first
    was refuted by the record for the second, and the fragment that
    survived was all the customer saw. So a conjunction inside brackets,
    or between two bare specifics, joins items, not claims.
    """
    parts: list[str] = []
    start = 0
    for match in _CONJUNCTION.finditer(sentence):
        if _joins_items(sentence, match.start(), match.end()):
            continue
        parts.append(sentence[start : match.start()])
        start = match.end()
    parts.append(sentence[start:])
    return parts


def _with_subjects(parts: list[str]) -> list[str]:
    """Give each clause that lost its subject to a split the subject of the one before."""
    carried: list[str] = []
    for part in parts:
        text = part.strip()
        if carried and _AUXILIARY.match(text):
            subject = _subject(carried[-1])
            text = f"{subject} {text}" if subject else text
        elif carried and _PARTICIPLE.match(text):
            lead = _subject_and_verb_group(carried[-1])
            text = f"{lead} {text}" if lead else text
        carried.append(text)
    return carried


def _subject(clause: str) -> str:
    """The words of a clause before its first auxiliary, else its opening pronoun, else ""."""
    found = re.search(r"\b" + _AUXILIARY.pattern, clause, re.IGNORECASE)
    if found:
        return clause[: found.start()].strip()
    pronoun = _PRONOUN.match(clause)
    return pronoun.group(0) if pronoun else ""


def _subject_and_verb_group(clause: str) -> str:
    """A clause up to the end of its auxiliaries: "Order 10988 has already been"."""
    found = re.search(r"\b" + _AUXILIARY.pattern, clause, re.IGNORECASE)
    if not found or not clause[: found.start()].strip():
        return ""
    words = clause[found.start() :].split()
    group = []
    for word in words:
        if word.casefold() not in _VERB_GROUP:
            break
        group.append(word)
    return f"{clause[: found.start()].strip()} {' '.join(group)}"


def _joins_items(sentence: str, begin: int, end: int) -> bool:
    """Whether the conjunction at [begin, end) sits in brackets or between two specifics."""
    before = sentence[:begin]
    depth = sum(before.count(c) for c in _OPENERS) - sum(before.count(c) for c in _CLOSERS)
    if depth > 0:
        return True
    left, right = before.split()[-1:], sentence[end:].split()[:1]
    return bool(left and right) and all(
        _SPECIFIC.fullmatch(word.strip(_ITEM_PUNCTUATION)) for word in (*left, *right)
    )
