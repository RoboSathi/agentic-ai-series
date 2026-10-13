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
from decimal import Decimal
from functools import cache

from agent_platform.config import get_settings
from agent_platform.core.primitives.pii_tokens import bare_token, framed_token
from agent_platform.hallucination_check.types import AtomicClaim, ClaimClass
from agent_platform.llm.embeddings import tokenize

# Sentence boundaries, and the connectives that join two claims into one
# sentence — both produce claims that must be checked separately.
#
# The semicolon belongs here for the same reason the full stop does. Left
# out, "the refund has not been initiated; the status shows none" stayed a
# single claim, and its two halves were scored against evidence that
# matched only one of them — so a sentence half of which was verbatim
# supported scored as entirely unsupported.
_SENTENCE = re.compile(r"[ \t]*\n\s*|(?<=[.!?;])(?<![\s(][A-Z]\.)[ \t]+")
# A line break ends a claim too (checked first, so "policy.  \n- next" keeps its break), and
# an initial does not end one ("Governor Lisa D. Cook" is one name). A model writes lists and labelled lines
# ("- **Humidity**: 70 %") with no full stop, and a whole bulleted answer read
# as one claim is one premise no evidence line can entail.
#
# Markdown is layout, not content: a bullet, a heading marker or emphasis is
# taken off before a claim is checked, so "**Wind**: 2.8 km/h" is read as
# "Wind: 2.8 km/h".
_MARKDOWN_LEAD = re.compile(r"^\s*(?:#{1,6}\s+|[-*•+]\s+|\d+[.)]\s+|>\s*)")
_MARKDOWN_MARKS = re.compile(r"\*\*|__|`")
# How a line that introduces a list ends.
_LEAD_IN = ":"
# How a question ends: it asks, so it asserts nothing.
_QUESTION = "?"


# Causal connectives split too. "Queued for review because the amount
# exceeds the limit" is two claims from two tiers — the outcome from a tool
# result, the reason from the policy — and no single premise entails both.
# "Totals $117.23, so I cannot issue a $1 refund" likewise: joined, NLI
# read the pair as contradicting "a refund is always for the full total".
# Which connectives split is configuration (`hallucination_check.claim_conjunctions`):
# a support reply joins an outcome and its reason, a news summary joins items of one fact.
def conjunction_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    return _conjunction(words)


@cache
def _conjunction(words: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(word) for word in words) or "(?!)"
    return re.compile(rf",?\s+\b(?:{alternatives})\b\s+")


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

# The concrete details a reply can only get from the record: identifiers,
# amounts, dates, weekdays. They are what separates a claim the detector
# merely could not score from one that asserts something the evidence
# does not contain — "the refund status shows none" against a line that
# says exactly that, versus "your package will arrive on Tuesday" against
# a line that mentions no day at all. The first is an abstention worth
# tolerating; the second is a fabrication wearing the same score.
# A month is a date only as one: a full name, or a name beside a day number
# ("Mar 3", "3 Nov", "May 3"). A word that merely *starts* like one is not:
# Margaret, Augustine, Decker, and the verb "may".
_MONTH = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_FULL_MONTH = r"january|february|march|april|june|july|august|september|october|november|december"
_SPECIFIC = re.compile(
    rf"\b(?:{_MONTH})\.?\s+\d{{1,2}}\b"
    rf"|\b\d{{1,2}}\s+(?:{_MONTH})\b"
    r"|\d[\d,.:/-]*"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
    rf"|\b(?:{_FULL_MONTH})\b",
    re.IGNORECASE,
)


# An amount in prose: "$210", "$1,234.56", or a bare "210.00". Money is a
# shape, not vocabulary, so it is code; what counts as an outcome is not.
#
# A figure with a scale ("$1.2 M", "$1.2M", "$1.2 million", "₹4 lakh") is a size, not an exact
# amount: read as money it became "$1.20", an amount no evidence held. The scale words are
# configuration (`hallucination_check.money_scales`).
def money_pattern() -> re.Pattern[str]:
    """Exact amounts in prose, as this deployment's scale words leave them."""
    return _money(tuple(get_settings().hallucination_check.money_scales))


@cache
def _money(scales: tuple[str, ...]) -> re.Pattern[str]:
    scale = "|".join(re.escape(word) for word in sorted(scales, key=len, reverse=True))
    unscaled = rf"(?!\s*(?i:{scale})\b)" if scale else ""
    return re.compile(
        r"(?<!\w)(?:\$\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*\.\d{2})(?!\.?\w)" + unscaled
    )


_PLAIN_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_MONTH_FIRST = re.compile(rf"(?P<month>{_MONTH})\.?\s+(?P<day>\d{{1,2}})", re.IGNORECASE)
_DAY_FIRST = re.compile(rf"(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH})", re.IGNORECASE)
# How many letters of a month name identify it ("Sept", "September" and "Sep" are one month).
_MONTH_KEY_LETTERS = 3
# The month keys in calendar order, so an ISO date's month number reads as a month.
_MONTH_KEYS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_ISO_DATE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")


def _normalised(value: str) -> str:
    """One written form for a thing a text names; a date is the same date either way round.

    "30 September" and "September 30" name one day, so a claim written one way is not
    "unsourced" against evidence written the other.
    """
    for pattern in (_MONTH_FIRST, _DAY_FIRST):
        found = pattern.fullmatch(value)
        if found:
            month = found.group("month").casefold()[:_MONTH_KEY_LETTERS]
            return f"{int(found.group('day'))} {month}"
    plain = value.casefold().rstrip(".,:")
    if _PLAIN_NUMBER.fullmatch(plain):
        # "0.0" and "0", "26.40" and "26.4", "1,200" and "1200" are the same figure.
        return format(Decimal(plain.replace(",", "")).normalize(), "f")
    return plain


def _without_tokens(text: str) -> str:
    """`text` with each PII token blanked to spaces, so offsets into it stay valid.

    A token is an opaque handle, not a figure the reply states: the digits inside
    `[PERSON:3f9a1c2b7e10]` or a bare `f0ca051ccfa3` are not numbers to be sourced.
    """
    blank = lambda match: " " * len(match.group())  # noqa: E731
    return bare_token().sub(blank, framed_token().sub(blank, text))


def specific_spans(text: str) -> list[tuple[int, int]]:
    """Where each identifier, amount, date or weekday sits in `text`, as (start, end)."""
    return [match.span() for match in _SPECIFIC.finditer(_without_tokens(text))]


def specifics(text: str) -> set[str]:
    """The identifiers, amounts, dates and weekdays a text names, normalised for comparison."""
    found = _SPECIFIC.findall(_without_tokens(text))
    return {form for value in found for form in _forms(value)} - {""}


def _forms(value: str) -> set[str]:
    """Every way a claim may write what `value` names.

    An ISO date ("2026-10-02", as an API or a page's metadata writes it) is also the day a
    sentence names ("2 October") and its year ("2026"): evidence that says 2026-10-02 sources a
    claim about "2 October 2026".
    """
    iso = _ISO_DATE.match(value)
    if iso is None or not 1 <= int(iso.group("month")) <= len(_MONTH_KEYS):
        return {_normalised(value)}
    day, month = int(iso.group("day")), _MONTH_KEYS[int(iso.group("month")) - 1]
    return {_normalised(value), f"{day} {month}", iso.group("year")}


class ClaimClassifier:
    """Decides whether a claim reports a fact about this case, states a rule, or is only manner.

    A fact about the case names an order or an amount, or states an
    outcome as done. Anything that promises ("will", "once", "'ll") states
    a rule instead: it needs no tool to have run, and no tool result could
    show it. What opens as courtesy and states no outcome is conversation.
    """

    def __init__(self) -> None:
        settings = get_settings().hallucination_check
        self._identifier = re.compile(settings.record_identifier_pattern, re.IGNORECASE)
        self._outcome = re.compile(settings.outcome_pattern, re.IGNORECASE)
        self._promise = re.compile(settings.promissory_pattern, re.IGNORECASE)
        self._openers = [
            re.compile(pattern, re.IGNORECASE) for pattern in settings.conversational_openers
        ]
        self._offer = re.compile(settings.offer_pattern, re.IGNORECASE)

    def classify(self, claim: str, asked: bool = False) -> ClaimClass:
        """The class that decides which tiers may ground this claim.

        A question asserts nothing, even one that names a record: "How can I help with
        order 10633?" was read as a record claim and refuted by the order's state. `asked`
        says the claim was cut from a question, whose clauses lose the question mark.
        """
        if self._offer.match(claim.strip()) or self._asks(claim, asked):
            return "conversational"
        if self._is_record(claim):
            return "record"
        if self._is_conversation(claim):
            return "conversational"
        return "rule"

    def _is_record(self, claim: str) -> bool:
        if self._identifier.search(claim) or money_pattern().search(claim):
            return True
        return bool(self._outcome.search(claim)) and not self._promise.search(claim)

    def promises(self, claim: str) -> bool:
        """Whether the claim promises ("will", "once", "'ll") rather than reports."""
        return bool(self._promise.search(claim))

    def _asks(self, claim: str, asked: bool) -> bool:
        """Whether the claim is a question that states no outcome ("Did you know it was refunded?" does)."""
        text = claim.strip()
        return (asked or text.endswith(_QUESTION)) and not self._outcome.search(text)

    def _is_conversation(self, claim: str) -> bool:
        text = claim.strip().casefold()
        return any(opener.match(text) for opener in self._openers) and not self._outcome.search(
            text
        )


class ClaimDecomposer:
    """Breaks a generation into atomic, separately checkable claims."""

    def __init__(self, classifier: ClaimClassifier | None = None) -> None:
        self._classifier = classifier or ClaimClassifier()
        settings = get_settings().hallucination_check
        self._rule_phrases = [
            (re.compile(phrase.pattern, re.IGNORECASE), phrase) for phrase in settings.rule_phrases
        ]
        self._conjunction = conjunction_pattern(tuple(settings.claim_conjunctions))
        # Fewer content words than this and a fragment is not a checkable claim.
        self._min_tokens = settings.claim_min_content_tokens

    def decompose(self, generation: str) -> list[AtomicClaim]:
        """Split on sentence boundaries, then on joining conjunctions, then by class."""
        pieces = [
            (index, text, forced, plain.endswith(_QUESTION))
            for index, sentence in enumerate(split_sentences(generation))
            for plain in [plain_text(sentence)]
            for part in _with_subjects(_clauses(plain, self._conjunction))
            for text, forced in self._carve(part.strip().rstrip(".;"))
            if text and not _is_lead_in(text)
        ]
        return [
            self._claim(text, position, forced, index, asked)
            for position, (index, text, forced, asked) in enumerate(pieces)
        ]

    def _claim(
        self, text: str, position: int, forced: ClaimClass | None, sentence: int, asked: bool
    ) -> AtomicClaim:
        claim_class = forced or self._classifier.classify(text, asked)
        return AtomicClaim(
            text=text,
            position=position,
            sentence=sentence,
            claim_class=claim_class,
            is_verifiable=claim_class != "conversational"
            and len(tokenize(text)) >= self._min_tokens,
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
                text = re.sub(r"\s+", " ", pattern.sub("", text)).strip(" ,")
                pieces.append((phrase.statement, "rule"))
        return [(text, None), *pieces]


def _is_lead_in(text: str) -> bool:
    """A line that introduces what follows ("Tomorrow in Pune:") labels a list; it claims nothing."""
    return text.rstrip().endswith(_LEAD_IN)


def split_sentences(text: str) -> list[str]:
    """A text's sentences, split exactly as claims are split into them."""
    return [sentence for sentence, _ in sentence_pieces(text)]


def sentence_pieces(text: str) -> list[tuple[str, str]]:
    """Each sentence with the separator that follows it, so a text can be rebuilt as written."""
    stripped = text.strip()
    pieces: list[tuple[str, str]] = []
    start = 0
    for match in _SENTENCE.finditer(stripped):
        pieces.append((stripped[start : match.start()], match.group(0)))
        start = match.end()
    pieces.append((stripped[start:], ""))
    return pieces


def plain_text(sentence: str) -> str:
    """A sentence without its markdown: no bullet or heading marker, no emphasis."""
    return _MARKDOWN_MARKS.sub("", _MARKDOWN_LEAD.sub("", sentence)).strip()


def _clauses(sentence: str, conjunction: re.Pattern[str]) -> list[str]:
    """Split one sentence at the conjunctions that join clauses, and only those.

    "and" also joins the items of a list. Split there and "orders 10876
    and 10854" becomes two fragments, each checked on its own — the first
    was refuted by the record for the second, and the fragment that
    survived was all the customer saw. So a conjunction inside brackets,
    or between two bare specifics, joins items, not claims.
    """
    parts: list[str] = []
    start = 0
    for match in conjunction.finditer(sentence):
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
