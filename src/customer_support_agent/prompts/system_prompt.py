"""Builds the customer support agent's system prompt.

The instructions below are deliberately explicit about when *not* to act.
An earlier version opened with "You are a customer support agent handling
refund requests", which framed every incoming message as a refund request
— so "who are you?" was answered by looking up an arbitrary order and
queuing it for approval. A prompt that only describes the happy path
invites the model to find one.

The prompt is not a control. The Gatekeeper is. But a prompt that pushes
the model toward unnecessary tool calls wastes money and fills the review
queue, which is worth fixing on its own terms.

What used to be one page of instructions here is now two kinds of file
(ADR 0010): the profile, `AGENT.md`, which every turn carries, and the
skills under `skills/`, of which a turn carries only the ones its
intent selects. This module adds the refund policy and owns the prompts
for the final, customer-facing turn.
"""

from collections.abc import Sequence

from agent_platform.config import get_settings
from agent_platform.core.primitives.money import Money
from agent_platform.llm.prompt import POLICY, PromptPart
from agent_platform.memory.procedural.skills import load_skill_documents, render_profile
from customer_support_agent.config import get_agent_settings
from customer_support_agent.prompts.policy import policy_text

# The prompt sections only this agent's requests carry (labels: `observability.context_card`).
REFUND_LIMIT = "refund_limit"
INTENT_FALLBACK = "intent_fallback"
REPLY_INSTRUCTION = "reply_instruction"

_REPLY_SYSTEM = """You are a customer support agent for an online store,
writing the message the customer will read.

Everything you may state has already been established. Write from what
you are given and add nothing to it: no amount, date, tracking number or
outcome that is not in front of you, and no policy explanation nobody
asked for.
"""


_REPLY_INSTRUCTION = """Now write the reply the customer will read.

Address them directly and answer only what they asked. One or two
sentences is usually enough.

Do not add information they did not ask for — no refund offers, no
related order details, no suggestions about what they might want next. Do
not narrate your reasoning, restate the policy, mention tool names, or
refer to "the user" in the third person; they are the person reading
this.
"""


def build_reply_system_prompt() -> str:
    """The system prompt for the final, customer-facing turn.

    Deliberately not `build_system_prompt`. That prompt exists to help
    the model *choose a tool*: it is most of a page of tool instructions
    and the full refund policy, and none of it bears on phrasing a
    sentence for somebody who is waiting. Reusing it meant the last call
    of every turn — the one that offers no tools at all — re-read the
    whole policy and every retrieved order to write two lines.
    """
    return _REPLY_SYSTEM


def build_reply_instruction() -> str:
    """The instruction used for the final, customer-facing turn.

    A separate instruction because the two turns want different things.
    The working prompt tells the model how to *decide*; reusing it for
    the last turn invites a model that reasons out loud to hand its
    deliberation to the customer — "The user wants a refund for order
    10003. According to policy, we need to confirm the amount…".
    """
    return _REPLY_INSTRUCTION


def system_prompt_parts(
    instructions: Sequence[PromptPart], refund_limit: Money | None, *, governed: bool
) -> list[PromptPart]:
    """The working prompt as named parts: the turn's instructions, the policy, the limit.

    The same text `build_system_prompt` joins, attributable part by part, for a request
    built with `PromptBuilder`.
    """
    policy = PromptPart(POLICY, policy_text().strip(), get_agent_settings().policy_path)
    limit = PromptPart(REFUND_LIMIT, refund_limit_notice(refund_limit, governed=governed).strip())
    return [*instructions, policy, limit]


def build_system_prompt(
    instructions: str | None = None, refund_limit: Money | None = None, *, governed: bool = True
) -> str:
    """The working system prompt: the turn's instructions, the refund policy, the limit.

    `instructions` is what `AgentInstructions.compose` rendered for this
    turn — the profile plus the skills its intent selected. Without
    it (a screen, a leak check, evidence for the grounding check) the
    prompt carries the profile alone: per-turn skills are not the
    rules a reply is judged against.

    The policy file carries its own `# Refund Policy` heading, so this
    template does not add a second one. It is also the *only* place the
    policy reaches the model: the assembled context window deliberately
    contributes no policy chunk, because a second verbatim copy teaches
    the model nothing and costs a third of the window.

    `refund_limit` is the DB's per-refund limit, read once when the agent
    is built. `governed` says whether the Gatekeeper was built. With it,
    the model calls `process_refund` whatever the total, and the
    Gatekeeper escalates a refund above the limit. Without it, the
    prompt forbids that call: the model holds to the rule on a plain
    request and can be talked past it, which is the point. A rule in the
    prompt is a request, and the Gatekeeper is the control.
    """
    given = instructions if instructions is not None else render_profile(get_settings().skills)
    notice = refund_limit_notice(refund_limit, governed=governed)
    return f"{given}\n\n{policy_text().strip()}\n{notice}"


def refund_limit_notice(refund_limit: Money | None, *, governed: bool) -> str:
    """The prompt's limit sentence for this Gatekeeper posture, or nothing without a limit."""
    if refund_limit is None:
        return ""
    settings = get_agent_settings()
    notices = settings.refund_limit_notice
    template = notices.governed if governed else notices.ungoverned
    notice = template.format(limit=refund_limit, contact=settings.support_contact)
    return f"\n{notice}\n"


def instruction_corpus() -> str:
    """Every instruction this agent could send a model, whatever a turn selects.

    The profile, every authored skill, the policy and the reply
    prompts: what a leak check compares output against, and what must
    never contain the refund limit.
    """
    skills = [document.body for document in load_skill_documents(get_settings().skills)]
    return "\n\n".join(
        [build_system_prompt(), *skills, build_reply_system_prompt(), build_reply_instruction()]
    )
