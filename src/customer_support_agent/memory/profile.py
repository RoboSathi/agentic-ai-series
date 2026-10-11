"""The support agent's memory profile: the default, with its own session summaries.

Only episodic memory is customised. A support session is summarised from
its exact tool outcomes (lookups condensed to "looked up …"), plus — when
`memory.dream.summarize_intent` is on — one locally generated sentence of
what the customer wanted, kept only if NLI finds the conversation entails
it. Facts come from the default pattern tier (customers say "my X is Y"),
and skills from the default recurring-signal learner, fed by this agent's
refusal wording (`memory.dream.refusal_markers`).
"""

from agent_platform.config import get_settings
from agent_platform.evals.entailment import create_entailment_model
from agent_platform.llm.local_summarizer import LocalDialogueSummarizer
from agent_platform.memory import MemoryProfile
from agent_platform.memory.dream import SessionSummarizer


def support_memory_profile() -> MemoryProfile:
    """Support sessions summarised with a checked sentence of intent."""
    settings = get_settings()
    dream = settings.memory.dream
    summarizer = SessionSummarizer(
        entailment=create_entailment_model(),
        threshold=settings.hallucination_check.nli_entailment_threshold,
        outcome_chars=dream.outcome_chars,
        summarizer=LocalDialogueSummarizer() if dream.summarize_intent else None,
        lookup_tools=dream.lookup_tools,
        forbidden_words=dream.forbidden_words,
        min_word_ratio=dream.min_word_ratio,
    )
    return MemoryProfile.default().replace(episodic=summarizer)
