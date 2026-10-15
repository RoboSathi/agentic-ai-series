"""The NOLLM scoring cascade: claims resolved cheapest tier first (LLD §18).

Runtime infrastructure, not an eval: retrieval relevance, tool fitness,
grounding and memory consolidation all resolve claims here. A leaf package
— it imports configuration, embeddings and logging, nothing that imports it.
"""

from agent_platform.scoring.cascade import (
    NOLLM_TIERS,
    TIER_ORDER,
    CalibratedJudge,
    CalibratedJudgeScorer,
    DeterministicRuleScorer,
    EmbeddingSimilarityScorer,
    ExactAssertionScorer,
    HumanAdjudicationScorer,
    NliEntailmentScorer,
    NollmScoringCascade,
    NollmScoringCascadeImpl,
    Scorer,
    create_scoring_cascade,
    lexical_overlap,
)
from agent_platform.scoring.entailment import (
    EntailmentModel,
    EntailmentResult,
    LexicalEntailmentModel,
    create_entailment_model,
)
from agent_platform.scoring.types import Claim, ScorerTier, Verdict, VerdictLabel

__all__ = [
    "NOLLM_TIERS",
    "TIER_ORDER",
    "CalibratedJudge",
    "CalibratedJudgeScorer",
    "Claim",
    "DeterministicRuleScorer",
    "EmbeddingSimilarityScorer",
    "EntailmentModel",
    "EntailmentResult",
    "ExactAssertionScorer",
    "HumanAdjudicationScorer",
    "LexicalEntailmentModel",
    "NliEntailmentScorer",
    "NollmScoringCascade",
    "NollmScoringCascadeImpl",
    "Scorer",
    "ScorerTier",
    "Verdict",
    "VerdictLabel",
    "create_entailment_model",
    "create_scoring_cascade",
    "lexical_overlap",
]
