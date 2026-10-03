"""Hallucination detection, diagnosis and trust calibration (LLD §10)."""

from agent_platform.hallucination_protection.checker import (
    ClaimResolver,
    DetectionLadder,
    GroundingReasoningDiagnoser,
    HallucinationDetector,
    TrustCalibrationTracker,
)
from agent_platform.hallucination_protection.claims import (
    ClaimClassifier,
    ClaimDecomposer,
    plain_text,
    sentence_pieces,
    split_sentences,
)
from agent_platform.hallucination_protection.direct import (
    AMOUNT_MISMATCH,
    NO_RECORD,
    NOT_EXECUTED,
    NOT_EXECUTED_MARKER,
    ORDER_MISMATCH,
    REFUND_STATE_MISMATCH,
)
from agent_platform.hallucination_protection.evidence import (
    TOOL_RESULT_KIND,
    Observation,
    observation_facts,
    tool_evidence,
)
from agent_platform.hallucination_protection.resolver import (
    OBSERVATION_SEQUENCE,
    STATE_CHANGING,
    GroundingResolver,
)
from agent_platform.hallucination_protection.types import (
    AtomicClaim,
    ClaimClass,
    DetectionVerdict,
    EvidenceCheck,
    FailureKind,
    GenerationAssessment,
    TrustMeasurement,
    TrustReport,
)

__all__ = [
    "AMOUNT_MISMATCH",
    "NOT_EXECUTED",
    "NOT_EXECUTED_MARKER",
    "NO_RECORD",
    "OBSERVATION_SEQUENCE",
    "STATE_CHANGING",
    "ORDER_MISMATCH",
    "REFUND_STATE_MISMATCH",
    "AtomicClaim",
    "ClaimClass",
    "ClaimClassifier",
    "ClaimDecomposer",
    "ClaimResolver",
    "DetectionLadder",
    "EvidenceCheck",
    "GroundingResolver",
    "DetectionVerdict",
    "FailureKind",
    "GenerationAssessment",
    "GroundingReasoningDiagnoser",
    "HallucinationDetector",
    "TrustCalibrationTracker",
    "TrustMeasurement",
    "TrustReport",
    "TOOL_RESULT_KIND",
    "Observation",
    "observation_facts",
    "plain_text",
    "sentence_pieces",
    "split_sentences",
    "tool_evidence",
]
