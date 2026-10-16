"""Hallucination detection, diagnosis and trust calibration (LLD §10)."""

from agent_platform.hallucination_check.checker import (
    ClaimResolver,
    DetectionLadder,
    GroundingReasoningDiagnoser,
    HallucinationDetector,
    TrustCalibrationTracker,
)
from agent_platform.hallucination_check.claims import (
    ClaimClassifier,
    ClaimDecomposer,
    plain_text,
    sentence_pieces,
    split_sentences,
)
from agent_platform.hallucination_check.direct import (
    AMOUNT_MISMATCH,
    IDENTIFIER_MISMATCH,
    NO_RECORD,
    NOT_EXECUTED,
    NOT_EXECUTED_MARKER,
    RECORD_STATE_MISMATCH,
)
from agent_platform.hallucination_check.evidence import (
    TOOL_RESULT_KIND,
    Observation,
    observation_facts,
    passages,
    tool_evidence,
)
from agent_platform.hallucination_check.resolver import (
    OBSERVATION_SEQUENCE,
    PASSAGE,
    STATE_CHANGING,
    GroundingResolver,
)
from agent_platform.hallucination_check.types import (
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
    "PASSAGE",
    "STATE_CHANGING",
    "IDENTIFIER_MISMATCH",
    "RECORD_STATE_MISMATCH",
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
    "passages",
    "plain_text",
    "sentence_pieces",
    "split_sentences",
    "tool_evidence",
]
