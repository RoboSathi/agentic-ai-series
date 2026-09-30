"""`/approve`: freeze Brief + Rubric + Budget into the Research Contract.

The contract is an immutable model hashed with the platform's canonical
hash. The hash is recorded in the audit trail at approval and re-checked
at the end of the run, so "the run quietly changed what it was asked to
do" is a detectable event rather than a matter of reading two transcripts.
"""

import uuid
from datetime import datetime

from agentic_commons.artifacts import FrozenArtifact
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import ScopeBrief

BRIEF_ARTIFACT = "brief"
RUBRIC_ARTIFACT = "rubric"


def freeze_contract(
    brief: ScopeBrief, rubric: Rubric, prefix: str, frozen_at: datetime
) -> Contract:
    """The approved contract, identified by a fresh id and its content hash.

    `frozen_at` is "now" for every recency window of the run — injected, so
    a replayed world with fixed publication dates stays reproducible.
    """
    contract = Contract.freeze(f"{prefix}-{uuid.uuid4().hex[:10]}", brief, rubric, brief.budget)
    return contract.model_copy(update={"frozen_at": frozen_at})


def contract_artifacts(contract: Contract) -> list[FrozenArtifact]:
    """The Brief and the Rubric as frozen artifacts, for the archive."""
    return [
        FrozenArtifact.freeze(BRIEF_ARTIFACT, contract.contract_id, contract.brief),
        FrozenArtifact.freeze(RUBRIC_ARTIFACT, contract.contract_id, contract.rubric),
    ]
