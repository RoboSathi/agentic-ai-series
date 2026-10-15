"""What is deployed, versioned independently (FR-48).

Nine things in this system change on nine different schedules: the
prompt, the tool schema, the policy, the evaluator, the model, the
retrieval index, the memory schema, the guardrail rules and the PII
rules. Treating them as one "release" is what makes an incident
un-diagnosable — the question after a regression is always *which* of
them moved, and a single version number cannot answer it.

So each carries its own version, and a decision records the tuple it was
made under (FR-33). A span that names a policy version and a model
version can be replayed; one that names "v2.1.4" cannot.

The Gatekeeper's live authority path is deliberately excluded from
canary rollout. A canary means two versions are live at once, and two
authority policies live at once means the same action is allowed and
denied depending on which instance answered — an inconsistency that is
indistinguishable from an exploit and impossible to audit afterwards.
Authority changes go through a full cutover with the old version
retired, or they do not go.
"""

from pydantic import BaseModel

from agent_platform.config.settings import get_settings
from agent_platform.logging import get_logger, instrument

# The component whose version must never be split across a fleet. Named
# here rather than described in a comment so that `canaryable()` is a
# rule the code applies, not advice somebody reads.
AUTHORITY_COMPONENT = "policy"


class ComponentVersion(BaseModel):
    """One independently versioned part of the deployment."""

    component: str
    version: str
    # Whether two versions of this may be live at once. False for
    # anything on the authority path.
    canaryable: bool = True
    source: str = ""


@instrument("config")
class VersionManifest:
    """Every component version, and what may be rolled out gradually.

    Read from configuration and from the artifacts themselves rather than
    hand-maintained: a version list somebody updates by hand is a version
    list that is wrong exactly when it matters.
    """

    def __init__(self) -> None:
        self._logger = get_logger("config")

    def components(self) -> list[ComponentVersion]:
        """The deployed version of each independently versioned part."""
        settings = get_settings()
        provider = settings.llm.provider_settings()
        encoders = settings.llm.encoders
        return [
            ComponentVersion(
                component="prompt",
                version=settings.audit.prompt_version,
                source=settings.audit.prompt_source,
            ),
            ComponentVersion(
                component="tool_schema",
                version=settings.audit.tool_schema_version,
                source="tools registry",
            ),
            ComponentVersion(
                component=AUTHORITY_COMPONENT,
                version=settings.audit.policy_version,
                # The one exception, and the reason this field exists.
                canaryable=False,
                source=settings.audit.policy_source,
            ),
            ComponentVersion(
                component="evaluator",
                version=settings.evals.version,
                # The evaluator is the judge model behind the graders.
                source=settings.judge.model,
            ),
            ComponentVersion(
                component="model", version=provider.model, source=settings.llm.provider
            ),
            ComponentVersion(
                component="retrieval_index",
                version=settings.context.retrieval_mode,
                source="context retrieval",
            ),
            ComponentVersion(
                component="memory_schema",
                version=settings.memory.schema_version,
                source="memory stores",
            ),
            ComponentVersion(
                component="guardrail",
                version=settings.guardrails.mode,
                source="guardrails",
            ),
            ComponentVersion(
                component="pii_rules",
                version=str(len(settings.security.pii.patterns)) + "-patterns",
                source="security/pii",
            ),
            ComponentVersion(
                component="encoders",
                version=f"{encoders.nli_model} / {encoders.reranker_model}",
                source="llm/local_nli + llm/local_reranker",
            ),
        ]

    def canaryable(self, component: str) -> bool:
        """Whether this component may run two versions at once.

        Everything except the authority path. Two live authority policies
        mean the same action is allowed and denied depending on which
        instance answered, which is indistinguishable from an exploit in
        the audit trail.
        """
        return component != AUTHORITY_COMPONENT

    def decision_tuple(self) -> dict[str, str]:
        """The (model, policy, data, prompt) tuple a decision is made under.

        Attached to audit records and spans so a decision can be replayed
        against exactly what produced it — the lineage requirement
        (FR-33) that a single release number cannot satisfy.
        """
        settings = get_settings()
        return {
            "model_version": settings.llm.provider_settings().model,
            "policy_version": settings.audit.policy_version,
            "data_version": settings.audit.data_version,
            "prompt_version": settings.audit.prompt_version,
        }
