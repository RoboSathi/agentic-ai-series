"""The Authority Hierarchy (E12/E13, LLD §11).

Ordered strongest to weakest:

    provider > organization > product > workspace > user_task >
    runtime_reminders > tool_observations > untrusted_content

Only the top ranks can grant or expand authority. Everything below them
is *input to be acted on*, never *instruction to be obeyed* — and that
single rule is why a prompt injected into a retrieved document cannot
authorize a refund. It arrives as `untrusted_content`, the weakest rank
there is, however imperative its grammar.

This is also where Authority ≠ Agency becomes concrete. An agent can have
the agency to attempt anything; what it has the authority to do is
decided here, by where the instruction came from.
"""

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import AuthorityContext
from agentic_commons.logging import get_logger


class AuthorityHierarchyResolver:
    """Resolves a source to its rank, and whether that rank may grant."""

    def __init__(self) -> None:
        settings = get_settings().gatekeeper
        self._hierarchy = settings.authority_hierarchy
        self._max_granting_rank = settings.authority_grant_max_rank
        self._ranks = {source: index + 1 for index, source in enumerate(self._hierarchy)}
        self._logger = get_logger("gatekeeper")

    @property
    def hierarchy(self) -> list[str]:
        """The configured hierarchy, strongest first."""
        return list(self._hierarchy)

    def rank_of(self, source: str) -> int:
        """The rank of a source.

        An unrecognised source gets the weakest rank, not a middling one:
        anything the hierarchy has not explicitly placed is, by default,
        untrusted (Saltzer & Schroeder's fail-safe defaults).
        """
        rank = self._ranks.get(source)
        if rank is None:
            self._logger.warning("Unknown authority source; treating as weakest", source=source)
            return len(self._hierarchy)
        return rank

    def can_grant(self, source: str) -> bool:
        """Whether a source may grant or expand authority at all."""
        return self.rank_of(source) <= self._max_granting_rank

    def outranks(self, source: str, other: str) -> bool:
        """Whether one source's instructions take precedence over another's."""
        return self.rank_of(source) < self.rank_of(other)

    def resolve(self, context: AuthorityContext) -> AuthorityContext:
        """Normalize a context so its rank always matches its named source.

        A caller-supplied rank is never trusted over the source name: a
        component that could assert its own rank could assert rank 1, and
        the hierarchy would be decorative.
        """
        return context.model_copy(update={"granted_by_rank": self.rank_of(context.granted_by)})

    def may_authorize(self, context: AuthorityContext) -> tuple[bool, str]:
        """Whether this context can authorize an action, and why not if not."""
        resolved = self.resolve(context)
        if resolved.granted_by_rank <= self._max_granting_rank:
            return True, f"'{resolved.granted_by}' is rank {resolved.granted_by_rank}"
        return False, (
            f"'{resolved.granted_by}' is rank {resolved.granted_by_rank}; only ranks 1-"
            f"{self._max_granting_rank} may grant or expand authority"
        )
