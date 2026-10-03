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

from agent_platform.config import get_settings
from agent_platform.gatekeeper.types import AuthorityContext
from agent_platform.logging import get_logger


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

    @property
    def max_granting_rank(self) -> int:
        """Highest (numerically weakest) rank allowed to grant authority."""
        return self._max_granting_rank

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
        """Return the trusted context unchanged; rank is derived from source when needed."""
        return context

    def may_authorize(self, context: AuthorityContext) -> tuple[bool, str]:
        """Whether this context can authorize an action, using the source-derived rank."""
        rank = self.rank_of(context.granted_by)
        if rank <= self._max_granting_rank:
            return True, f"'{context.granted_by}' is rank {rank}"
        return False, (
            f"'{context.granted_by}' is rank {rank}; only ranks 1-"
            f"{self._max_granting_rank} may grant or expand authority"
        )
