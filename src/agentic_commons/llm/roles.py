"""Role-based model routing: which model plays which part.

An agent that plans, extracts, writes and judges wants different models
for different jobs — a cheap local model for bulk extraction, a strong
one for synthesis, and a *different* one for judging, so the judge does
not share the writer's blind spots. The roles are the agent's vocabulary
(`llm.roles` in its overlay); this module only resolves a role name to a
client and answers "do these two roles resolve to the same model?".

This is not the cascade. The cascade (`cascade.py`) asks which tier is
good enough for one request; routing decides who is asked at all.
"""

from collections.abc import Mapping

from agentic_commons.config.settings import ProviderName, RoleModelSettings
from agentic_commons.llm.chain import ClientFactory, ModelChainBuilder, RequestExtras
from agentic_commons.llm.client import LLMClient, create_pinned_llm_client
from agentic_commons.llm.observer import ModelCallObserver


class RoleNotConfiguredError(KeyError):
    """A component asked for a role the agent's configuration does not map."""


class RoleCollisionError(ValueError):
    """Two roles that must be independent resolve to the same model."""


def _pinned(provider: ProviderName, model: str) -> LLMClient:
    return create_pinned_llm_client(provider, model)


class RoleModelRouter:
    """Maps role names to protected clients: the role's model, then its backups.

    One adapter and one resilience stack per distinct model, shared by every role on it
    (`ModelChainBuilder`); one chain per role, named after the role on its backup cards.
    """

    def __init__(
        self,
        roles: Mapping[str, RoleModelSettings],
        factory: ClientFactory = _pinned,
        observer: ModelCallObserver | None = None,
    ) -> None:
        self._roles = dict(roles)
        self._builder = ModelChainBuilder(factory, observer)
        self._clients: dict[str, LLMClient] = {}

    @property
    def roles(self) -> list[str]:
        return sorted(self._roles)

    def model_key(self, role: str) -> str:
        """`provider:model` for a role's primary model — what "the same model" means."""
        return self._spec(role).key

    def client_for(self, role: str) -> LLMClient:
        """The role's protected client, built once."""
        if role not in self._clients:
            spec = self._spec(role)
            self._clients[role] = self._builder.build(spec, spec, label=role)
        return self._clients[role]

    def require_distinct(self, first: str, second: str) -> None:
        """Raise unless two roles resolve to different models (checked at start-up)."""
        if self.model_key(first) == self.model_key(second):
            raise RoleCollisionError(
                f"roles '{first}' and '{second}' must use different models; "
                f"both resolve to {self.model_key(first)}"
            )

    def _spec(self, role: str) -> RoleModelSettings:
        try:
            return self._roles[role]
        except KeyError:
            raise RoleNotConfiguredError(
                f"no model configured for role '{role}' (llm.roles)"
            ) from None


__all__ = ["RequestExtras", "RoleCollisionError", "RoleModelRouter", "RoleNotConfiguredError"]
