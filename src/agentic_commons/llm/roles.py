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

from collections.abc import Callable, Mapping

from agentic_commons.config.settings import ProviderName, RoleModelSettings
from agentic_commons.llm.client import LLMClient, create_pinned_llm_client

ClientFactory = Callable[[ProviderName, str], LLMClient]


class RoleNotConfiguredError(KeyError):
    """A component asked for a role the agent's configuration does not map."""


class RoleCollisionError(ValueError):
    """Two roles that must be independent resolve to the same model."""


def _pinned(provider: ProviderName, model: str) -> LLMClient:
    return create_pinned_llm_client(provider, model)


class RoleModelRouter:
    """Maps role names to clients, one client per distinct model."""

    def __init__(
        self, roles: Mapping[str, RoleModelSettings], factory: ClientFactory = _pinned
    ) -> None:
        self._roles = dict(roles)
        self._factory = factory
        self._clients: dict[str, LLMClient] = {}

    @property
    def roles(self) -> list[str]:
        return sorted(self._roles)

    def model_key(self, role: str) -> str:
        """`provider:model` for a role — what "the same model" means."""
        spec = self._spec(role)
        return f"{spec.provider}:{spec.model}"

    def client_for(self, role: str) -> LLMClient:
        """The client for a role, shared with every role on the same model."""
        key = self.model_key(role)
        if key not in self._clients:
            spec = self._spec(role)
            self._clients[key] = self._factory(spec.provider, spec.model)
        return self._clients[key]

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
