"""Every model the system uses, read from configuration, one entry per distinct model.

`make model-check` asks this for the whole system: the platform's own configuration and
each agent's layer over it. A model named in several places (lightning as one agent's chat
model and another's planner, super-120b as a chat fallback and the judge's backup) is one
entry, carrying every use — so it is probed once and the report says what depends on it.

A use is *primary* when something calls the model first (an agent's chat model, a role, the
judge, an enabled guard); a fallback is only called when that one does not answer. A
failing primary fails the check; a failing fallback is reported.
"""

from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.config import AppConfig
from agent_platform.config.settings import ModelChainSettings, ModelRef, ProviderName
from agent_platform.llm.embeddings import HOSTED

ModelKind = Literal["chat", "embedding", "guard"]
GUARDS = ("jailbreak", "content_safety")


class ModelUse(BaseModel):
    """One model in one place: what it is used for, in which layer, first or as a fallback."""

    kind: ModelKind
    provider: str
    model: str
    use: str
    layer: str
    primary: bool


class InventoryEntry(BaseModel):
    """One distinct model and every use of it across the system."""

    kind: ModelKind
    provider: str
    model: str
    uses: list[ModelUse] = Field(default_factory=list)

    @property
    def required(self) -> bool:
        """Something calls this model first: if it does not answer, that thing breaks."""
        return any(use.primary for use in self.uses)

    def described(self) -> list[str]:
        """Each use once, with the layers that use it: `agent chat (platform, csa, ops)`."""
        return grouped([(use.use, use.layer) for use in self.uses])


def grouped(pairs: list[tuple[str, str]]) -> list[str]:
    """`use (layer, layer)` per distinct use, uses and layers in the order found."""
    layers: dict[str, list[str]] = {}
    for use, layer in pairs:
        found = layers.setdefault(use, [])
        if layer not in found:
            found.append(layer)
    return [f"{use} ({', '.join(names)})" for use, names in layers.items()]


def uses_in(config: AppConfig, layer: str) -> list[ModelUse]:
    """Every hosted model one configuration layer names, with what it is used for."""
    words = config.llm.check.uses
    found: list[ModelUse] = []

    def chat(ref: ModelRef, use: str, primary: bool) -> None:
        found.append(
            ModelUse(
                kind="chat",
                provider=ref.provider,
                model=ref.model,
                use=use,
                layer=layer,
                primary=primary,
            )
        )

    def chain(primary: ModelRef, settings: ModelChainSettings, use: str, backup: str) -> None:
        chat(primary, use, primary=True)
        for fallback in settings.fallbacks:
            chat(fallback, backup, primary=False)

    if config.llm.enabled:
        provider: ProviderName = config.llm.provider
        primary = ModelRef(provider=provider, model=config.llm.provider_settings(provider).model)
        chain(primary, config.llm.chain, words["chat"], words["chat_fallback"])
        for role, spec in sorted(config.llm.roles.items()):
            chain(
                spec,
                spec,
                words["role"].format(role=role),
                words["role_fallback"].format(role=role),
            )
    if config.evals.enabled:
        judge = config.judge
        chain(judge, judge, words["judge"], words["judge_fallback"])
    found.extend(_embedding(config, layer))
    for guard in GUARDS:
        found.extend(_guard(config, layer, guard))
    return found


def _embedding(config: AppConfig, layer: str) -> list[ModelUse]:
    embedding = config.llm.embedding
    if embedding.backend != HOSTED:
        return []
    return [
        ModelUse(
            kind="embedding",
            provider=embedding.provider,
            model=embedding.model,
            use=config.llm.check.uses["embedding"],
            layer=layer,
            primary=True,
        )
    ]


def _guard(config: AppConfig, layer: str, name: str) -> list[ModelUse]:
    guards = config.security.guards
    settings = getattr(guards, name)
    if not (guards.enabled and settings.enabled):
        return []
    words = config.llm.check.uses
    # The guard model is required where the layer's posture screens with it.
    feature = "prompt_injection" if name == GUARDS[0] else "content_moderation"
    required = config.features.is_on(feature)
    primary = ModelUse(
        kind="guard",
        provider=settings.provider,
        model=settings.model,
        use=words[name],
        layer=layer,
        primary=required,
    )
    # A guard's fallbacks are chat models prompted as the classifier: probed as chat.
    fallbacks = [
        ModelUse(
            kind="chat",
            provider=config.llm.provider,
            model=item.model,
            use=words[f"{name}_fallback"],
            layer=layer,
            primary=False,
        )
        for item in settings.fallbacks
    ]
    return [primary, *fallbacks]


def merge(uses: list[ModelUse]) -> list[InventoryEntry]:
    """One entry per distinct (kind, model), uses in the order found."""
    entries: dict[tuple[str, str], InventoryEntry] = {}
    for use in uses:
        key = (use.kind, use.model)
        entry = entries.setdefault(
            key, InventoryEntry(kind=use.kind, provider=use.provider, model=use.model)
        )
        entry.uses.append(use)
    return list(entries.values())
