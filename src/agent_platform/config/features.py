"""Feature toggles, declared by the agent and resolved once at construction.

A *feature* is a control you can remove and still get an agent that runs —
just a worse or more dangerous one. That is the whole definition, and it is
what separates the toggleable features from the plumbing that is always on:
without the control loop, resilience or logging you do not get a reckless
agent, you get a broken program. Only the first kind is worth a switch,
because only the first kind has something instructive to show when it is
missing.

`agent_platform` owns the mechanism and nothing else. It never learns which
features exist — that tree is declared in the agent's own config.yaml, so two
agents in one process can hold different postures. This is why the type is a
value passed to a composition root rather than a global anyone can consult: a
component that reads a process-wide toggle has decided for every agent in the
process, which is precisely the coupling this replaces.
"""

from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The key that switches a node with children on or off. A node without one
# is on: `rerank: true` and `rerank: {enabled: true}` mean the same thing.
ENABLED_KEY = "enabled"
SEPARATOR = "."


class FeatureSet(BaseModel):
    """A feature tree with parent gating.

    A node is either a bool leaf (`pii: true`) or a mapping carrying an
    `enabled` key and children (`memory: {enabled: true, working: true}`).
    A child is on only if every ancestor is, so switching `memory` off
    switches off all four CoALA stores without restating any of them.

    Never mutated: every operation returns a new set, which is what makes
    it safe to hand the same posture to several collaborators.
    """

    model_config = ConfigDict(frozen=True)

    tree: dict[str, Any] = Field(default_factory=dict)

    def is_on(self, path: str) -> bool:
        """Whether this feature and every one of its ancestors is enabled.

        An undeclared path is off. A feature nobody declared is not a
        feature this agent was given, and guessing "on" for an unknown
        name is how a control silently reappears after a typo.
        """
        node: Any = self.tree
        for part in path.split(SEPARATOR):
            if not isinstance(node, Mapping) or part not in node:
                return False
            node = node[part]
            if not _is_enabled(node):
                return False
        return True

    def child(self, name: str) -> "FeatureSet":
        """The subtree beneath one feature, as a set in its own right."""
        node = self.tree.get(name)
        return FeatureSet(tree=_copy(node)) if isinstance(node, Mapping) else FeatureSet()

    def paths(self) -> list[str]:
        """Every declared path, parents before their own children."""
        return list(_walk(self.tree, ""))

    def with_override(self, path: str, value: bool) -> "FeatureSet":
        """A copy with one path set, raising if it was never declared.

        Declared-only, because the caller is usually a human typing a
        feature name: silently creating `hallucinaton_guard` would report
        success and change nothing.
        """
        if path not in self.paths():
            raise KeyError(path)
        return FeatureSet(tree=_with_value(self.tree, path.split(SEPARATOR), value))

    def with_all(self, value: bool) -> "FeatureSet":
        """A copy with every feature and sub-feature set one way."""
        return FeatureSet(tree=_with_every_value(self.tree, value))


def _is_enabled(node: Any) -> bool:
    """Whether one node is switched on, whichever shape it has."""
    if isinstance(node, Mapping):
        return bool(node.get(ENABLED_KEY, True))
    return bool(node)


def _walk(node: Mapping[str, Any], prefix: str) -> Iterator[str]:
    """Every declared path under a node, parents first."""
    for name, value in node.items():
        if name == ENABLED_KEY:
            continue
        path = f"{prefix}{SEPARATOR}{name}" if prefix else name
        yield path
        if isinstance(value, Mapping):
            yield from _walk(value, path)


def _with_value(node: Mapping[str, Any], parts: list[str], value: bool) -> dict[str, Any]:
    """A copy of the tree with one path set, sharing no mutable state."""
    head, rest = parts[0], parts[1:]
    copied = {key: _copy(item) for key, item in node.items()}
    target = copied.get(head)
    if rest:
        copied[head] = _with_value(target if isinstance(target, Mapping) else {}, rest, value)
    elif isinstance(target, Mapping):
        copied[head] = {**target, ENABLED_KEY: value}
    else:
        copied[head] = value
    return copied


def _with_every_value(node: Mapping[str, Any], value: bool) -> dict[str, Any]:
    """A copy with every leaf — `enabled` keys included — set one way."""
    return {
        key: _with_every_value(item, value) if isinstance(item, Mapping) else value
        for key, item in node.items()
    }


def _copy(value: Any) -> Any:
    """A structural copy, so an override cannot reach back into the original."""
    return (
        {key: _copy(item) for key, item in value.items()} if isinstance(value, Mapping) else value
    )


# The posture in force right now. `None` means nothing has overridden the
# declared configuration yet, which is not the same as "no features" —
# conflating the two is what made `/harness off` appear to work and then
# hand back every feature on the next rebuild.
_active: FeatureSet | None = None


def active_features(declared: FeatureSet | None = None) -> FeatureSet:
    """The posture in force: what a console last set, else what was declared.

    The composition root passes `declared` so a first build starts from
    configuration. The console passes nothing, because by the time an
    operator can type a command an agent has been built and published.

    `declared` is a parameter rather than a config read because this
    module is imported *by* the settings module; asking it for the answer
    here would be a cycle.
    """
    if _active is not None:
        return _active
    return declared if declared is not None else FeatureSet()


def set_active_features(features: FeatureSet | None) -> None:
    """Publish the posture in force, or `None` to fall back to config."""
    global _active
    _active = features
