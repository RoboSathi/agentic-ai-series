"""What an episode says about itself (LLD §22).

An episode's identity — its title, the one claim it exists to make, and
the conversation that makes it — is configuration, not code. It lives in
the same `config.yaml` that decides which subsystems are live, because
the two are the same decision: E30's turns are the turns they are
*because* E30 is the episode where prompt injection is on.

Keeping it here rather than in `AppConfig` is deliberate. The platform's
root config model describes a harness, and an episode is not a harness
feature — adding an `episode` section to it would mean every agent built
on this library carried a field about a YouTube series.
"""

from pydantic_settings import SettingsConfigDict

from agentic_commons.config import YamlBackedSettings


class EpisodeSettings(YamlBackedSettings):
    """One episode's identity and the conversation it runs."""

    _yaml_section = "episode"

    title: str
    # The single sentence the episode has to earn. It is printed before
    # anything runs, so a viewer knows what to watch the trace for.
    principle: str
    # The customer's side of a scripted conversation. Belongs to the
    # shape being retired: an episode on the TUI is typed live, so it
    # declares no turns. Empty here rather than required, so a converted
    # episode does not have to carry a script nothing reads.
    turns: list[str] = []

    # Extras are allowed *here* and nowhere else in the config tree. An
    # episode keeps whatever its own demonstration needs under
    # `episode.` — E01's cold-open order, say — and reads it back with a
    # model of its own pointed at `episode.<something>`. The platform
    # sections keep forbidding extras, where a misspelled key should
    # fail loudly rather than be silently ignored.
    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")
