"""Shared shape for episode applications (LLD §22).

Two shapes live here while the series converts:

`runner` — the current one. An episode runs the real customer support
agent, built by the same `build_agent()` the capstone uses, under a
`config.yaml` of its own that decides which subsystems are live.

`narrated` — the one being retired, where an episode hand-wired the few
subsystems it wanted to discuss. Still exported so unconverted episodes
keep running; delete it when the last one has moved.
"""

from episodes._shared.narrated import (
    EpisodeOutput,
    OfflineLLM,
    episode_llm,
    main,
    render,
)
from episodes._shared.runner import (
    CONFIG_FILENAME,
    EpisodeRun,
    Flow,
    Setup,
    load_episode,
    run_episode,
    run_scripted_episode,
)
from episodes._shared.settings import EpisodeSettings

__all__ = [
    "CONFIG_FILENAME",
    "EpisodeOutput",
    "EpisodeRun",
    "EpisodeSettings",
    "Flow",
    "Setup",
    "OfflineLLM",
    "episode_llm",
    "main",
    "load_episode",
    "render",
    "run_episode",
    "run_scripted_episode",
]
