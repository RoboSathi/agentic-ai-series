"""`python -m episodes.e22` entry point."""

from pathlib import Path

from episodes._shared import run_scripted_episode
from episodes.e22.app import flow

# Guarded, unlike the narrated episodes: importing this module must not
# run the episode. The smoke test imports every entry point to prove it
# exists, and an unguarded body would mean that import builds an agent
# and calls a live model.
if __name__ == "__main__":
    run_scripted_episode(Path(__file__).parent, flow)
