"""`python -m episodes.e01` entry point."""

from pathlib import Path

from episodes._shared import run_episode
from episodes.e01.app import setup

if __name__ == "__main__":
    run_episode(Path(__file__).parent, setup=setup)
