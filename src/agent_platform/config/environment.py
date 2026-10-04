"""The process environment, completed once from the repository's `.env`.

The environment is the one place secrets and per-machine overrides live:
bare provider keys (`nvidia_api_key`), the keys `key_env` names
(`LANGGRAPH_AES_KEY`, `PII_VAULT_KEY`), and every `SECTION__FIELD`. Each
consumer reads `os.environ`, so `.env` is only useful once it is *in* the
environment. Parsing it as part of `AppConfig` instead made every one of
those lines an "extra input" and stopped start-up, while still leaving
`os.environ` without them.

Resolved against the repository root, never the working directory, and
never overriding a variable the shell already exported.
"""

from functools import cache
from pathlib import Path

from dotenv import load_dotenv

from agent_platform.config.paths import resolve_path

ENVIRONMENT_FILE = ".env"


@cache
def load_environment() -> None:
    """Fill the process environment from the repository's `.env`, once per process."""
    apply_environment_file(resolve_path(ENVIRONMENT_FILE))


def apply_environment_file(path: Path) -> None:
    """Add every variable a file sets that the environment does not already have."""
    if path.is_file():
        load_dotenv(path, override=False)


__all__ = ["ENVIRONMENT_FILE", "apply_environment_file", "load_environment"]
