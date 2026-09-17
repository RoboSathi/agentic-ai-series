"""Command-line entry point for verifying LLM connectivity: `make llm-check`."""

import sys

from agentic_commons.config import get_settings
from agentic_commons.llm.client import create_llm_client
from agentic_commons.logging import get_logger


def main() -> int:
    """Run a live health check against the configured LLM endpoint."""
    logger = get_logger("llm")
    settings = get_settings().llm

    logger.info(
        f"[bright_blue]Checking {settings.provider} endpoint[/bright_blue] "
        f"{settings.base_url} (model={settings.model})"
    )

    client = create_llm_client(settings)
    result = client.health_check()

    if result.healthy:
        logger.info(
            f"[green]✓ LLM endpoint reachable[/green] "
            f"({result.latency_ms:.0f}ms)"
        )
        return 0

    logger.error(
        f"[red]✗ LLM endpoint unreachable[/red]: {result.error}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
