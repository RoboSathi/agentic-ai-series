"""Command-line entry point for verifying LLM connectivity: `make llm-check`."""

import asyncio
import sys

from agentic_commons.config import get_settings
from agentic_commons.llm.client import create_llm_client
from agentic_commons.logging import get_logger


async def check() -> int:
    """Run a live health check against the configured LLM endpoint."""
    logger = get_logger("llm")
    settings = get_settings().llm
    provider = settings.provider_settings()

    logger.info(
        f"[bright_blue]Checking {settings.provider} endpoint[/bright_blue] "
        f"{provider.base_url} (model={provider.model})"
    )
    result = await create_llm_client(settings).health_check()

    if result.healthy:
        logger.info(f"[green]✓ LLM endpoint reachable[/green] ({result.latency_ms:.0f}ms)")
        return 0
    logger.error(f"[red]✗ LLM endpoint unreachable[/red]: {result.error}")
    return 1


def main() -> int:
    """Synchronous wrapper so this stays a plain console entry point."""
    return asyncio.run(check())


if __name__ == "__main__":
    sys.exit(main())
