"""LLM client stubs for Phase 0."""

from agentic_commons.llm.types import LLMResponse
from agentic_commons.logging import log_call


class LLMClient:
    """Base LLM client interface."""

    @log_call("llm")
    def generate(self, prompt: str) -> LLMResponse:
        """Phase 0 stub — no real LLM calls yet."""
        return LLMResponse(content="[stub response]")


class NimClient(LLMClient):
    """NVIDIA NIM-based LLM client."""

    @log_call("llm")
    def generate(self, prompt: str) -> LLMResponse:
        """Phase 0 stub — no real NIM calls yet."""
        return LLMResponse(content="[stub NIM response]", model="nim")


class LocalClient(LLMClient):
    """Local mistral-nemo LLM client."""

    @log_call("llm")
    def generate(self, prompt: str) -> LLMResponse:
        """Phase 0 stub — no real local calls yet."""
        return LLMResponse(content="[stub local response]", model="local")
