from pathlib import Path


def test_finalize_routes_model_completion_through_resilience_wrapper() -> None:
    source = Path("src/customer_support_agent/core/graph.py").read_text()
    finalize_start = source.index("    async def finalize(")
    finalize_block = source[finalize_start:source.index("    def _system_prompt(", finalize_start)]
    assert "result = await self._call_model(" in finalize_block
    assert "await self._llm.complete(" not in finalize_block
