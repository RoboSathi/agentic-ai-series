"""Unit tests for the customer support agent's system prompt builder."""

from customer_support_agent.prompts.system_prompt import build_system_prompt


def test_build_system_prompt_embeds_policy_content() -> None:
    prompt = build_system_prompt()

    assert "under $50" in prompt
    assert "queue_for_approval" in prompt
    assert "process_refund" in prompt


def test_build_system_prompt_is_stable_across_calls() -> None:
    assert build_system_prompt() == build_system_prompt()
