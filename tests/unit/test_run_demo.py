"""Unit tests for the demo entry point."""

from unittest.mock import patch

from customer_support_agent.dev.run_demo import main


def test_main_returns_run_agent_result() -> None:
    with patch(
        "customer_support_agent.dev.run_demo.run_agent", return_value="demo response"
    ) as mock_run_agent:
        result = main()

    assert result == "demo response"
    mock_run_agent.assert_called_once()
