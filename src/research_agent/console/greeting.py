"""The greeting: one line on how to use the agent.

Ask a question and it is answered; `/research` for an audited deep run; `/help` for the rest.
Limits, budgets, providers and models are in the configuration and the logs, not repeated
here. Notes (a missing-key warning) follow on their own lines, only when there are any.
"""

from research_agent.config.settings import ResearchAgentSettings


def build_greeting(agent: ResearchAgentSettings, notes: list[str]) -> str:
    """The greeting line, then any notes as given."""
    return "\n".join([agent.greeting.line, *notes])
