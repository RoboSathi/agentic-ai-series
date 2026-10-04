"""The researcher's prompt when the quarantine is withheld (an ablation).

With `quarantined_extractor` off, the worker's own model reads the page
and decides what to do next in one step — the ordinary, unquarantined
ReAct shape. Its answer can therefore carry *actions* (follow-up queries
and URLs) that a page may have planted. This is the configuration the
quarantine exists to replace; it is built only when that feature is off.
"""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from research_agent.prompts.base import RolePrompt, dated
from research_agent.prompts.extractor import ExtractedClaim

ROLE = "researcher"

INSTRUCTIONS = """
You are researching one question. Read the page, extract claims that
answer it (claim + exact supporting span), and propose any follow-up web
searches or pages to fetch that would help answer the question.
"""


class ResearcherAnswer(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    follow_up_urls: list[str] = Field(default_factory=list)


def build(
    question: str,
    url: str,
    page_text: str,
    max_claims: int,
    brief: dict[str, Any],
    today: date,
) -> RolePrompt:
    """The ordinary single-context agent: it holds the whole Brief *and* reads the page."""
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload=dated(
            {
                "task": "research_read",
                "question": question,
                "url": url,
                "page": page_text,
                "max_claims": max_claims,
                "brief": brief,
            },
            today,
        ),
    )
