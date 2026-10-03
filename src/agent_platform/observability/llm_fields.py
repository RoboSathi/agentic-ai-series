"""Model request and response cards, as fields a person can read — shared by every agent.

What the model was handed (its system prompt, the question, the tools offered, how big the window
was) and what came back (tokens, latency, the tool calls or the text). Pure formatting: a request
or result in, a dict of display fields out. The agent decides when to show them.
"""

from typing import Any

from agent_platform.config import get_settings


def llm_request_fields(request: Any, *, model: str = "") -> dict[str, Any]:
    """Render the important inputs supplied to the LLM.

    Keep the operator-facing event concise: the system prompt and latest user
    query are the actual prompt, while the offered tool names explain the
    action space. Provider-specific request envelopes are deliberately not
    repeated here.
    """
    system_prompt = next(
        (message.content for message in request.messages if message.role == "system"),
        "",
    )
    user_messages = [message.content for message in request.messages if message.role == "user"]
    tools = request.tools or []
    tool_names: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            function = tool.get("function", tool)
            if isinstance(function, dict) and function.get("name"):
                tool_names.append(str(function["name"]))
                continue
        name = getattr(tool, "name", None)
        if name:
            tool_names.append(str(name))

    return {
        "model": model or "unknown",
        "context_tokens": _tokens(request.messages),
        "system_prompt": system_prompt,
        "user_query": user_messages[-1] if user_messages else "",
        "tools_offered": tool_names,
    }


def _tokens(messages: list[Any]) -> int:
    """How big what is sent is, in tokens (the context pipeline's own estimate)."""
    characters = sum(len(str(getattr(message, "content", "") or "")) for message in messages)
    return max(1, characters // get_settings().context.chars_per_token)


def llm_response_fields(result: Any) -> dict[str, Any]:
    """Render only the operator-useful parts of an LLM response.

    Not the model or the tools offered: the request card, directly above,
    already says both.
    """
    fields: dict[str, Any] = {
        "input_tokens": result.usage.prompt_tokens,
        "output_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "latency_ms": round(float(result.latency_ms), 2),
    }
    if result.finish_reason:
        fields["finish_reason"] = result.finish_reason
    if result.tool_calls:
        fields["tool_calls"] = [
            {"tool": call.name, "arguments": call.arguments} for call in result.tool_calls
        ]
    elif result.content:
        fields["response"] = result.content
    return fields
