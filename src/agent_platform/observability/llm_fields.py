"""Model request and response cards, as fields a person can read — shared by every agent.

What the model was handed (its system prompt, the question, the tools offered, how big the window
was) and what came back (tokens, latency, the tool calls or the text). Pure formatting: a request
or result in, a dict of display fields out. The agent decides when to show them.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.llm.prompt import tool_names
from agent_platform.llm.tokenizer import count_tokens


def llm_request_fields(request: Any, *, model: str = "") -> dict[str, Any]:
    """Render the important inputs supplied to the LLM.

    The system prompt and latest user query lead; `messages` is every other
    message the model receives, in order, including its own earlier tool
    calls and their results, so a later round's card shows the whole
    context the model reasons over. The offered tool names explain the
    action space. Provider-specific request envelopes are deliberately not
    repeated here.
    """
    system_prompt = next(
        (message.content for message in request.messages if message.role == "system"),
        "",
    )
    user_messages = [message.content for message in request.messages if message.role == "user"]
    manifest = getattr(request, "manifest", None)
    return {
        "model": model or "unknown",
        "context_tokens": manifest.total_tokens if manifest else _tokens(request.messages),
        "system_prompt": system_prompt,
        "user_query": user_messages[-1] if user_messages else "",
        "messages": _messages(request.messages),
        "tools_offered": tool_names(request.tools or []),
    }


def _messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Every non-system message as sent; a tool result is named by the tool that produced it.

    A result carries only its call's opaque id, so the id is resolved to the tool name the
    earlier assistant message requested it under.
    """
    tool_names: dict[str, str] = {}
    shown: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {"tool": call.name, "arguments": call.arguments} for call in message.tool_calls
            ]
            tool_names.update({call.id: call.name for call in message.tool_calls if call.id})
        if message.tool_call_id:
            entry["tool"] = tool_names.get(message.tool_call_id, message.tool_call_id)
        shown.append(entry)
    return shown


def _tokens(messages: list[Any]) -> int:
    """How big what is sent is, in tokens, by the shared tokenizer."""
    return sum(count_tokens(str(getattr(message, "content", "") or "")) for message in messages)


def estimate_fields(estimated: int, counted: int) -> dict[str, Any]:
    """The request's own size beside the provider's count, and how far apart they are.

    Nothing when either is unknown: a drift against zero says nothing.
    """
    if not estimated or not counted:
        return {}
    return {
        "estimated_input_tokens": estimated,
        "estimate_drift": f"{(estimated - counted) / counted:+.1%}",
    }


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


def cache_fields(usage: Any) -> dict[str, Any]:
    """Whether the provider served the prompt's prefix from its cache.

    A response card puts these last, after everything else it shows about the call.

    A provider that never said how many tokens it cached shows that, not a 0 it did not send.
    """
    reported = getattr(usage, "cache_reported", False)
    cached = getattr(usage, "cached_prompt_tokens", 0)
    shown = cached if reported else get_settings().observability.cache_not_reported
    return {"cached_input_tokens": shown, "cache_hit": bool(cached)}
