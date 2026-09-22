"""Normalize ExperientialLabs / OpenAI-compatible ``usage`` (tokens + cost)."""

from __future__ import annotations

from typing import Any


def empty_usage() -> dict[str, Any]:
    """Zeroed usage accumulator (``cost`` may stay ``None`` until a known value)."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": None,
        "n_calls": 0,
        "n_calls_with_cost": 0,
    }


def _attr_or_key(obj: Any, *names: str) -> Any:
    for name in names:
        if obj is None:
            break
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return None


def extract_usage(response: Any) -> dict[str, Any] | None:
    """Pull tokens + optional USD ``cost`` from a chat-completion response.

    ExperientialLabs attaches ``usage.cost`` (float USD; free-tier often ``0.0``).
    Returns ``None`` when the response has no usage block.
    """
    usage = _attr_or_key(response, "usage")
    if usage is None:
        return None
    prompt = int(_attr_or_key(usage, "prompt_tokens", "input_tokens") or 0)
    completion = int(_attr_or_key(usage, "completion_tokens", "output_tokens") or 0)
    total = _attr_or_key(usage, "total_tokens")
    total_i = int(total) if total is not None else prompt + completion
    cost_raw = _attr_or_key(usage, "cost")
    cost: float | None
    try:
        cost = float(cost_raw) if cost_raw is not None else None
    except (TypeError, ValueError):
        cost = None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total_i,
        "cost": cost,
        "n_calls": 1,
        "n_calls_with_cost": 1 if cost is not None else 0,
    }


def add_usage(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Sum two usage dicts. ``None`` + x = x; both ``None`` → ``None``."""
    if left is None:
        return dict(right) if right is not None else None
    if right is None:
        return dict(left)
    cost_sum = 0.0
    n_with_cost = 0
    for blob in (left, right):
        if blob.get("cost") is None:
            continue
        cost_sum += float(blob["cost"])
        n_with_cost += int(blob.get("n_calls_with_cost") or 1)
    return {
        "prompt_tokens": int(left.get("prompt_tokens") or 0)
        + int(right.get("prompt_tokens") or 0),
        "completion_tokens": int(left.get("completion_tokens") or 0)
        + int(right.get("completion_tokens") or 0),
        "total_tokens": int(left.get("total_tokens") or 0)
        + int(right.get("total_tokens") or 0),
        "cost": cost_sum if n_with_cost else None,
        "n_calls": int(left.get("n_calls") or 0) + int(right.get("n_calls") or 0),
        "n_calls_with_cost": n_with_cost,
    }


def usage_from_chat_payload(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read ``_usage`` attached by ``default_llm_chat`` (safe for mocks)."""
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("_usage")
    return dict(raw) if isinstance(raw, dict) else None
