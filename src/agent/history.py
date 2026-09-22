"""Conversation history helpers for multi-turn Chat / API / CLI."""

from __future__ import annotations

from typing import Any

HistoryTurn = dict[str, str]


def normalize_history(
    history: list[dict[str, Any]] | None,
    *,
    max_turns: int = 12,
) -> list[HistoryTurn]:
    """Keep recent user/assistant turns with non-empty string content."""
    if not history:
        return []
    out: list[HistoryTurn] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = turn.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        out.append({"role": role, "content": text})
    if max_turns > 0 and len(out) > max_turns:
        out = out[-max_turns:]
    return out


def history_as_prior_text(history: list[HistoryTurn]) -> str:
    """Flatten history for CVE / intent recovery from earlier turns."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in history)


def history_to_messages(history: list[HistoryTurn]) -> list[dict[str, Any]]:
    """Map normalized history into OpenAI-style chat messages."""
    return [{"role": t["role"], "content": t["content"]} for t in history]
