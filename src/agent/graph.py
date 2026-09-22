"""LLM chat agent with validated JSON Schema output.

``run_agent`` is the imperative path used by CLI / API / UI when a query is
not CVE→ATT&CK Live. Answers are a single LLM completion (no tools).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.agent.history import history_to_messages, normalize_history
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tracing import TraceLogger
from src.llm.client import chat_completion, get_model
from src.schemas.answer import AgentAnswer

ChatFn = Callable[..., Any]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _validate_answer(raw: dict[str, Any]) -> AgentAnswer:
    if not raw.get("tools_used"):
        raw = {**raw, "tools_used": []}
    return AgentAnswer.model_validate(raw)


def run_agent(
    query: str,
    *,
    history: list[dict[str, Any]] | None = None,
    chat_fn: ChatFn | None = None,
    model: str | None = None,
    trace: bool = True,
    trace_info: dict[str, Any] | None = None,
) -> AgentAnswer:
    """Run one chat completion and return a structured ``AgentAnswer``.

    Args:
        query: User question (current turn).
        history: Prior user/assistant turns (multi-turn Chat).
        chat_fn: Injectable chat completion (tests pass a mock).
        model: Override ``EXPLABS_MODEL``.
        trace: Persist a JSON trace under ``traces/``.
        trace_info: Optional mutable dict filled with ``path``, ``trace_id``,
            ``steps``, ``meta`` for UI / observability.

    Returns:
        Validated ``AgentAnswer``. On parse failure returns ``partial``/``error``.
    """
    chat = chat_fn or chat_completion
    prior = normalize_history(history)
    tracer = TraceLogger(query, enabled=trace)
    tracer.set_meta(
        model=model or get_model(),
        history_turns=len(prior),
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history_to_messages(prior),
        {"role": "user", "content": query},
    ]
    tracer.add_step("user", content=query)
    if prior:
        tracer.add_step("history", turns=len(prior))

    resp = chat(messages, model=model)
    content = resp.choices[0].message.content or ""
    tracer.add_step("assistant_final", content=content)
    try:
        answer = _validate_answer(_extract_json(content))
    except Exception as exc:  # noqa: BLE001
        answer = AgentAnswer(
            answer=content or f"Failed to parse structured answer: {exc}",
            confidence=0.2,
            tools_used=[],
            status="partial" if content else "error",
        )
    path = tracer.finalize(answer.model_dump())
    tracer.fill(trace_info, path)
    return answer
