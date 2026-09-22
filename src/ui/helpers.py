"""Shared UI helpers for sec-tool-agent Streamlit pages.

Pure helpers avoid importing Streamlit so unit tests stay cheap.
Rendering helpers import Streamlit lazily inside the function body.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def api_key_configured() -> bool:
    """True when ``EXPLABS_API_KEY`` is set (live Chat / Live-agent eval)."""
    return bool(os.environ.get("EXPLABS_API_KEY"))


# Starter Live model ids for Streamlit pickers (not a full catalog).
STARTER_MODELS: tuple[str, ...] = (
    "deepseek-v4-flash",
    "gpt-6-astra",
    "gpt-5.6-luna",
    "claude-fable-5.1",
    "qwen3.8-27b",
    "gpt-5.6-sol",
    "claude-sonnet-5",
)
# Public: the Streamlit pages compare the picker value against this.
CUSTOM_MODEL_LABEL = "Custom / EXPLABS_MODEL"


def model_picker_options() -> list[str]:
    """Preset labels plus a custom free-text option."""
    return [*STARTER_MODELS, CUSTOM_MODEL_LABEL]


def resolve_ui_model(choice: str, custom_text: str = "") -> str:
    """Map picker choice (+ optional custom text) to a concrete model id."""
    from src.llm.client import DEFAULT_MODEL, get_model

    label = (choice or "").strip()
    if label == CUSTOM_MODEL_LABEL or label.lower() == "custom":
        custom = (custom_text or "").strip()
        if custom:
            return custom
        return get_model() or DEFAULT_MODEL
    if label in STARTER_MODELS:
        return label
    # Free-text choice pasted as the select value
    if label:
        return label
    return get_model() or DEFAULT_MODEL


def apply_runtime_model(model: str) -> str:
    """Resolve a model id and set ``EXPLABS_MODEL`` for callers that read env.

    ``run_query`` / ``run_chat_query`` / ``run_eval`` all take ``model=``
    directly now, so prefer that over calling this first. Kept for callers
    that still need ``get_model()`` itself to reflect a picker choice.
    """
    from src.llm.client import DEFAULT_MODEL

    resolved = (model or "").strip() or DEFAULT_MODEL
    os.environ["EXPLABS_MODEL"] = resolved
    return resolved


def session_history_from_messages(
    messages: list[dict[str, Any]] | None,
    *,
    max_turns: int = 12,
) -> list[dict[str, str]]:
    """Build API-style history from Streamlit session messages (exclude current)."""
    from src.agent.history import normalize_history

    raw: list[dict[str, Any]] = []
    for msg in messages or []:
        role = str(msg.get("role") or "").strip().lower()
        if role == "user":
            content = msg.get("content")
        elif role == "assistant":
            answer = msg.get("answer")
            if answer is not None and hasattr(answer, "answer"):
                content = getattr(answer, "answer", None)
            elif isinstance(answer, dict):
                content = answer.get("answer")
            else:
                content = msg.get("content")
        else:
            continue
        if content is None:
            continue
        raw.append({"role": role, "content": str(content)})
    return normalize_history(raw, max_turns=max_turns)


def run_query(
    query: str,
    *,
    trace: bool = True,
    model: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run the same product path as the CLI / FastAPI ``/chat`` endpoint.

    CVE→ATT&CK queries use the shared Live ATT&CK pipeline (same function as
    golden ``mode=agent``; Chat ICL is CTID-neighbor, not CIRCL train).
    Other queries use a JSON-only LLM chat (no tools). Pass ``history`` for
    multi-turn follow-ups.

    Returns ``(AgentAnswer, trace_info)``.
    """
    from src.agent.dispatch import run_chat_query

    trace_info: dict[str, Any] = {}
    answer = run_chat_query(
        query,
        history=history,
        trace=trace,
        trace_info=trace_info,
        model=model,
    )
    return answer, trace_info


def format_trace_summary(trace_info: dict[str, Any]) -> str:
    """One-line sidebar summary of the last run trace."""
    if not trace_info:
        return "No trace yet."
    tid = trace_info.get("trace_id") or "?"
    path = trace_info.get("path") or "(not written)"
    steps = trace_info.get("steps") or []
    tool_calls = [s for s in steps if s.get("kind") == "tool_call"]
    meta = trace_info.get("meta") or {}
    turns = meta.get("history_turns")
    extra = f" · {turns} prior turn(s)" if turns else ""
    return (
        f"`{tid}` · {len(tool_calls)} tool call(s){extra}\n\n"
        f"Path: `{path}`"
    )


def _cell_for_dataframe(value: Any) -> Any:
    """Coerce nested values so ``st.dataframe`` / Arrow can serialize them.

    ATT&CK predictions are lists of technique IDs; Arrow rejects ``object``
    columns that mix strings and lists (``Expected bytes, got a 'list'``).
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def results_dataframe(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten golden-eval payload rows for ``st.dataframe`` (Arrow-safe)."""
    rows: list[dict[str, Any]] = []
    for r in payload.get("results") or []:
        scores = r.get("scores") or {}
        pred = r.get("prediction") or {}
        exp = r.get("expected") or {}
        predicted = pred.get("triage_label")
        if predicted is None:
            predicted = pred.get("attack_techniques")
        if predicted is None or predicted == "":
            predicted = "(skipped)" if r.get("skipped") else ""
        expected = exp.get("triage_label")
        if expected is None:
            expected = exp.get("attack_techniques")
        if expected is None:
            expected = ""
        row_out: dict[str, Any] = {
            "id": r.get("id"),
            "task": r.get("task"),
            "skipped": bool(r.get("skipped")),
            "correct": r.get("correct"),
            "predicted": _cell_for_dataframe(predicted),
            "expected": _cell_for_dataframe(expected),
        }
        # CTID two-head predictions (Arrow-safe strings) when present
        if "primary_impact" in pred and pred.get("primary_impact") is not None:
            row_out["predicted_primary_impact"] = _cell_for_dataframe(
                pred.get("primary_impact")
            )
        if "exploitation_techniques" in pred and pred.get(
            "exploitation_techniques"
        ) is not None:
            row_out["predicted_exploitation"] = _cell_for_dataframe(
                pred.get("exploitation_techniques")
            )
        row_out.update({f"score_{k}": v for k, v in scores.items()})
        row_out["error"] = _cell_for_dataframe(r.get("error", ""))
        rows.append(row_out)
    return rows


def status_label(status: str) -> str:
    """Sentence-case status with a material icon (no emoji)."""
    icons = {
        "ok": ":material/check_circle:",
        "partial": ":material/warning:",
        "error": ":material/error:",
    }
    icon = icons.get(status, ":material/info:")
    return f"{icon} {status}"


def render_answer_card(answer: Any) -> None:
    """Render AgentAnswer in a bordered card with metrics."""
    import streamlit as st

    payload = answer.model_dump() if hasattr(answer, "model_dump") else dict(answer)
    status = payload.get("status", "ok")
    confidence = float(payload.get("confidence", 0.0))
    tools = payload.get("tools_used") or []
    citations = payload.get("citations") or []

    with st.container(border=True):
        st.markdown(payload.get("answer", ""))
        with st.container(horizontal=True):
            st.metric("Status", status_label(status))
            st.metric("Confidence", f"{confidence:.0%}")
            st.metric("Tools", str(len(tools)))
        if tools:
            st.caption("Tools used: " + ", ".join(f"`{t}`" for t in tools))
        if citations:
            with st.expander(f"Citations ({len(citations)})", expanded=False):
                for cite in citations:
                    src = cite.get("source", "?") if isinstance(cite, dict) else cite.source
                    snip = (
                        cite.get("snippet")
                        if isinstance(cite, dict)
                        else getattr(cite, "snippet", None)
                    )
                    st.markdown(f"**{src}**")
                    if snip:
                        st.caption(snip)
        with st.expander("Raw JSON (AgentAnswer schema)", expanded=False):
            st.code(json.dumps(payload, indent=2), language="json")


def render_tool_timeline(steps: list[dict[str, Any]], *, live: bool = False) -> None:
    """Render tool call / result steps as a compact + step status timeline."""
    import streamlit as st

    tool_steps = [s for s in steps if s.get("kind") in {"tool_call", "tool_result"}]
    if not tool_steps:
        return

    label = ":shimmer[Running tools]" if live else "Tool timeline"
    with st.status(
        label,
        type="compact",
        expanded=live,
        state="running" if live else "complete",
    ) as outer:
        for step in tool_steps:
            kind = step.get("kind")
            name = step.get("name", "?")
            if kind == "tool_call":
                with st.status(f"Tool call · `{name}`", type="step", state="complete"):
                    args = step.get("arguments", "")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                        st.json(parsed)
                    except (TypeError, json.JSONDecodeError):
                        st.code(str(args))
            else:
                with st.status(f"Tool result · `{name}`", type="step", state="complete"):
                    st.json(step.get("result"))
        if live:
            outer.update(label="Tool timeline", state="complete", expanded=False)
