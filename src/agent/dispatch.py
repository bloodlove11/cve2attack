"""Product Chat dispatch: Live ATT&CK for CVE→ATT&CK, else JSON-only LLM chat."""

from __future__ import annotations

from typing import Any

from src.agent.graph import ChatFn, run_agent
from src.agent.history import history_as_prior_text, normalize_history
from src.attack.chat_live import run_live_attack_answer
from src.attack.detect import extract_cve_to_attack_intent
from src.schemas.answer import AgentAnswer


def run_chat_query(
    query: str,
    *,
    history: list[dict[str, Any]] | None = None,
    chat_fn: ChatFn | None = None,
    model: str | None = None,
    trace: bool = True,
    trace_info: dict[str, Any] | None = None,
    use_live_attack: bool = True,
    live_attack_rag: bool = True,
) -> AgentAnswer:
    """Route Chat / API / CLI queries.

    When ``use_live_attack`` is True (default) and the query looks like
    CVE→ATT&CK mapping, call the same Live pipeline as golden eval
    ``mode=agent`` (``evals.golden.live_attack.predict_cve_to_attack``).
    Neighbor ICL on this path uses the CTID corpus (not CIRCL train) unless
    a CIRCL protocol is passed through. Otherwise fall through to the
    JSON-only chat agent (no tools).

    ``history`` is prior user/assistant turns for multi-turn Chat. Follow-ups
    that omit the CVE id but keep ATT&CK language recover the id from history.
    ``model`` is passed explicitly (does not mutate ``EXPLABS_MODEL``).
    """
    prior = normalize_history(history)
    prior_text = history_as_prior_text(prior) if prior else None

    if use_live_attack:
        cve_id = extract_cve_to_attack_intent(query, prior_text=prior_text)
        if cve_id:
            return run_live_attack_answer(
                cve_id,
                query=query,
                use_rag=live_attack_rag,
                model=model,
                trace=trace,
                trace_info=trace_info,
            )
    return run_agent(
        query,
        history=prior,
        chat_fn=chat_fn,
        model=model,
        trace=trace,
        trace_info=trace_info,
    )
