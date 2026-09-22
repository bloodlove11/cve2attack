"""Chat product path → shared Live ATT&CK pipeline (parity with golden agent)."""

from __future__ import annotations

from typing import Any

from src.agent.tracing import TraceLogger
from src.schemas.answer import AgentAnswer, Citation


def _format_answer(cve_id: str, pred: dict[str, Any]) -> str:
    exploitation = pred.get("exploitation_techniques") or []
    primary = pred.get("primary_impact") or []
    attack = pred.get("attack_techniques") or []
    src = pred.get("enrichment_source") or "unknown"
    def id_list(ids: list[str]) -> str:
        return ", ".join(f"`{t}`" for t in ids) if ids else "_(none)_"

    lines = [
        f"**Live ATT&CK mapping for `{cve_id}`** "
        "(same `predict_cve_to_attack` helper as golden eval `mode=agent`. "
        "Chat neighbor ICL uses the CTID corpus, not CIRCL train, so numbers "
        "will not match `--protocol circl_test`).",
        "",
        f"- **exploitation_techniques**: {id_list(exploitation)}",
        f"- **primary_impact**: {id_list(primary)}",
        f"- **attack_techniques** (union): {id_list(attack)}",
        "",
        f"Enrichment source: `{src}`. "
        "No CTID gold lookup for the target CVE.",
    ]
    if pred.get("error"):
        lines.extend(["", f"Note: pipeline reported an error: `{pred['error']}`."])
    return "\n".join(lines)


def run_live_attack_answer(
    cve_id: str,
    *,
    query: str | None = None,
    description: str | None = None,
    use_rag: bool = True,
    model: str | None = None,
    trace: bool = True,
    trace_info: dict[str, Any] | None = None,
    llm_chat: Any | None = None,
) -> AgentAnswer:
    """Run shared ``predict_cve_to_attack`` and wrap as ``AgentAnswer``.

    Used by Chat / FastAPI / CLI when a query is detected as CVE→ATT&CK so
    product demos match golden ``mode=agent`` numbers.
    """
    from evals.golden.live_attack import predict_cve_to_attack

    cid = (cve_id or "").strip().upper()
    tracer = TraceLogger(query or f"Live ATT&CK {cid}", enabled=trace)
    tracer.set_meta(
        pipeline="live_attack",
        cve_id=cid,
        rag=use_rag,
        model=model,
        parity_with="golden_mode_agent",
    )
    tracer.add_step("user", content=query or cid)
    tracer.add_step(
        "tool_call",
        name="live_attack_pipeline",
        arguments={"cve_id": cid, "use_rag": use_rag, "model": model},
    )

    pred = predict_cve_to_attack(
        cid,
        description=description,
        use_rag=use_rag,
        llm_chat=llm_chat,
        model=model,
    )
    tracer.add_step(
        "tool_result",
        name="live_attack_pipeline",
        result={k: v for k, v in pred.items() if k != "_raw"},
    )

    attack = pred.get("attack_techniques") or []
    err = pred.get("error")
    status = "error" if err and not attack else ("ok" if attack else "partial")
    confidence = 0.75 if attack and not err else (0.35 if attack else 0.15)

    answer = AgentAnswer(
        answer=_format_answer(cid, pred),
        confidence=confidence,
        tools_used=["live_attack_pipeline"],
        citations=[
            Citation(
                source="live_attack_pipeline",
                snippet=(
                    "Shared with evals.golden.live_attack.predict_cve_to_attack "
                    "/ golden mode=agent"
                ),
            )
        ],
        status=status,  # type: ignore[arg-type]
    )
    path = tracer.finalize(answer.model_dump())
    tracer.fill(
        trace_info,
        path,
        pipeline="live_attack",
        prediction={k: v for k, v in pred.items() if k != "_raw"},
    )
    return answer
