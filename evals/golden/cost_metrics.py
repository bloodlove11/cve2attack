"""Aggregate per-case LLM usage into eval-run cost / token metrics."""

from __future__ import annotations

from typing import Any


def _prediction_usage(row: dict[str, Any]) -> dict[str, Any] | None:
    pred = row.get("prediction") or {}
    if not isinstance(pred, dict):
        return None
    usage = pred.get("usage")
    return usage if isinstance(usage, dict) else None


def aggregate_cost_metrics(
    rows: list[dict[str, Any]],
    *,
    model: str | None = None,
    hit_rate: float | None = None,
) -> dict[str, Any]:
    """Sum prediction ``usage`` across result rows for cost benchmarking.

    Free-tier ExperientialLabs models often report ``cost=0.0``; that is still
    a valid reading (not missing). Missing usage blocks are excluded from
    averages but counted under ``n_cases_without_usage``.
    """
    prompt = completion = total = 0
    cost_sum = 0.0
    n_with_cost = 0
    n_with_usage = 0
    llm_calls = 0
    models: set[str] = set()
    if model:
        models.add(model)

    for row in rows:
        if row.get("skipped"):
            continue
        pred = row.get("prediction") or {}
        if isinstance(pred, dict) and pred.get("model"):
            models.add(str(pred["model"]))
        calls = pred.get("llm_calls") if isinstance(pred, dict) else None
        if isinstance(calls, int):
            llm_calls += calls
        usage = _prediction_usage(row)
        if usage is None:
            continue
        n_with_usage += 1
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
        if usage.get("cost") is not None:
            cost_sum += float(usage["cost"])
            n_with_cost += int(usage.get("n_calls_with_cost") or 1)
        u_calls = usage.get("n_calls")
        if isinstance(u_calls, int) and not isinstance(calls, int):
            llm_calls += u_calls

    n_scored = sum(1 for r in rows if not r.get("skipped"))
    n_without = max(0, n_scored - n_with_usage)
    avg_cost = (cost_sum / n_with_usage) if n_with_usage else None
    avg_tokens = (total / n_with_usage) if n_with_usage else None
    cost_per_hit: float | None = None
    if hit_rate is not None and n_scored > 0 and n_with_cost > 0:
        hits = hit_rate * n_scored
        if hits > 0:
            cost_per_hit = cost_sum / hits

    resolved_model = model
    if not resolved_model and len(models) == 1:
        resolved_model = next(iter(models))

    return {
        "model": resolved_model,
        "models_seen": sorted(models),
        "total_cost_usd": cost_sum if n_with_cost else None,
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_tokens": total,
        "total_llm_calls": llm_calls,
        "n_cases_with_usage": n_with_usage,
        "n_cases_without_usage": n_without,
        "n_calls_with_cost": n_with_cost,
        "avg_cost_per_case_usd": avg_cost if n_with_cost else None,
        "avg_tokens_per_case": avg_tokens,
        "cost_per_hit_usd": cost_per_hit,
    }


def cost_row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one results JSON into a cost-benchmark table row."""
    metrics = payload.get("metrics") or {}
    cost = metrics.get("cost") or {}
    att = metrics.get("attack") or {}
    if not cost:
        cost = aggregate_cost_metrics(
            list(payload.get("results") or []),
            model=payload.get("model") or metrics.get("model"),
            hit_rate=att.get("hit_rate", metrics.get("hit_rate")),
        )
    return {
        "model": cost.get("model") or payload.get("model") or "?",
        "protocol": payload.get("protocol"),
        "refine_exploitation": payload.get("refine_exploitation"),
        "n": metrics.get("n_scored") or att.get("n") or metrics.get("n_total"),
        "hit": att.get("hit_rate", metrics.get("hit_rate")),
        "recall_at_k": att.get("recall_at_k", metrics.get("recall_at_k")),
        "precision_at_k": att.get("precision_at_k", metrics.get("precision_at_k")),
        "total_cost_usd": cost.get("total_cost_usd"),
        "avg_cost_per_case_usd": cost.get("avg_cost_per_case_usd"),
        "cost_per_hit_usd": cost.get("cost_per_hit_usd"),
        "total_tokens": cost.get("total_tokens"),
        "avg_tokens_per_case": cost.get("avg_tokens_per_case"),
        "total_llm_calls": cost.get("total_llm_calls"),
        "n_cases_with_usage": cost.get("n_cases_with_usage"),
        "source": payload.get("note") or payload.get("generated_at"),
    }
