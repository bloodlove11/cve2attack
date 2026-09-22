"""CLI / programmatic golden-eval runner (Live LLM agent only).

Examples::

    python -m evals.golden.runner --mode agent --task cve_to_attack --limit 10
    python -m evals.golden.runner --mode agent --task cve_to_attack --heldout --protocol circl_test
    python -m evals.golden.runner --mode agent --task cve_to_attack --limit 20 --seeds 42,43,44 --jobs 1
    python -m evals.golden.runner --mode agent --task cve_to_attack --heldout --protocol circl_test --no-exploit-refine
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

# Local-package imports below need the project root on sys.path when this
# module runs standalone (``python -m evals.golden.runner``) outside an
# editable install / pytest's configured pythonpath.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.golden.eval_rag import build_triage_context  # noqa: E402
from evals.golden.live_attack import (  # noqa: E402
    LIVE_PIPELINE_VERSION,
    default_llm_chat,
    predict_cve_to_attack,
)
from evals.golden.score import (  # noqa: E402
    load_cases,
    score_attack_with_heads,
    score_triage,
    summarize,
)

RESULTS_DIR = Path(__file__).with_name("results")
LATEST_JSON = RESULTS_DIR / "latest.json"
LATEST_MD = RESULTS_DIR / "latest.md"

VALID_MODES = frozenset({"agent"})
ProgressCb = Callable[[int, int, dict[str, Any] | None], None]

DEFAULT_JOBS_CAP = 8


def default_jobs_cap() -> int:
    """Concurrency cap: env ``GOLDEN_JOBS_CAP`` or default 8 (Live CIRCL speed)."""
    raw = os.getenv("GOLDEN_JOBS_CAP", "").strip()
    if not raw:
        return DEFAULT_JOBS_CAP
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_JOBS_CAP


def resolve_jobs(n_cases: int, jobs: int | None = None) -> int:
    """Effective worker count: ``min(n_cases, cap)`` unless ``jobs`` overrides.

    - ``jobs is None`` → auto ``min(n_cases, default_jobs_cap())`` (at least 1 when n>0)
    - ``jobs == 1`` → serial (repro)
    - ``jobs > 1`` → explicit override (still capped by n_cases)
    """
    n = max(0, int(n_cases))
    if n == 0:
        return 1
    if jobs is None:
        return max(1, min(n, default_jobs_cap()))
    j = max(1, int(jobs))
    return max(1, min(n, j))


def atomic_write_text(path: Path, content: str) -> None:
    """Write via temp file + replace so latest.* is never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _evaluate_case(
    case: dict[str, Any],
    predict: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Predict + score one case (safe to call from a worker thread)."""
    try:
        prediction = predict(case)
    except Exception as exc:  # noqa: BLE001
        prediction = {"error": str(exc), "skipped": True}
    row = _score_case(case, prediction)
    if prediction.get("error"):
        row["error"] = prediction["error"]
    return row


def _run_cases_parallel(
    selected: list[dict[str, Any]],
    predict: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    jobs: int,
    progress: ProgressCb | None = None,
) -> list[dict[str, Any]]:
    """Fan out cases (not heads). Results ordered by original case index."""
    n = len(selected)
    if n == 0:
        return []
    workers = resolve_jobs(n, jobs)
    if workers <= 1:
        rows: list[dict[str, Any]] = []
        for i, case in enumerate(selected):
            row = _evaluate_case(case, predict)
            rows.append(row)
            if progress is not None:
                progress(i + 1, n, row)
        return rows

    rows_by_idx: list[dict[str, Any] | None] = [None] * n
    lock = threading.Lock()
    done = 0

    def _work(idx: int, case: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return idx, _evaluate_case(case, predict)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, i, case) for i, case in enumerate(selected)]
        for fut in as_completed(futures):
            idx, row = fut.result()
            rows_by_idx[idx] = row
            if progress is not None:
                with lock:
                    done += 1
                    completed = done
                progress(completed, n, row)

    return [r for r in rows_by_idx if r is not None]



def _filter_cases(
    cases: list[dict[str, Any]],
    *,
    task: str,
    limit: int | None,
    seed: int | None,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter by task, optional frozen case_ids, then optional shuffle+limit.

    When ``case_ids`` is set, cases are returned in that freeze order (missing
    ids skipped). ``seed`` is ignored. ``limit`` takes a prefix of the freeze
    so CIRCL smokes can run ``--heldout --protocol circl_test --limit 10``.
    """
    if task != "all":
        cases = [c for c in cases if c.get("task") == task]
    else:
        cases = list(cases)
    if case_ids is not None:
        by_id = {c.get("id"): c for c in cases}
        ordered = [by_id[i] for i in case_ids if i in by_id]
        if limit is not None and limit > 0:
            return ordered[:limit]
        return ordered
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(cases)
    if limit is not None and limit > 0:
        cases = cases[:limit]
    return cases



def _triage_prompt(
    inp: dict[str, Any],
    *,
    rag_context: str | None = None,
) -> list[dict[str, str]]:
    payload = {
        "description": inp.get("description"),
        "cvss_score": inp.get("cvss_score"),
        "epss": inp.get("epss"),
        "in_kev": inp.get("in_kev", inp.get("kev")),
        "asset_context": inp.get("asset_context", inp.get("asset")),
        "critical_asset": inp.get("critical_asset", inp.get("critical")),
    }
    system = (
        "You are a vulnerability triage assistant. "
        "Reply with ONLY a JSON object: "
        '{"triage_label": "ACT"|"ATTEND"|"TRACK"}. '
        "ACT = remediate now; ATTEND = prioritize soon; TRACK = monitor. "
        "Ground the decision in the numeric signals (KEV, CVSS, EPSS, critical asset). "
        "Any retrieved policy hint is advisory only — not a forced label."
    )
    parts = [
        "Triage this CVE using the signals below.",
        f"{json.dumps(payload, indent=2)}",
    ]
    if rag_context and rag_context.strip():
        parts.extend(["", rag_context.strip()])
    user = "\n".join(parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



def _agent_predict(
    case: dict[str, Any],
    *,
    use_rag: bool = True,
    protocol: str | None = None,
    refine_exploitation: bool = True,
    use_prep_cache: bool | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Live agent: LLM inference with NVD + ATT&CK-doc RAG (not CTID gold).

    ``model`` overrides ``EXPLABS_MODEL`` for this call only (passed straight
    through to ``default_llm_chat`` / ``predict_cve_to_attack``); ``None``
    keeps today's env-var resolution.
    """
    task = case.get("task")
    inp = case.get("input") or {}
    # Case-level protocol (CIRCL gold) wins over runner default when set
    protocol = protocol or case.get("protocol")
    if task == "cve_triage":
        rag_ctx = build_triage_context(inp) if use_rag else None
        messages = _triage_prompt(inp, rag_context=rag_ctx)
        try:
            parsed = default_llm_chat(messages, model=model)
        except Exception as exc:  # noqa: BLE001
            return {
                "triage_label": "",
                "mode": "agent_llm_failed",
                "error": str(exc),
                "skipped": True,
            }
        label = str(parsed.get("triage_label") or "").strip().upper()
        valid = label in {"ACT", "ATTEND", "TRACK"}
        out: dict[str, Any] = {
            "triage_label": label,
            "mode": "agent_llm" if valid else "agent_llm_invalid",
            "_raw": parsed.get("_raw"),
            "llm_calls": 1,
        }
        if not valid:
            out["error"] = "invalid triage_label from LLM"
        if parsed.get("_usage") is not None:
            out["usage"] = parsed["_usage"]
        if parsed.get("_model") is not None:
            out["model"] = parsed["_model"]
        return out

    if task == "cve_to_attack":
        # Shared Live ATT&CK pipeline (same code Chat product path calls)
        pred = predict_cve_to_attack(
            str(inp.get("cve_id") or ""),
            description=(
                str(inp.get("description")) if inp.get("description") else None
            ),
            use_rag=use_rag,
            protocol=protocol,
            refine_exploitation=refine_exploitation,
            use_prep_cache=use_prep_cache,
            extra_cwes=list(inp.get("cwes") or []) or None,
            model=model,
        )
        # Scored prediction: drop pipeline bookkeeping / rationale
        out: dict[str, Any] = {
            "exploitation_techniques": pred.get("exploitation_techniques") or [],
            "primary_impact": pred.get("primary_impact") or [],
            "attack_techniques": pred.get("attack_techniques") or [],
            "mode": pred.get("mode") or "agent_llm",
            "enrichment_source": pred.get("enrichment_source"),
            "llm_calls": pred.get("llm_calls"),
            "refine_exploitation": pred.get("refine_exploitation"),
            "prep_cache_hit": pred.get("prep_cache_hit"),
            "model": pred.get("model"),
            "pipeline_version": pred.get("pipeline_version"),
        }
        if pred.get("usage") is not None:
            out["usage"] = pred["usage"]
        if pred.get("error"):
            out["error"] = pred["error"]
        if pred.get("_raw") is not None:
            out["_raw"] = pred["_raw"]
        if pred.get("mode") == "agent_llm_failed":
            out["skipped"] = True
        return out

    return {"error": f"unsupported task: {task}", "skipped": True}


def _score_case(
    case: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    task = case.get("task")
    expected = case.get("expected") or {}
    row: dict[str, Any] = {
        "id": case.get("id"),
        "task": task,
        "prediction": {k: v for k, v in prediction.items() if k != "_raw"},
        "expected": expected,
        "skipped": bool(prediction.get("skipped"))
        or prediction.get("mode") == "agent_llm_failed",
        "scores": {},
    }
    if row["skipped"]:
        row["scores"] = {}
        return row

    if task == "cve_triage":
        pred_label = str(prediction.get("triage_label") or "")
        row["scores"] = score_triage(pred_label, str(expected.get("triage_label") or ""))
        row["correct"] = row["scores"].get("label_exact_match", 0.0) >= 1.0
    elif task == "cve_to_attack":
        pred_techs = prediction.get("attack_techniques") or []
        if isinstance(pred_techs, str):
            pred_techs = [pred_techs]
        exp_techs = expected.get("attack_techniques") or []
        pred_pi = prediction.get("primary_impact") or []
        if isinstance(pred_pi, str):
            pred_pi = [pred_pi]
        pred_ex = prediction.get("exploitation_techniques") or []
        if isinstance(pred_ex, str):
            pred_ex = [pred_ex]
        # Empty exploitation gold → head score omitted (N/A)
        exp_pi = expected.get("primary_impact")
        exp_ex = expected.get("exploitation_techniques")
        row["scores"] = score_attack_with_heads(
            predicted_techniques=list(pred_techs),
            expected_techniques=list(exp_techs),
            predicted_primary_impact=list(pred_pi),
            expected_primary_impact=list(exp_pi) if exp_pi is not None else None,
            predicted_exploitation=list(pred_ex),
            expected_exploitation=list(exp_ex) if exp_ex is not None else None,
            k=5,
        )
        row["correct"] = row["scores"].get("hit", 0.0) >= 1.0
    return row


# (raw summarize() key, renamed "*_rate" alias) for the optional CTID head
# metrics. Each pair is present only when gold for that head existed on at
# least one scored case (score_technique_head returns None otherwise, so
# summarize() never sees the raw key).
_ATTACK_HEAD_ALIASES: tuple[tuple[str, str], ...] = (
    ("canonical_hit", "canonical_hit_rate"),
    ("primary_impact_hit", "primary_impact_hit_rate"),
    ("exploitation_hit", "exploitation_hit_rate"),
    ("primary_impact_canonical_hit", "primary_impact_canonical_hit_rate"),
    ("exploitation_canonical_hit", "exploitation_canonical_hit_rate"),
)


def _attack_head_aliases(att: dict[str, float]) -> dict[str, float]:
    """``{alias: value}`` for every head metric present in ``att``.

    Shared by ``attack_block`` and the flat ``attack_<alias>`` keys in
    ``metrics`` below (previously two separately-maintained if-chains).
    """
    return {alias: att[raw] for raw, alias in _ATTACK_HEAD_ALIASES if raw in att}


def _aggregate(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    scored = [r for r in rows if not r.get("skipped")]
    triage_scores = [
        r["scores"] for r in scored if r.get("task") == "cve_triage" and r.get("scores")
    ]
    attack_scores = [
        r["scores"] for r in scored if r.get("task") == "cve_to_attack" and r.get("scores")
    ]
    n_failed = sum(
        1
        for r in rows
        if (r.get("prediction") or {}).get("mode") == "agent_llm_failed"
    )
    metrics: dict[str, Any] = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_skipped": sum(1 for r in rows if r.get("skipped")),
        "n_failed": n_failed,
        "mode": mode,
    }

    triage_acc: float | None = None
    attack_hit: float | None = None

    if triage_scores:
        tri = summarize(triage_scores)
        triage_acc = tri.get("label_exact_match")
        metrics["triage"] = {
            **tri,
            "accuracy": triage_acc,
            "n": len(triage_scores),
        }
        # Top-level convenience (split, not blended with ATT&CK)
        metrics["triage_accuracy"] = triage_acc

    if attack_scores:
        att = summarize(attack_scores)
        attack_hit = att.get("hit")
        aliases = _attack_head_aliases(att)
        attack_block: dict[str, Any] = {
            **att,
            "hit_rate": attack_hit,
            "n": len(attack_scores),
            **aliases,
        }
        metrics["attack"] = attack_block
        metrics["attack_hit_rate"] = attack_hit
        metrics["attack_recall_at_k"] = att.get("recall_at_k")
        metrics.update({f"attack_{alias}": v for alias, v in aliases.items()})
        if "canonical_hit" in att:
            metrics["attack_canonical_recall_at_k"] = att.get("canonical_recall_at_k")
        # Legacy flat keys (attack-only convenience; do not treat as triage accuracy)
        metrics["recall_at_k"] = att.get("recall_at_k")
        metrics["precision_at_k"] = att.get("precision_at_k")
        metrics["hit_rate"] = attack_hit
        if "canonical_hit" in att:
            metrics["canonical_hit_rate"] = att["canonical_hit"]
            metrics["canonical_recall_at_k"] = att.get("canonical_recall_at_k")

    # ``accuracy`` means triage only; never blend it with the ATT&CK hit rate.
    # Attack-only runs get neither ``accuracy`` nor ``macro_score``.
    if triage_scores and not attack_scores:
        metrics["accuracy"] = triage_acc
    elif triage_scores and attack_scores:
        # Mixed run: keep split metrics, expose macro_score, leave accuracy unset
        if triage_acc is not None and attack_hit is not None:
            metrics["macro_score"] = (triage_acc + attack_hit) / 2.0

    # LLM cost / token benchmarking (provider usage.cost when present)
    from evals.golden.cost_metrics import aggregate_cost_metrics

    hit_for_cost = attack_hit
    if hit_for_cost is None and triage_acc is not None and not attack_scores:
        hit_for_cost = triage_acc
    metrics["cost"] = aggregate_cost_metrics(rows, hit_rate=hit_for_cost)

    return metrics


# (report label, metrics["attack"] key, flat metrics[] fallback key or None).
# The fallback exists for older published result JSON that only carried some
# metrics at the flat top level; skipped for attack.hit_rate, which has its
# own "(n=...)" suffix and stays a one-off above.
_ATTACK_REPORT_ROWS: tuple[tuple[str, str, str | None], ...] = (
    ("attack.recall_at_k", "recall_at_k", "attack_recall_at_k"),
    ("attack.precision_at_k", "precision_at_k", None),
    ("attack.canonical_hit_rate", "canonical_hit_rate", "attack_canonical_hit_rate"),
    (
        "attack.canonical_recall_at_k",
        "canonical_recall_at_k",
        "attack_canonical_recall_at_k",
    ),
    ("attack.primary_impact_hit_rate", "primary_impact_hit_rate", None),
    ("attack.exploitation_hit_rate", "exploitation_hit_rate", None),
    (
        "attack.primary_impact_canonical_hit_rate",
        "primary_impact_canonical_hit_rate",
        None,
    ),
    (
        "attack.exploitation_canonical_hit_rate",
        "exploitation_canonical_hit_rate",
        None,
    ),
)


def markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics") or {}
    lines = [
        "# Golden eval results",
        "",
        f"- mode: `{payload.get('mode')}`",
        f"- task: `{payload.get('task')}`",
        f"- model: `{payload.get('model') or (metrics.get('cost') or {}).get('model') or '—'}`",
        f"- limit: `{payload.get('limit')}`",
        f"- seed: `{payload.get('seed')}`",
        f"- jobs: `{payload.get('jobs')}`",
        f"- eval_rag: `{payload.get('eval_rag', (payload.get('metrics') or {}).get('eval_rag'))}`",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- n_total: {metrics.get('n_total', 0)}",
        f"- n_scored: {metrics.get('n_scored', 0)}",
        f"- n_skipped: {metrics.get('n_skipped', 0)}",
        f"- n_failed: {metrics.get('n_failed', 0)}",
        "",
        "## Split metrics",
        "",
    ]
    tri = metrics.get("triage") or {}
    att = metrics.get("attack") or {}
    if tri:
        acc = tri.get("accuracy", metrics.get("triage_accuracy"))
        lines.append(
            f"- **triage.accuracy**: {acc:.4f} (n={tri.get('n', '?')})"
            if acc is not None
            else f"- **triage**: n={tri.get('n', 0)}"
        )
    if att:
        hit = att.get("hit_rate", metrics.get("attack_hit_rate"))
        if hit is not None:
            lines.append(f"- **attack.hit_rate**: {hit:.4f} (n={att.get('n', '?')})")
        for label, att_key, flat_fallback_key in _ATTACK_REPORT_ROWS:
            value = att.get(att_key)
            if value is None and flat_fallback_key:
                value = metrics.get(flat_fallback_key)
            if value is not None:
                lines.append(f"- **{label}**: {value:.4f}")
    if metrics.get("accuracy") is not None:
        lines.append(f"- **accuracy** (triage-only run): {metrics['accuracy']:.4f}")
    if metrics.get("macro_score") is not None:
        lines.append(
            f"- **macro_score** (avg triage.accuracy + attack.hit_rate): "
            f"{metrics['macro_score']:.4f}"
        )
    cost = metrics.get("cost") or {}
    if cost:
        lines.extend(["", "## Cost / tokens (per model)", ""])
        if cost.get("model"):
            lines.append(f"- **model**: `{cost['model']}`")
        if cost.get("total_cost_usd") is not None:
            lines.append(f"- **total_cost_usd**: {cost['total_cost_usd']:.6f}")
        if cost.get("avg_cost_per_case_usd") is not None:
            lines.append(
                f"- **avg_cost_per_case_usd**: {cost['avg_cost_per_case_usd']:.6f}"
            )
        if cost.get("cost_per_hit_usd") is not None:
            lines.append(f"- **cost_per_hit_usd**: {cost['cost_per_hit_usd']:.6f}")
        lines.append(f"- **total_tokens**: {cost.get('total_tokens', 0)}")
        if cost.get("avg_tokens_per_case") is not None:
            lines.append(
                f"- **avg_tokens_per_case**: {cost['avg_tokens_per_case']:.1f}"
            )
        lines.append(f"- **total_llm_calls**: {cost.get('total_llm_calls', 0)}")
        lines.append(
            f"- **n_cases_with_usage**: {cost.get('n_cases_with_usage', 0)} "
            f"(without: {cost.get('n_cases_without_usage', 0)})"
        )
    lines.extend(["", "| id | task | correct | scores |", "|---|---|---|---|"])
    for r in payload.get("results") or []:
        scores = r.get("scores") or {}
        score_s = ", ".join(f"{k}={v:.2f}" for k, v in scores.items()) or (
            "skipped" if r.get("skipped") else ""
        )
        correct = r.get("correct")
        correct_s = "—" if r.get("skipped") else ("yes" if correct else "no")
        lines.append(f"| {r.get('id')} | {r.get('task')} | {correct_s} | {score_s} |")
    lines.append("")
    return "\n".join(lines)


def run_eval(
    *,
    mode: str = "agent",
    task: str = "cve_to_attack",
    limit: int | None = None,
    seed: int | None = None,
    write_results: bool = True,
    progress: ProgressCb | None = None,
    cases: list[dict[str, Any]] | None = None,
    use_rag: bool = True,
    jobs: int | None = None,
    case_ids: list[str] | None = None,
    heldout: bool = False,
    protocol: str | None = None,
    refine_exploitation: bool = True,
    use_prep_cache: bool | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run golden eval and optionally write ``results/latest.{json,md}``.

    Mode ``agent`` (only): live LLM with NVD/Zenodo enrichment + ATT&CK-doc RAG
    (optional leave-one-out neighbor few-shot; NOT CTID gold lookup for the
    target CVE). Non-LLM predictors (Prior / kNN / Policy / KB / Hybrid) were
    removed; this harness scores the Live LLM path only.

    ``use_rag`` (default True) augments agent prompts. Set False via ``--no-rag``
    for bare prompting.

    ``refine_exploitation`` (default True) runs the second exploitation LLM
    pass. Set False via ``--no-exploit-refine`` to roughly halve Live LLM calls
    for fast sweeps.

    ``use_prep_cache`` disk-caches non-LLM prep (default on; ``LIVE_PREP_CACHE=0``
    or ``--no-prep-cache`` disables).

    ``model`` overrides ``EXPLABS_MODEL`` for every LLM call this run makes,
    without mutating the process environment; omit it to keep reading
    ``EXPLABS_MODEL`` as before.

    Parallelism fans out cases (each case still makes sequential Live LLM calls).
    Default workers: ``min(n_cases, GOLDEN_JOBS_CAP|8)``. Pass ``jobs=1`` for
    serial repro, or ``jobs=K`` / env ``GOLDEN_JOBS_CAP`` to override the cap.
    Results are always re-ordered by original case order before aggregate/write.

    Metrics always split ``triage.accuracy`` vs ``attack.hit_rate`` /
    ``attack.recall_at_k``. Live-agent ATT&CK also reports
    ``attack.primary_impact_hit_rate`` / ``attack.exploitation_hit_rate``
    when CTID head gold is present. Mixed runs expose ``macro_score``
    (not ``accuracy``).
    """
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    task = task.lower().strip()
    if task not in {"cve_triage", "cve_to_attack", "all"}:
        raise ValueError(f"task must be cve_triage|cve_to_attack|all, got {task!r}")

    rag_effective = bool(use_rag)
    refine_eff = bool(refine_exploitation)

    from evals.golden.heldout import (
        DEFAULT_HELDOUT_PROTOCOL,
        PROTOCOL_CIRCL_TEST,
        heldout_case_ids,
    )

    effective_protocol = (protocol or "").strip() or None
    if heldout and not effective_protocol:
        effective_protocol = DEFAULT_HELDOUT_PROTOCOL

    if cases is not None:
        all_cases = cases
    elif effective_protocol == PROTOCOL_CIRCL_TEST:
        from evals.golden.circl_import import load_circl_cases

        # Official CIRCL test only; never score train as if it were test.
        all_cases = load_circl_cases(split="test")
    else:
        all_cases = load_cases()

    effective_case_ids = case_ids
    # ``circl_test`` always uses the frozen official test IDs (n=121).
    # ``--heldout`` is implied so ``--protocol circl_test --limit 10`` is a
    # prefix of that freeze, not CIRCL train.
    freeze_circl = (
        effective_protocol == PROTOCOL_CIRCL_TEST
        and cases is None
        and case_ids is None
    )
    if heldout or freeze_circl:
        effective_case_ids = heldout_case_ids(protocol=effective_protocol)
        heldout = True
        # Held-out freeze is attack-only by default
        if task == "cve_triage":
            raise ValueError("--heldout is for cve_to_attack (or all with attack ids)")
        if task == "all":
            task = "cve_to_attack"
    selected = _filter_cases(
        all_cases,
        task=task,
        limit=limit,
        seed=seed,
        case_ids=effective_case_ids,
    )

    predict = partial(
        _agent_predict,
        use_rag=rag_effective,
        protocol=effective_protocol,
        refine_exploitation=refine_eff,
        use_prep_cache=use_prep_cache,
        model=model,
    )
    effective_jobs = resolve_jobs(len(selected), jobs)
    rows = _run_cases_parallel(
        selected,
        predict,
        jobs=effective_jobs,
        progress=progress,
    )

    metrics = _aggregate(rows, mode=mode)
    metrics["eval_rag"] = rag_effective
    metrics["jobs"] = effective_jobs
    metrics["refine_exploitation"] = refine_eff
    try:
        from src.llm.client import get_model

        resolved_model = model or get_model()
    except Exception:  # noqa: BLE001
        resolved_model = model
    if resolved_model and isinstance(metrics.get("cost"), dict):
        metrics["cost"]["model"] = metrics["cost"].get("model") or resolved_model
    payload: dict[str, Any] = {
        "mode": mode,
        "task": task,
        "model": resolved_model,
        "limit": limit if effective_case_ids is None else len(selected),
        "seed": seed if effective_case_ids is None else None,
        "jobs": effective_jobs,
        "eval_rag": rag_effective,
        "refine_exploitation": refine_eff,
        "prep_cache": use_prep_cache if use_prep_cache is not None else True,
        "heldout": bool(heldout) or effective_case_ids is not None,
        "protocol": effective_protocol,
        "pipeline_version": LIVE_PIPELINE_VERSION,
        "case_ids": [r.get("id") for r in rows],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "results": rows,
    }

    if write_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(LATEST_JSON, json.dumps(payload, indent=2) + "\n")
        atomic_write_text(LATEST_MD, markdown_report(payload))

    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    print(markdown_report(payload))


def build_parser() -> argparse.ArgumentParser:
    """CLI flags for ``python -m evals.golden.runner``."""
    p = argparse.ArgumentParser(description="Golden cyber eval runner")
    p.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default="agent",
        help="Live LLM agent only (mode=agent)",
    )
    p.add_argument(
        "--task",
        choices=["cve_triage", "cve_to_attack", "all"],
        default="cve_to_attack",
    )
    p.add_argument("--limit", type=int, default=None, help="Max cases to evaluate")
    p.add_argument("--seed", type=int, default=None, help="Shuffle seed before limit")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated seeds for multi-seed Live reporting "
            "(e.g. 42,43,44). Runs each seed with --jobs 1 by default via "
            "evals.golden.multi_seed; single --seed path unchanged."
        ),
    )
    p.add_argument(
        "--heldout",
        action="store_true",
        help=(
            "Use frozen case IDs for the selected --protocol "
            "(default deepseek_live_n20 from heldout_ids.json; "
            "circl_test from heldout_circl_test_ids.json). "
            "Ignores --seed shuffle; --limit takes a prefix of the freeze."
        ),
    )
    p.add_argument(
        "--protocol",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Held-out / gold protocol: deepseek_live_n20 (default with --heldout) "
            "or circl_test (CIRCL HF official test split; train-only neighbors). "
            "circl_test always loads the frozen official test IDs (never train). "
            "Headline reporting uses circl_test (n=121). "
            "deepseek_live_n20 is a smoke list, not the number of record."
        ),
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write results/latest.*",
    )
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable CTID neighbor few-shot; agent still uses NVD + ATT&CK-doc candidates (default: full RAG on)",
    )
    p.add_argument(
        "--no-exploit-refine",
        action="store_true",
        help=(
            "Skip the second exploitation-refine LLM call "
            "(~halves Live API calls for fast CIRCL sweeps)"
        ),
    )
    p.add_argument(
        "--no-prep-cache",
        action="store_true",
        help="Disable disk cache for non-LLM Live prep (enrich/tech/neighbors)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel case workers. Default: min(n_cases, GOLDEN_JOBS_CAP|8). "
            "Use --jobs 1 for serial repro; --jobs K to override."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Explicit model id for this run (default: EXPLABS_MODEL env var)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run eval, print markdown summary."""
    args = build_parser().parse_args(argv)
    if args.seeds:
        from evals.golden.multi_seed import run_multi_seed

        # Multi-seed: force serial jobs unless explicitly overridden
        jobs = 1 if args.jobs is None else args.jobs
        agg = run_multi_seed(
            mode=args.mode,
            task=args.task,
            limit=args.limit,
            seeds=args.seeds,
            write_results=not args.no_write,
            use_rag=not args.no_rag,
            jobs=jobs,
            heldout=args.heldout,
            protocol=args.protocol,
            refine_exploitation=not args.no_exploit_refine,
            use_prep_cache=False if args.no_prep_cache else None,
            model=args.model,
        )
        from evals.golden.heldout import format_mean_std

        print(format_mean_std(agg))
        return 0

    payload = run_eval(
        mode=args.mode,
        task=args.task,
        limit=args.limit,
        seed=args.seed,
        write_results=not args.no_write,
        use_rag=not args.no_rag,
        jobs=args.jobs,
        heldout=args.heldout,
        protocol=args.protocol,
        refine_exploitation=not args.no_exploit_refine,
        use_prep_cache=False if args.no_prep_cache else None,
        model=args.model,
    )
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
