"""Frozen golden-eval grid: models × RAG vs no-RAG, with cost capture.

Default slice: first N CIRCL test IDs (same list for every cell). This is a
comparative budget slice, not the of-record n=121 run.

    python -m evals.golden.model_grid \\
      --models deepseek-v4-flash,qwen3.8-27b,gpt-5.6-luna \\
      --n 20 --jobs 3
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.golden.cost_benchmark import format_table
from evals.golden.cost_metrics import cost_row_from_payload
from evals.golden.heldout import heldout_case_ids
from evals.golden.runner import RESULTS_DIR, run_eval

DEFAULT_MODELS = ("deepseek-v4-flash", "qwen3.8-27b", "gpt-5.6-luna")
OUT_DIR = Path("evals/golden/results/model_grid")
PUB_JSON = Path("evals/golden/published/circl_model_rag_cost_grid.json")
PUB_MD = Path("docs/CIRCL_MODEL_RAG_COST.md")

# ExperientialLabs catalog (host-managed waterfall), USD per 1M tokens.
# Refresh via GET /api/models/<slug> if rates change.
CATALOG_USD_PER_M: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.042448, 0.084896),
    "qwen3.8-27b": (0.32, 2.40),
    "gpt-5.6-luna": (0.20, 1.20),
}


def _slug(model: str, use_rag: bool) -> str:
    rag = "rag" if use_rag else "norag"
    safe = model.replace("/", "-")
    return f"{safe}_{rag}"


def _cell_path(model: str, use_rag: bool, n: int) -> Path:
    return OUT_DIR / f"circl_n{n}_{_slug(model, use_rag)}.json"


def _n_ok(payload: dict[str, Any]) -> int:
    n = 0
    for r in payload.get("results") or []:
        pred = r.get("prediction") or {}
        if r.get("error"):
            continue
        if isinstance(pred, dict) and pred.get("mode") == "agent_llm":
            n += 1
    return n


def list_price_cost(prompt: int, completion: int, model: str) -> float | None:
    rates = CATALOG_USD_PER_M.get(model)
    if not rates:
        return None
    inp, out = rates
    return (prompt / 1e6) * inp + (completion / 1e6) * out


def enrich_row(payload: dict[str, Any], *, model: str, use_rag: bool) -> dict[str, Any]:
    row = cost_row_from_payload(payload)
    row["model"] = model
    row["rag"] = use_rag
    row["n_ok"] = _n_ok(payload)
    cost = (payload.get("metrics") or {}).get("cost") or {}
    prompt = int(cost.get("total_prompt_tokens") or 0)
    completion = int(cost.get("total_completion_tokens") or 0)
    n_usage = int(cost.get("n_cases_with_usage") or 0)
    row["prompt_tokens"] = prompt
    row["completion_tokens"] = completion
    list_total = list_price_cost(prompt, completion, model)
    row["list_total_cost_usd"] = list_total
    if list_total is not None and n_usage:
        row["list_avg_cost_per_case_usd"] = list_total / n_usage
        hit = row.get("hit")
        if hit and n_usage and hit > 0:
            row["list_cost_per_hit_usd"] = list_total / (hit * n_usage)
        else:
            row["list_cost_per_hit_usd"] = None
    else:
        row["list_avg_cost_per_case_usd"] = None
        row["list_cost_per_hit_usd"] = None
    row["observed_total_cost_usd"] = cost.get("total_cost_usd")
    return row


def format_grid_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | RAG | n OK | hit | recall@5 | P@5 | tokens | list $/case | list full-n121 | observed $ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def money(v: Any) -> str:
        if v is None:
            return "—"
        return f"${float(v):.4f}"

    def num(v: Any, d: int = 3) -> str:
        if v is None:
            return "—"
        return f"{float(v):.{d}f}"

    for r in rows:
        n_ok = r.get("n_ok") or r.get("n") or 0
        avg = r.get("list_avg_cost_per_case_usd")
        full = (avg * 121) if avg is not None else None
        lines.append(
            "| {m} | {rag} | {n} | {hit} | {rec} | {p} | {tok} | {avg} | {full} | {obs} |".format(
                m=r.get("model") or "?",
                rag="on" if r.get("rag") else "off",
                n=int(n_ok),
                hit=num(r.get("hit")),
                rec=num(r.get("recall_at_k")),
                p=num(r.get("precision_at_k")),
                tok=int(r.get("total_tokens") or 0),
                avg=money(avg),
                full=money(full),
                obs=money(r.get("observed_total_cost_usd")),
            )
        )
    lines.append("")
    lines.append(
        "_RAG on = neighbor ICL + ATT&CK-doc retrieval. RAG off = `--no-rag` "
        "(no neighbor few-shot; ATT&CK-doc candidates still present). "
        "List $ uses ExperientialLabs catalog rates × measured tokens. "
        "Observed $ is provider `usage.cost` (often $0 on promo / free tier)._"
    )
    return "\n".join(lines)


def run_cell(
    *,
    model: str,
    use_rag: bool,
    case_ids: list[str],
    jobs: int,
    refine: bool,
) -> dict[str, Any]:
    """Run one grid cell."""
    path = _cell_path(model, use_rag, len(case_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if _n_ok(prev) == len(case_ids):
            print(f"SKIP complete {path.name}", flush=True)
            return prev
    print(f"RUN model={model} rag={use_rag} n={len(case_ids)} jobs={jobs}", flush=True)
    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        use_rag=use_rag,
        jobs=jobs,
        case_ids=case_ids,
        protocol="circl_test",
        refine_exploitation=refine,
        use_prep_cache=True,
        model=model,
    )
    payload["model"] = model
    payload["eval_rag"] = use_rag
    payload["grid_n"] = len(case_ids)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE {path.name} n_ok={_n_ok(payload)} hit="
        f"{((payload.get('metrics') or {}).get('attack') or {}).get('hit_rate')}",
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--n", type=int, default=20, help="Prefix of circl_test IDs")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--no-exploit-refine", action="store_true")
    p.add_argument("--write-docs", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_ids = heldout_case_ids(protocol="circl_test")
    n = min(args.n, len(all_ids))
    case_ids = all_ids[:n]
    refine = not args.no_exploit_refine
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for model in models:
        for use_rag in (True, False):
            try:
                payload = run_cell(
                    model=model,
                    use_rag=use_rag,
                    case_ids=case_ids,
                    jobs=args.jobs,
                    refine=refine,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {model} rag={use_rag}: {exc}", flush=True)
                rows.append(
                    {
                        "model": model,
                        "rag": use_rag,
                        "n_ok": 0,
                        "hit": None,
                        "error": str(exc)[:300],
                    }
                )
                continue
            cells.append({"model": model, "rag": use_rag, "payload": payload})
            rows.append(enrich_row(payload, model=model, use_rag=use_rag))

    table = format_grid_table(rows)
    print(table)
    # also quality-only cost_benchmark table
    print("\nProvider usage (observed):\n")
    print(format_table(rows))

    summary = {
        "protocol": "circl_test",
        "slice": f"prefix_n{n}",
        "case_ids": case_ids,
        "refine_exploitation": refine,
        "models": models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_usd_per_million": {
            k: {"input": v[0], "output": v[1]} for k, v in CATALOG_USD_PER_M.items()
        },
        "rows": rows,
        "note": (
            "Comparative slice (not of-record n=121). RAG off = --no-rag "
            "(no neighbor ICL). List $ from catalog × measured tokens."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"summary_n{n}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.write_docs:
        PUB_JSON.parent.mkdir(parents=True, exist_ok=True)
        PUB_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        PUB_MD.write_text(
            "# CIRCL model × RAG cost comparison\n\n"
            f"**Slice:** first {n} of frozen `circl_test` (not of-record n=121). "
            "Exploit-refine on. Same case IDs for every cell.\n\n"
            + table
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {PUB_JSON} and {PUB_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
