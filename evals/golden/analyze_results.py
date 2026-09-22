"""Analyze golden Live ATT&CK results for error taxonomy / CV report."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in (payload.get("results") or []) if not r.get("skipped")]
    n = len(rows)
    hits = sum(1 for r in rows if r.get("correct"))
    empty_pred = 0
    miss_rows: list[dict[str, Any]] = []
    fp_counter: Counter[str] = Counter()
    fn_counter: Counter[str] = Counter()
    exploit_miss = 0
    exploit_n = 0
    impact_miss = 0
    impact_n = 0

    for r in rows:
        pred = r.get("prediction") or {}
        exp = r.get("expected") or {}
        scores = r.get("scores") or {}
        pred_t = [str(t).upper() for t in (pred.get("attack_techniques") or [])]
        exp_t = [str(t).upper() for t in (exp.get("attack_techniques") or [])]
        if not pred_t:
            empty_pred += 1
        pred_set, exp_set = set(pred_t), set(exp_t)
        for t in pred_set - exp_set:
            fp_counter[t] += 1
        for t in exp_set - pred_set:
            fn_counter[t] += 1
        if not r.get("correct"):
            miss_rows.append(
                {
                    "id": r.get("id"),
                    "predicted": pred_t,
                    "expected": exp_t,
                    "recall_at_k": scores.get("recall_at_k"),
                    "hit": scores.get("hit"),
                }
            )
        if exp.get("exploitation_techniques"):
            exploit_n += 1
            if scores.get("exploitation_hit", 0) < 1:
                exploit_miss += 1
        if exp.get("primary_impact"):
            impact_n += 1
            if scores.get("primary_impact_hit", 0) < 1:
                impact_miss += 1

    return {
        "n_scored": n,
        "hit_rate": hits / n if n else None,
        "empty_prediction_rate": empty_pred / n if n else None,
        "top_false_positives": fp_counter.most_common(15),
        "top_false_negatives": fn_counter.most_common(15),
        "exploitation_miss_rate": (exploit_miss / exploit_n) if exploit_n else None,
        "exploitation_n_with_gold": exploit_n,
        "primary_impact_miss_rate": (impact_miss / impact_n) if impact_n else None,
        "primary_impact_n_with_gold": impact_n,
        "example_misses": miss_rows[:12],
        "metrics": payload.get("metrics"),
        "mode": payload.get("mode"),
        "model": payload.get("model"),
        "protocol": payload.get("protocol"),
        "refine_exploitation": payload.get("refine_exploitation"),
        "jobs": payload.get("jobs"),
        "wall_seconds": payload.get("wall_seconds"),
        "cost": (payload.get("metrics") or {}).get("cost"),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live result error analysis",
        "",
        f"- mode: `{report.get('mode')}` · protocol: `{report.get('protocol')}`",
        f"- n_scored: **{report.get('n_scored')}** · hit_rate: **{report.get('hit_rate')}**",
        f"- empty predictions: **{report.get('empty_prediction_rate')}**",
        f"- refine_exploitation: `{report.get('refine_exploitation')}` · jobs: `{report.get('jobs')}`",
        f"- wall_seconds: `{report.get('wall_seconds')}`",
        "",
        "## Head misses (when gold heads present)",
        "",
        f"- exploitation miss rate: **{report.get('exploitation_miss_rate')}** "
        f"(n={report.get('exploitation_n_with_gold')})",
        f"- primary_impact miss rate: **{report.get('primary_impact_miss_rate')}** "
        f"(n={report.get('primary_impact_n_with_gold')})",
        "",
        "## Top false positives (predicted ∉ gold)",
        "",
    ]
    for tid, c in report.get("top_false_positives") or []:
        lines.append(f"- `{tid}` × {c}")
    lines.extend(["", "## Top false negatives (gold ∉ predicted)", ""])
    for tid, c in report.get("top_false_negatives") or []:
        lines.append(f"- `{tid}` × {c}")
    lines.extend(["", "## Example misses", ""])
    for row in report.get("example_misses") or []:
        lines.append(
            f"- `{row.get('id')}`: pred={row.get('predicted')} · "
            f"gold={row.get('expected')} · recall@k={row.get('recall_at_k')}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_json", type=Path)
    p.add_argument("--md", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)
    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    report = analyze(payload)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = to_markdown(report)
    if args.md:
        args.md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
