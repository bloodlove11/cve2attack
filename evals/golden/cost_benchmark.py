"""Compare Live eval result JSONs on quality vs API cost / tokens.

Usage::

    python -m evals.golden.cost_benchmark \\
      evals/golden/published/circl_test_live_refine.json \\
      evals/golden/results/circl_refine_qwen38_27b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.golden.cost_metrics import cost_row_from_payload


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


def format_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table: model × quality × cost."""
    lines = [
        "| model | n | hit | recall@k | P@k | total cost | $/case | $/hit | tokens | tok/case | llm calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {n} | {hit} | {rec} | {prec} | {cost} | {avg} | {cph} | {tok} | {atk} | {calls} |".format(
                model=r.get("model") or "?",
                n=_fmt_int(r.get("n")),
                hit=_fmt_num(r.get("hit")),
                rec=_fmt_num(r.get("recall_at_k")),
                prec=_fmt_num(r.get("precision_at_k")),
                cost=_fmt_money(r.get("total_cost_usd")),
                avg=_fmt_money(r.get("avg_cost_per_case_usd")),
                cph=_fmt_money(r.get("cost_per_hit_usd")),
                tok=_fmt_int(r.get("total_tokens")),
                atk=_fmt_num(r.get("avg_tokens_per_case"), 1),
                calls=_fmt_int(r.get("total_llm_calls")),
            )
        )
    lines.append("")
    lines.append(
        "_Cost from provider ``usage.cost`` (USD). Free-tier models often report "
        "`$0.0000`. Rows without usage metadata show — for cost columns._"
    )
    return "\n".join(lines)


def load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark Live eval cost / tokens per model across result JSONs"
    )
    p.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="One or more golden results JSON files (published/ or results/)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the comparison rows as JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    for path in args.results:
        payload = load_payload(path)
        row = cost_row_from_payload(payload)
        row["path"] = str(path)
        rows.append(row)
    print(format_table(rows))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
