"""Multi-seed Live / golden reporting (mean±std).

Examples::

    python -m evals.golden.multi_seed --mode agent --task cve_to_attack --limit 20 --seeds 42,43,44 --jobs 1
    python -m evals.golden.multi_seed --mode agent --task cve_to_attack --heldout --seeds 42,43,44 --jobs 1

Leave API-costly Live runs to the caller; this module only orchestrates
``run_eval`` and aggregates metrics. Single-seed path remains
``python -m evals.golden.runner --seed N``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.golden.heldout import (  # noqa: E402
    aggregate_seed_metrics,
    format_mean_std,
    parse_seeds,
)
from evals.golden.runner import (  # noqa: E402
    RESULTS_DIR,
    VALID_MODES,
    atomic_write_text,
    run_eval,
)

MULTI_LATEST_JSON = RESULTS_DIR / "multi_seed_latest.json"
MULTI_LATEST_MD = RESULTS_DIR / "multi_seed_latest.md"


def run_multi_seed(
    *,
    mode: str = "agent",
    task: str = "cve_to_attack",
    limit: int | None = 20,
    seeds: str | list[int] = "42,43,44",
    write_results: bool = True,
    use_rag: bool = True,
    jobs: int = 1,
    heldout: bool = False,
    protocol: str | None = None,
    progress: Any = None,
    refine_exploitation: bool = True,
    use_prep_cache: bool | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run ``run_eval`` once per seed and aggregate mean±std.

    Defaults to ``jobs=1`` for reproducible Live A/Bs (CIRCL-style).
    When ``heldout=True``, each seed still runs the same frozen case list
    (seed is recorded for LLM sampling variance only; case selection is fixed).

    ``refine_exploitation`` / ``use_prep_cache`` / ``model`` are forwarded to
    every seed so ``--seeds`` honours the same flags as a single-seed run.
    """
    seed_list = parse_seeds(seeds)
    payloads: list[dict[str, Any]] = []

    for seed in seed_list:
        payload = run_eval(
            mode=mode,
            task=task,
            limit=None if heldout else limit,
            seed=None if heldout else seed,
            write_results=False,  # write aggregate once below
            use_rag=use_rag,
            jobs=jobs,
            heldout=heldout,
            protocol=protocol,
            progress=progress,
            refine_exploitation=refine_exploitation,
            use_prep_cache=use_prep_cache,
            model=model,
        )
        # Preserve which seed this run represents (heldout clears seed in payload)
        payload = dict(payload)
        payload["seed"] = seed
        payloads.append(payload)

    agg = aggregate_seed_metrics(payloads)
    agg["mode"] = mode
    agg["task"] = task
    agg["limit"] = limit
    agg["jobs"] = jobs
    agg["heldout"] = heldout
    agg["protocol"] = protocol
    agg["eval_rag"] = use_rag
    agg["refine_exploitation"] = refine_exploitation
    agg["generated_at"] = datetime.now(timezone.utc).isoformat()

    if write_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            **agg,
            "runs": [
                {
                    "seed": p.get("seed"),
                    "metrics": p.get("metrics"),
                    "case_ids": p.get("case_ids"),
                }
                for p in payloads
            ],
        }
        atomic_write_text(MULTI_LATEST_JSON, json.dumps(out, indent=2) + "\n")
        atomic_write_text(MULTI_LATEST_MD, format_mean_std(agg))

    return agg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-seed golden eval (mean±std for hit / impact / exploit / P@k / recall@k)"
    )
    p.add_argument("--mode", choices=sorted(VALID_MODES), default="agent")
    p.add_argument(
        "--task",
        choices=["cve_triage", "cve_to_attack", "all"],
        default="cve_to_attack",
    )
    p.add_argument("--limit", type=int, default=20)
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (default 42,43,44)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Case workers per seed (default 1 for reproducible Live A/Bs)",
    )
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument(
        "--no-exploit-refine",
        action="store_true",
        help="Skip the second exploitation-refine LLM call on every seed",
    )
    p.add_argument(
        "--no-prep-cache",
        action="store_true",
        help="Disable the disk cache for non-LLM Live prep on every seed",
    )
    p.add_argument(
        "--heldout",
        action="store_true",
        help="Use frozen held-out case list for --protocol (same cases each seed)",
    )
    p.add_argument(
        "--protocol",
        type=str,
        default=None,
        metavar="NAME",
        help="Held-out protocol: deepseek_live_n20 (default) or circl_test",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Explicit model id for every seed (default: EXPLABS_MODEL env var)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agg = run_multi_seed(
        mode=args.mode,
        task=args.task,
        limit=args.limit,
        seeds=args.seeds,
        write_results=not args.no_write,
        use_rag=not args.no_rag,
        jobs=args.jobs,
        heldout=args.heldout,
        protocol=args.protocol,
        refine_exploitation=not args.no_exploit_refine,
        use_prep_cache=False if args.no_prep_cache else None,
        model=args.model,
    )
    print(format_mean_std(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
