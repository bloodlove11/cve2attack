"""Held-out case-ID freeze + multi-seed metric aggregation (eval protocol P0)."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

HELDOUT_PATH = Path(__file__).with_name("heldout_ids.json")
HELDOUT_CIRCL_PATH = Path(__file__).with_name("heldout_circl_test_ids.json")

PROTOCOL_DEEPSEEK_N20 = "deepseek_live_n20"
PROTOCOL_CIRCL_TEST = "circl_test"

# Default ``--heldout`` without ``--protocol`` keeps the DeepSeek n=20 smoke list.
# Headline reporting uses ``circl_test`` (pass --protocol explicitly).
DEFAULT_HELDOUT_PROTOCOL = PROTOCOL_DEEPSEEK_N20

PROTOCOL_PATHS: dict[str, Path] = {
    PROTOCOL_DEEPSEEK_N20: HELDOUT_PATH,
    PROTOCOL_CIRCL_TEST: HELDOUT_CIRCL_PATH,
}

# Keys pulled from run_eval payload["metrics"] (attack block preferred).
ATTACK_METRIC_KEYS = (
    "hit_rate",
    "primary_impact_hit_rate",
    "exploitation_hit_rate",
    "precision_at_k",
    "recall_at_k",
)


def resolve_heldout_path(protocol: str | None = None, path: Path | None = None) -> Path:
    """Resolve held-out JSON path for a protocol name (or explicit path)."""
    if path is not None:
        return path
    proto = (protocol or DEFAULT_HELDOUT_PROTOCOL).strip()
    if proto not in PROTOCOL_PATHS:
        raise ValueError(
            f"Unknown held-out protocol {proto!r}; "
            f"expected one of {sorted(PROTOCOL_PATHS)}"
        )
    return PROTOCOL_PATHS[proto]


def load_heldout(
    path: Path | None = None,
    *,
    protocol: str | None = None,
) -> dict[str, Any]:
    """Load frozen held-out metadata + case_ids.

    ``protocol`` selects ``deepseek_live_n20`` (default) or ``circl_test``.
    Explicit ``path`` wins over protocol.
    """
    p = resolve_heldout_path(protocol=protocol, path=path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"heldout file must be a JSON object: {p}")
    ids = data.get("case_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"heldout file missing non-empty case_ids: {p}")
    data["case_ids"] = [str(x) for x in ids]
    return data


def heldout_case_ids(
    path: Path | None = None,
    *,
    protocol: str | None = None,
) -> list[str]:
    """Return the frozen ordered case id list for the protocol."""
    return list(load_heldout(path, protocol=protocol)["case_ids"])

def parse_seeds(raw: str | list[int] | None) -> list[int]:
    """Parse ``\"42,43,44\"`` or ``[42, 43, 44]`` into a non-empty int list."""
    if raw is None:
        raise ValueError("seeds required")
    if isinstance(raw, list):
        seeds = [int(s) for s in raw]
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if not parts:
            raise ValueError(f"empty seeds string: {raw!r}")
        seeds = [int(p) for p in parts]
    if not seeds:
        raise ValueError("seeds list is empty")
    return seeds


def _metric_from_payload(payload: dict[str, Any], key: str) -> float | None:
    """Read an attack metric from a run_eval payload (nested or flat)."""
    metrics = payload.get("metrics") or {}
    attack = metrics.get("attack") or {}
    if key in attack and attack[key] is not None:
        return float(attack[key])
    flat_aliases = {
        "hit_rate": ("hit_rate", "attack_hit_rate"),
        "precision_at_k": ("precision_at_k",),
        "recall_at_k": ("recall_at_k", "attack_recall_at_k"),
        "primary_impact_hit_rate": (
            "attack_primary_impact_hit_rate",
            "primary_impact_hit_rate",
        ),
        "exploitation_hit_rate": (
            "attack_exploitation_hit_rate",
            "exploitation_hit_rate",
        ),
    }
    for alias in flat_aliases.get(key, (key,)):
        if alias in metrics and metrics[alias] is not None:
            return float(metrics[alias])
        if alias in attack and attack[alias] is not None:
            return float(attack[alias])
    raw_map = {
        "hit_rate": "hit",
        "primary_impact_hit_rate": "primary_impact_hit",
        "exploitation_hit_rate": "exploitation_hit",
    }
    raw_key = raw_map.get(key)
    if raw_key and raw_key in attack and attack[raw_key] is not None:
        return float(attack[raw_key])
    return None


def extract_attack_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Pull the dual-metric suite from one Live/golden payload."""
    out: dict[str, float] = {}
    for key in ATTACK_METRIC_KEYS:
        val = _metric_from_payload(payload, key)
        if val is not None:
            out[key] = val
    return out


def _mean_std(vals: list[float]) -> dict[str, float]:
    mean = statistics.fmean(vals)
    if len(vals) >= 2:
        std = statistics.stdev(vals)
    else:
        std = 0.0
    return {"mean": mean, "std": std, "n": float(len(vals))}


def aggregate_seed_metrics(
    payloads: list[dict[str, Any]],
    *,
    keys: tuple[str, ...] = ATTACK_METRIC_KEYS,
) -> dict[str, Any]:
    """Aggregate mean±std across multi-seed run_eval payloads.

    Returns::

        {
          "n_seeds": N,
          "seeds": [...],
          "metrics": {
            "hit_rate": {"mean": ..., "std": ..., "n": ...},
            ...
          },
          "per_seed": [{"seed": 42, "metrics": {...}}, ...],
        }
    """
    if not payloads:
        return {"n_seeds": 0, "seeds": [], "metrics": {}, "per_seed": []}

    per_seed: list[dict[str, Any]] = []
    buckets: dict[str, list[float]] = {k: [] for k in keys}

    for payload in payloads:
        seed = payload.get("seed")
        mets = extract_attack_metrics(payload)
        per_seed.append({"seed": seed, "metrics": mets})
        for k in keys:
            if k in mets:
                buckets[k].append(mets[k])

    agg: dict[str, dict[str, float]] = {}
    for k, vals in buckets.items():
        if vals:
            agg[k] = _mean_std(vals)

    seeds = [p.get("seed") for p in payloads]
    return {
        "n_seeds": len(payloads),
        "seeds": seeds,
        "metrics": agg,
        "per_seed": per_seed,
    }


def format_mean_std(agg: dict[str, Any], *, digits: int = 4) -> str:
    """Human-readable mean±std lines for the dual-metric suite."""
    lines = [
        f"# Multi-seed aggregate (n_seeds={agg.get('n_seeds', 0)}, "
        f"seeds={agg.get('seeds')})",
        "",
    ]
    metrics = agg.get("metrics") or {}
    for key in ATTACK_METRIC_KEYS:
        block = metrics.get(key)
        if not block:
            lines.append(f"- **{key}**: n/a")
            continue
        mean = block["mean"]
        std = block["std"]
        if math.isnan(mean) or math.isnan(std):
            lines.append(f"- **{key}**: n/a")
        else:
            lines.append(f"- **{key}**: {mean:.{digits}f} ± {std:.{digits}f}")
    lines.append("")
    return "\n".join(lines)
