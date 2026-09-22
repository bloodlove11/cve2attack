"""Score predictions against ``golden_dataset.json`` (triage + ATT&CK)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GOLD = Path(__file__).with_name("golden_dataset.json")

_TECH_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)


def load_cases() -> list[dict[str, Any]]:
    """Load the curated golden case list from ``golden_dataset.json``."""
    return json.loads(GOLD.read_text())["cases"]


def score_triage(predicted_label: str, expected_label: str) -> dict[str, float]:
    """Exact-match score for ACT / ATTEND / TRACK (case-insensitive).

    Returns:
        ``{"label_exact_match": 1.0 | 0.0}``.
    """
    pred = (predicted_label or "").strip().upper()
    exp = (expected_label or "").strip().upper()
    return {"label_exact_match": 1.0 if pred == exp else 0.0}


def normalize_technique_id(raw: Any) -> str | None:
    """Normalize to uppercase ``T####`` / ``T####.###``, or ``None`` if invalid.

    Format-only; does not follow revoked-by. Use
    :func:`resolve_technique_id` when scoring or comparing labels.
    """
    if raw is None:
        return None
    s = str(raw).strip().split()[0] if str(raw).strip() else ""
    if not s:
        return None
    s = s.upper()
    if _TECH_ID_RE.fullmatch(s):
        return s
    m = re.search(r"T\d{4}(?:\.\d{3})?", s, re.IGNORECASE)
    if not m:
        return None
    s = m.group(0).upper()
    return s if _TECH_ID_RE.fullmatch(s) else None


def resolve_technique_id(raw: Any) -> str | None:
    """Normalize then follow the STIX ``revoked-by`` map to the current ID."""
    from evals.golden.attack_catalog import resolve_technique_id as _resolve

    return _resolve(raw)


def technique_parent_id(tid: str) -> str:
    """Parent technique id (``T1204.002`` → ``T1204``; bare parent unchanged)."""
    return tid.split(".", 1)[0]


def canonical_match(pred: str, exp: str) -> bool:
    """True if IDs are equal or parent↔sub of the same technique.

    ``T1204`` <-> ``T1204.002`` matches. Sibling subs do not match
    (``T1566.001`` ≠ ``T1566.002``).
    """
    p = normalize_technique_id(pred)
    e = normalize_technique_id(exp)
    if not p or not e:
        return False
    if p == e:
        return True
    if technique_parent_id(p) != technique_parent_id(e):
        return False
    # Same family: only parent↔sub (one side must be the bare parent)
    return ("." not in p) or ("." not in e)


def _normalize_tech_list(predicted: list[str] | None) -> list[str]:
    pred: list[str] = []
    for t in predicted or []:
        nid = resolve_technique_id(t)
        if nid and nid not in pred:
            pred.append(nid)
    return pred


def _exact_hits(top: list[str], exp: set[str]) -> int:
    return sum(1 for t in top if t in exp)


def _canonical_hits(top: list[str], exp: set[str]) -> int:
    return sum(1 for t in top if any(canonical_match(t, e) for e in exp))


def score_attack(
    predicted: list[str], expected: list[str], k: int = 5
) -> dict[str, float]:
    """Recall@k / precision@k / hit over MITRE technique IDs (``T####``).

    Exact scores remain the lead metrics. Also returns ``canonical_hit`` /
    ``canonical_recall_at_k`` (parent↔sub soft match; siblings do not match).

    Only tokens starting with ``T`` count; duplicates are dropped. ``hit`` is
    1.0 if any predicted top-k technique is in the expected set.
    """
    exp = {nid for t in expected if t for nid in [resolve_technique_id(t)] if nid}
    top = _normalize_tech_list(predicted)[:k]
    hits = _exact_hits(top, exp)
    recall = hits / len(exp) if exp else 0.0
    precision = hits / len(top) if top else 0.0
    c_hits = _canonical_hits(top, exp)
    c_recall = c_hits / len(exp) if exp else 0.0
    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "hit": 1.0 if hits else 0.0,
        "canonical_hit": 1.0 if c_hits else 0.0,
        "canonical_recall_at_k": c_recall,
    }


def score_technique_head(
    predicted: list[str] | None,
    expected: list[str] | None,
    k: int = 5,
) -> dict[str, float] | None:
    """Hit / recall@k for one CTID head (exploitation or primary_impact).

    Returns ``None`` when expected is missing or empty (score N/A, skipped).
    Hit = at least one predicted ID in the expected set.
    Also includes ``canonical_hit`` / ``canonical_recall_at_k``.
    """
    if not expected:
        return None
    base = score_attack(list(predicted or []), list(expected), k=k)
    return {
        "hit": base["hit"],
        "recall_at_k": base["recall_at_k"],
        "canonical_hit": base["canonical_hit"],
        "canonical_recall_at_k": base["canonical_recall_at_k"],
    }


def score_attack_with_heads(
    *,
    predicted_techniques: list[str],
    expected_techniques: list[str],
    predicted_primary_impact: list[str] | None = None,
    expected_primary_impact: list[str] | None = None,
    predicted_exploitation: list[str] | None = None,
    expected_exploitation: list[str] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Full ATT&CK scores plus optional CTID head hit/recall metrics.

    Head metrics are omitted (not set to 0) when expected gold for that head
    is missing or empty, so aggregators can treat them as N/A.
    """
    scores = score_attack(predicted_techniques, expected_techniques, k=k)

    pi = score_technique_head(
        predicted_primary_impact, expected_primary_impact, k=k
    )
    if pi is not None:
        scores["primary_impact_hit"] = pi["hit"]
        scores["primary_impact_recall_at_k"] = pi["recall_at_k"]
        scores["primary_impact_canonical_hit"] = pi["canonical_hit"]
        scores["primary_impact_canonical_recall_at_k"] = pi["canonical_recall_at_k"]

    ex = score_technique_head(
        predicted_exploitation, expected_exploitation, k=k
    )
    if ex is not None:
        scores["exploitation_hit"] = ex["hit"]
        scores["exploitation_recall_at_k"] = ex["recall_at_k"]
        scores["exploitation_canonical_hit"] = ex["canonical_hit"]
        scores["exploitation_canonical_recall_at_k"] = ex["canonical_recall_at_k"]

    return scores


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each score key across per-case score dicts.

    Keys present only on a subset of cases (e.g. head metrics) are averaged
    over the cases that include them.
    """
    if not results:
        return {}
    keys: set[str] = set()
    for r in results:
        keys.update(r.keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in results if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
