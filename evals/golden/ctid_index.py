"""Offline CTID CVE→ATT&CK index for lookup, retrieval, and few-shot exemplars.

Source: MITRE Center for Threat-Informed Defense ``attack_to_cve`` mappings
(``evals/golden/raw/AttckToCveMappings.csv``).
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

CSV_PATH = Path(__file__).with_name("raw") / "AttckToCveMappings.csv"

_TECH_RE = re.compile(r"T\d{4}(?:\.\d{3})?", re.IGNORECASE)
_CVE_RE = re.compile(r"CVE-(\d{4})-(\d+)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _split_techniques(cell: str | None) -> list[str]:
    if not cell or not str(cell).strip():
        return []
    found: list[str] = []
    for part in re.split(r"[;,\s]+", str(cell).strip()):
        part = part.strip()
        if not part:
            continue
        m = _TECH_RE.fullmatch(part) or _TECH_RE.search(part)
        if not m:
            continue
        tid = m.group(0).upper()
        if tid not in found:
            found.append(tid)
    return found


def _stable_seed(text: str) -> int:
    """Process-stable seed for a string (``hash()`` is salted per interpreter)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _tokens(*parts: str) -> set[str]:
    out: set[str] = set()
    for p in parts:
        if not p:
            continue
        for t in _TOKEN_RE.findall(p.lower()):
            if len(t) >= 2:
                out.add(t)
    return out


def _row_to_record(row: dict[str, str]) -> dict[str, Any] | None:
    cve_id = (row.get("CVE ID") or row.get("cve_id") or "").strip().upper()
    if not cve_id.startswith("CVE-"):
        return None
    primary = _split_techniques(row.get("Primary Impact"))
    secondary = _split_techniques(row.get("Secondary Impact"))
    exploitation = _split_techniques(row.get("Exploitation Technique"))
    uncategorized = _split_techniques(row.get("Uncategorized"))
    all_techs: list[str] = []
    for group in (primary, secondary, exploitation, uncategorized):
        for t in group:
            if t not in all_techs:
                all_techs.append(t)
    if not all_techs:
        return None
    phase = (row.get("Phase") or "").strip()
    text_blob = " ".join(
        filter(
            None,
            [
                cve_id,
                " ".join(primary),
                " ".join(secondary),
                " ".join(exploitation),
                " ".join(uncategorized),
                phase,
                "primary impact",
                "exploitation technique",
            ],
        )
    )
    year = None
    m = _CVE_RE.match(cve_id)
    if m:
        year = m.group(1)
    return {
        "cve_id": cve_id,
        "techniques": {
            "primary": primary,
            "secondary": secondary,
            "exploitation": exploitation,
            "uncategorized": uncategorized,
            "all": all_techs,
        },
        "attack_techniques": all_techs,
        "phase": phase,
        "text": text_blob,
        "year": year,
        "tokens": _tokens(cve_id, text_blob, *all_techs),
        "source": "MITRE CTID attack_to_cve",
    }


def load_ctid_records(csv_path: Path | None = None) -> list[dict[str, Any]]:
    """Load and parse AttckToCveMappings.csv into searchable records."""
    path = csv_path or CSV_PATH
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rec = _row_to_record(row)
            if rec:
                records.append(rec)
    return records


@lru_cache(maxsize=1)
def get_index() -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    """Return ``(by_cve_id, records_tuple)`` cached in memory."""
    records = load_ctid_records()
    by_id = {r["cve_id"]: r for r in records}
    return by_id, tuple(records)


def clear_index_cache() -> None:
    """Drop the in-memory CTID index cache (tests / CSV reload)."""
    get_index.cache_clear()


def lookup_ctid_mapping(cve_id: str) -> dict[str, Any] | None:
    """Exact CVE→ATT&CK mapping from the local CTID knowledge base."""
    if not cve_id:
        return None
    key = str(cve_id).strip().upper()
    by_id, _ = get_index()
    rec = by_id.get(key)
    if not rec:
        return None
    return {
        "cve_id": rec["cve_id"],
        "attack_techniques": list(rec["attack_techniques"]),
        "techniques": {k: list(v) for k, v in rec["techniques"].items()},
        "phase": rec.get("phase"),
        "source": rec.get("source"),
        "match": "exact",
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _query_tokens(
    cve_id: str | None,
    query_text: str | None,
) -> set[str]:
    parts: list[str] = []
    if cve_id:
        parts.append(cve_id)
        m = _CVE_RE.match(cve_id.strip().upper())
        if m:
            parts.append(m.group(1))
            parts.append(m.group(2))
    if query_text:
        parts.append(query_text)
        for t in _TECH_RE.findall(query_text):
            parts.append(t.upper())
    return _tokens(*parts)


def retrieve(
    cve_id: str | None = None,
    query_text: str | None = None,
    k: int = 5,
    *,
    exclude_cve_ids: set[str] | frozenset[str] | None = None,
    allow_exact: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve similar CTID mappings.

    - If ``allow_exact`` and ``cve_id`` is in the index (and not excluded),
      that mapping is returned first.
    - Otherwise lexical similarity on CVE year / id tokens and optional
      description / technique strings. Empty list if nothing useful matches.
    """
    by_id, records = get_index()
    excluded = {x.strip().upper() for x in (exclude_cve_ids or set()) if x}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    norm_cve = (cve_id or "").strip().upper() or None

    if allow_exact and norm_cve and norm_cve not in excluded and norm_cve in by_id:
        hit = lookup_ctid_mapping(norm_cve)
        if hit:
            results.append(hit)
            seen.add(norm_cve)

    q_tokens = _query_tokens(norm_cve, query_text)
    if not q_tokens and not results:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        cid = rec["cve_id"]
        if cid in seen or cid in excluded:
            continue
        score = _jaccard(q_tokens, rec["tokens"])
        # Soft year boost when querying by CVE id
        if norm_cve and rec.get("year") and norm_cve[4:8] == rec["year"]:
            score += 0.05
        if score <= 0.0:
            continue
        scored.append((score, rec))

    scored.sort(key=lambda x: (-x[0], x[1]["cve_id"]))
    for score, rec in scored:
        if len(results) >= k:
            break
        results.append(
            {
                "cve_id": rec["cve_id"],
                "attack_techniques": list(rec["attack_techniques"]),
                "techniques": {kk: list(vv) for kk, vv in rec["techniques"].items()},
                "phase": rec.get("phase"),
                "source": rec.get("source"),
                "match": "lexical",
                "score": round(score, 4),
            }
        )
        seen.add(rec["cve_id"])

    return results[:k]


def build_few_shot_exemplars(
    target_cve_id: str,
    n: int = 3,
    *,
    seed: int | None = None,
    prefer_same_year: bool = True,
) -> list[dict[str, Any]]:
    """Return ``n`` CTID exemplars excluding ``target_cve_id``.

    Prefers same CVE year when possible, then fills with seeded random others.
    With no explicit ``seed``, the target id is hashed into one: ``random.seed``
    on a str would vary per process (PYTHONHASHSEED), so use a stable digest.
    """
    target = (target_cve_id or "").strip().upper()
    _, records = get_index()
    pool = [r for r in records if r["cve_id"] != target]
    if not pool or n <= 0:
        return []

    year = None
    m = _CVE_RE.match(target)
    if m:
        year = m.group(1)

    rng = random.Random(seed if seed is not None else _stable_seed(target))
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    if prefer_same_year and year:
        same = [r for r in pool if r.get("year") == year]
        rng.shuffle(same)
        for r in same:
            if len(chosen) >= n:
                break
            chosen.append(r)
            chosen_ids.add(r["cve_id"])

    rest = [r for r in pool if r["cve_id"] not in chosen_ids]
    rng.shuffle(rest)
    for r in rest:
        if len(chosen) >= n:
            break
        chosen.append(r)
        chosen_ids.add(r["cve_id"])

    out: list[dict[str, Any]] = []
    for r in chosen[:n]:
        out.append(
            {
                "cve_id": r["cve_id"],
                "attack_techniques": list(r["attack_techniques"]),
                "techniques": {k: list(v) for k, v in r["techniques"].items()},
            }
        )
    # Hard guarantee: never include target
    return [e for e in out if e["cve_id"] != target]


def technique_prior(top_n: int = 10) -> list[str]:
    """Most common exploitation / all techniques in CTID (prompt hint only)."""
    _, records = get_index()
    counter: Counter[str] = Counter()
    for r in records:
        # Prefer exploitation techniques when present
        techs = r["techniques"].get("exploitation") or r["attack_techniques"]
        counter.update(techs)
    return [t for t, _ in counter.most_common(top_n)]
