"""CIRCL Hugging Face gold import → our golden CVE→ATT&CK schema.

Dataset: ``CIRCL/vulnerability-attack-techniques`` (CTID-curated; arXiv:2607.25572).

Rules (CIRCL gold import):
- Pin an HF revision / commit (``HF_REVISION``).
- Score on flat ``techniques`` → ``expected.attack_techniques``.
- Populate ``exploitation_techniques`` / ``primary_impact`` only when CIRCL
  provides those fields (pass through non-empty lists; empty → omit head gold
  via empty list so scorers skip). Never invent heads from CAPEC /
  ``techniques_derived`` / LLM expansion.
- Never use ``techniques_derived`` as gold.
- Freeze official test IDs under protocol ``circl_test``; the neighbor ICL pool
  is train only (hard-exclude test CVE ids).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

HF_DATASET = "CIRCL/vulnerability-attack-techniques"
HF_REVISION = "319c3e324e7561592cf7e51ec003deb5dd90e61b"
HF_URL = f"https://huggingface.co/datasets/{HF_DATASET}"
PAPER = "arXiv:2607.25572"
DOI = "10.57967/hf/9621"

RAW_DIR = Path(__file__).with_name("raw") / "circl"
HELDOUT_CIRCL_PATH = Path(__file__).with_name("heldout_circl_test_ids.json")
PROTOCOL_CIRCL_TEST = "circl_test"
PROTOCOL_DEEPSEEK_N20 = "deepseek_live_n20"

_TECH_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
_CASE_PREFIX = "circl-"

SplitName = Literal["train", "test"]


def case_id_for_cve(cve_id: str) -> str:
    """Stable golden case id: ``circl-CVE-YYYY-NNNNN``."""
    return f"{_CASE_PREFIX}{_norm_cve(cve_id)}"


def cve_from_case_id(case_id: str) -> str:
    """Extract CVE id from a ``circl-…`` case id (or bare CVE)."""
    s = (case_id or "").strip()
    if s.lower().startswith(_CASE_PREFIX):
        s = s[len(_CASE_PREFIX) :]
    return _norm_cve(s)


def _norm_cve(raw: str) -> str:
    return (raw or "").strip().upper()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, float):  # NaN
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in list(value or []):
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def normalize_technique_ids(raw: Iterable[Any] | None) -> list[str]:
    """Keep valid ``T####`` / ``T####.###`` ids, uppercased, de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        s = str(item).strip().upper().split()[0] if str(item).strip() else ""
        if not s:
            continue
        if not _TECH_RE.fullmatch(s):
            m = re.search(r"T\d{4}(?:\.\d{3})?", s, re.IGNORECASE)
            if not m:
                continue
            s = m.group(0).upper()
        if not _TECH_RE.fullmatch(s) or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def map_circl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map one CIRCL HF row into our golden ``expected`` + metadata.

    - ``techniques`` → ``attack_techniques`` (scoring target).
    - ``exploitation_techniques`` / ``primary_impact`` passed through when CIRCL
      provides them (may be empty → scorers treat head as N/A).
    - ``techniques_derived`` is never copied into gold.
    - No invented heads: we do not derive exploitation/primary from the flat
      union or from CAPEC weak labels.
    """
    cve_id = _norm_cve(str(row.get("id") or row.get("cve_id") or ""))
    techniques = normalize_technique_ids(_as_str_list(row.get("techniques")))
    # Pass-through CIRCL heads only; do not invent from techniques / derived
    exploitation = normalize_technique_ids(
        _as_str_list(row.get("exploitation_techniques"))
    )
    primary_impact = normalize_technique_ids(_as_str_list(row.get("primary_impact")))
    # secondary_impact contributes to flat techniques upstream; we do not add a
    # secondary head to our schema (CTID golden also lacks one).
    expected: dict[str, Any] = {
        "attack_techniques": techniques,
        "primary_impact": primary_impact,
        "exploitation_techniques": exploitation,
    }
    return {
        "cve_id": cve_id,
        "expected": expected,
        "title": (row.get("title") or "") if isinstance(row.get("title"), str) else "",
        "description": (
            (row.get("description") or "")
            if isinstance(row.get("description"), str)
            else ""
        ),
        "label_sources": _as_str_list(row.get("label_sources")),
        "attack_version": str(row.get("attack_version") or ""),
        "cwes": _as_str_list(row.get("cwes")),
        # Explicitly discarded; documented for callers / tests
        "techniques_derived_ignored": True,
    }


def row_to_golden_case(
    row: dict[str, Any],
    *,
    split: SplitName,
) -> dict[str, Any]:
    """Build a golden_dataset-compatible case dict from a CIRCL row."""
    mapped = map_circl_row(row)
    cve_id = mapped["cve_id"]
    if not cve_id.startswith("CVE-"):
        raise ValueError(f"CIRCL row missing CVE id: {row.get('id')!r}")
    if not mapped["expected"]["attack_techniques"]:
        raise ValueError(f"CIRCL row {cve_id} has empty techniques (gold required)")
    inp: dict[str, Any] = {"cve_id": cve_id}
    desc = (mapped.get("description") or "").strip()
    if desc:
        inp["description"] = desc
    cwes = [str(x) for x in (mapped.get("cwes") or []) if x]
    if cwes:
        inp["cwes"] = cwes
    metrics = ["recall_at_k_techniques", "hit"]
    if mapped["expected"]["primary_impact"]:
        metrics.append("primary_hit")
    if mapped["expected"]["exploitation_techniques"]:
        metrics.append("exploitation_hit")
    return {
        "id": case_id_for_cve(cve_id),
        "task": "cve_to_attack",
        "input": inp,
        "expected": mapped["expected"],
        "metrics": metrics,
        "source": f"CIRCL/{HF_DATASET}@{HF_REVISION[:12]} ({split})",
        "license_note": (
            "CC-BY-4.0; cite CIRCL / VulnTrain and MITRE CTID. "
            f"Paper {PAPER}; DOI {DOI}. Do not train on techniques_derived."
        ),
        "split": split,
        "protocol": PROTOCOL_CIRCL_TEST if split == "test" else "circl_train",
        "label_sources": mapped.get("label_sources") or [],
        "attack_version": mapped.get("attack_version") or "",
    }


def _parquet_path(split: SplitName) -> Path:
    # HF default shard naming for this dataset
    return RAW_DIR / f"{split}-00000-of-00001.parquet"


def _rows_from_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        # Convert numpy arrays to plain lists for JSON-friendliness
        cleaned: dict[str, Any] = {}
        for k, v in rec.items():
            if hasattr(v, "tolist"):
                try:
                    v = v.tolist()
                except Exception:  # noqa: BLE001
                    pass
            cleaned[k] = v
        rows.append(cleaned)
    return rows


def load_circl_rows(
    split: SplitName | Literal["all"] = "all",
    *,
    revision: str | None = None,
    prefer_local: bool = True,
    rows_override: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load CIRCL train/test rows pinned to ``revision``.

    Prefers vendored parquet under ``evals/golden/raw/circl/`` (offline).
    Optional ``rows_override`` is for tests (mock HF load).
    When local files are missing, attempts ``datasets.load_dataset`` with the
    pinned revision (optional dependency).
    """
    rev = revision or HF_REVISION
    if rows_override is not None:
        # Treat override as a single split named by ``split`` (or test)
        key = "test" if split == "all" else split
        return {key: list(rows_override)}

    meta_path = RAW_DIR / "REVISION.json"
    if prefer_local and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pinned = meta.get("hf_revision")
        if pinned and pinned != rev:
            raise ValueError(
                f"Local CIRCL cache revision {pinned} != requested {rev}. "
                "Re-download or pass matching revision."
            )

    out: dict[str, list[dict[str, Any]]] = {}
    splits: list[SplitName] = (
        ["train", "test"] if split == "all" else [split]  # type: ignore[list-item]
    )
    missing = [s for s in splits if not _parquet_path(s).is_file()]
    if prefer_local and not missing:
        for s in splits:
            out[s] = _rows_from_parquet(_parquet_path(s))
        return out

    # Network / datasets fallback
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - env without datasets
        if missing:
            raise FileNotFoundError(
                f"CIRCL parquet missing for {missing} under {RAW_DIR} and "
                f"`datasets` is not installed. Vendor files or pip install datasets."
            ) from exc
        for s in splits:
            out[s] = _rows_from_parquet(_parquet_path(s))
        return out

    ds = load_dataset(HF_DATASET, revision=rev)
    for s in splits:
        out[s] = [dict(r) for r in ds[s]]
    return out


def load_circl_cases(
    split: SplitName | Literal["all"] = "test",
    *,
    revision: str | None = None,
    rows_override: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load golden-schema cases for CIRCL ``train`` / ``test`` / ``all``."""
    loaded = load_circl_rows(
        "all" if split == "all" else split,
        revision=revision,
        rows_override=rows_override,
    )
    cases: list[dict[str, Any]] = []
    order: list[SplitName] = (
        ["train", "test"] if split == "all" else [split]  # type: ignore[list-item]
    )
    for s in order:
        for row in loaded.get(s, []):
            cases.append(row_to_golden_case(row, split=s))
    return cases


@lru_cache(maxsize=1)
def circl_test_cve_ids() -> frozenset[str]:
    """Frozen CIRCL test CVE ids (hard-exclude set for neighbor ICL)."""
    data = json.loads(HELDOUT_CIRCL_PATH.read_text(encoding="utf-8"))
    ids = data.get("test_cve_ids")
    if isinstance(ids, list) and ids:
        return frozenset(_norm_cve(x) for x in ids)
    # Fallback: derive from case_ids
    case_ids = data.get("case_ids") or []
    return frozenset(cve_from_case_id(str(x)) for x in case_ids)


@lru_cache(maxsize=1)
def circl_train_neighbor_records() -> tuple[dict[str, Any], ...]:
    """Train-split labeled records for neighbor ICL (CIRCL protocol)."""
    cases = load_circl_cases(split="train")
    out: list[dict[str, Any]] = []
    for case in cases:
        exp = case.get("expected") or {}
        cve_id = _norm_cve(str((case.get("input") or {}).get("cve_id") or ""))
        primary = list(exp.get("primary_impact") or [])
        exploitation = list(exp.get("exploitation_techniques") or [])
        all_techs = list(exp.get("attack_techniques") or [])
        out.append(
            {
                "cve_id": cve_id,
                "attack_techniques": all_techs,
                "techniques": {
                    "primary": primary,
                    "secondary": [],
                    "exploitation": exploitation,
                    "uncategorized": [],
                    "all": all_techs,
                },
                "phase": "",
                "source": "circl_train",
                "description": str((case.get("input") or {}).get("description") or ""),
            }
        )
    return tuple(out)


def clear_circl_caches() -> None:
    """Drop cached CIRCL test ids / train neighbor records (tests).

    Also drops the neighbor corpus derived from those records, which would
    otherwise survive as a stale view of the cleared inputs.
    """
    circl_test_cve_ids.cache_clear()
    circl_train_neighbor_records.cache_clear()
    from evals.golden.eval_rag import _cached_circl_train_neighbor_corpus

    _cached_circl_train_neighbor_corpus.cache_clear()


def load_circl_heldout() -> dict[str, Any]:
    """Load the frozen ``circl_test`` held-out metadata + case_ids."""
    data = json.loads(HELDOUT_CIRCL_PATH.read_text(encoding="utf-8"))
    if data.get("protocol") != PROTOCOL_CIRCL_TEST:
        raise ValueError(
            f"Expected protocol {PROTOCOL_CIRCL_TEST!r}, got {data.get('protocol')!r}"
        )
    ids = data.get("case_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"heldout CIRCL file missing case_ids: {HELDOUT_CIRCL_PATH}")
    data["case_ids"] = [str(x) for x in ids]
    return data
