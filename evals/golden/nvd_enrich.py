"""Offline-friendly CVE enrichment (Zenodo descriptions + NVD cache/API).

Used by Live-agent ATT&CK prompting so the model sees a description without
looking up the target CVE's CTID gold mapping.
"""
from __future__ import annotations

import csv
import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

RAW_DIR = Path(__file__).with_name("raw")
NVD_CACHE_PATH = RAW_DIR / "nvd_cache.json"
ZENODO_CSV_PATH = RAW_DIR / "zenodo_dataset.csv"

_CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Serialize nvd_cache.json RMW across parallel Live/eval workers.
_CACHE_LOCK = threading.Lock()


def _norm_cve(cve_id: str) -> str:
    return (cve_id or "").strip().upper()


def _ensure_cache_file(path: Path | None = None) -> Path:
    p = path if path is not None else NVD_CACHE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        p.write_text("{}" + "\n", encoding="utf-8")
    return p


def _load_cache(path: Path | None = None) -> dict[str, Any]:
    p = _ensure_cache_file(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, Any], path: Path | None = None) -> None:
    p = _ensure_cache_file(path)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@lru_cache(maxsize=1)
def _zenodo_by_cve() -> dict[str, dict[str, Any]]:
    """Map CVE id to enrichment dict from the bundled Zenodo triage CSV."""
    out: dict[str, dict[str, Any]] = {}
    if not ZENODO_CSV_PATH.is_file():
        return out
    with ZENODO_CSV_PATH.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = _norm_cve(row.get("cve_id") or "")
            if not cid.startswith("CVE-"):
                continue
            desc = (row.get("description") or "").strip()
            cvss_raw = (row.get("cvss_score") or "").strip()
            try:
                cvss = float(cvss_raw) if cvss_raw else None
            except ValueError:
                cvss = None
            out[cid] = {
                "cve_id": cid,
                "description": desc,
                "cwes": [],
                "cvss": cvss,
                "source": "zenodo",
            }
    return out


def clear_zenodo_cache() -> None:
    """Drop cached Zenodo index (tests)."""
    _zenodo_by_cve.cache_clear()


def _parse_nvd_payload(cve_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Extract description / CWEs / CVSS from an NVD 2.0 CVE response."""
    vulns = payload.get("vulnerabilities") or []
    description = ""
    cwes: list[str] = []
    cvss: float | None = None
    if vulns:
        cve = (vulns[0] or {}).get("cve") or {}
        for d in cve.get("descriptions") or []:
            if (d.get("lang") or "").lower() == "en" and d.get("value"):
                description = str(d["value"]).strip()
                break
        if not description:
            for d in cve.get("descriptions") or []:
                if d.get("value"):
                    description = str(d["value"]).strip()
                    break
        for w_ in cve.get("weaknesses") or []:
            for desc in w_.get("description") or []:
                val = (desc.get("value") or "").strip().upper()
                if val.startswith("CWE-") and val not in cwes:
                    cwes.append(val)
        metrics = cve.get("metrics") or {}
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            arr = metrics.get(key) or []
            if not arr:
                continue
            data = (arr[0] or {}).get("cvssData") or {}
            score = data.get("baseScore")
            if score is not None:
                try:
                    cvss = float(score)
                except (TypeError, ValueError):
                    cvss = None
                break
    return {
        "cve_id": _norm_cve(cve_id),
        "description": description,
        "cwes": cwes,
        "cvss": cvss,
        "source": "nvd",
    }


def _fetch_nvd(cve_id: str, *, timeout: float = 8.0) -> dict[str, Any] | None:
    """GET NVD API 2.0 for one CVE. Returns None on any failure."""
    cid = _norm_cve(cve_id)
    if not _CVE_RE.match(cid):
        return None
    try:
        import httpx
    except ImportError:
        return None
    url = f"{NVD_API}?cveId={cid}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers={"User-Agent": "sec-tool-agent-nvd-enrich"})
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not (payload.get("vulnerabilities") or []):
        return None
    return _parse_nvd_payload(cid, payload)


def get_cve_enrichment(cve_id: str, *, fetch_nvd: bool = True) -> dict[str, Any]:
    """Return description/cwes/cvss for a CVE (Zenodo, cache, then optional NVD API)."""
    cid = _norm_cve(cve_id)
    empty: dict[str, Any] = {
        "cve_id": cid,
        "description": "",
        "cwes": [],
        "cvss": None,
        "source": "none",
    }
    if not cid.startswith("CVE-"):
        return empty

    zenodo = _zenodo_by_cve().get(cid)
    if zenodo and (zenodo.get("description") or "").strip():
        cwes = list(zenodo.get("cwes") or [])
        if not cwes:
            with _CACHE_LOCK:
                cached = _load_cache().get(cid)
            if isinstance(cached, dict) and cached.get("cwes"):
                cwes = list(cached.get("cwes") or [])
        return {
            "cve_id": cid,
            "description": zenodo.get("description") or "",
            "cwes": cwes,
            "cvss": zenodo.get("cvss"),
            "source": "zenodo",
        }

    with _CACHE_LOCK:
        cache = _load_cache()
        cached = cache.get(cid)
        if isinstance(cached, dict) and (
            (cached.get("description") or "").strip()
            or cached.get("cwes")
            or cached.get("cvss") is not None
        ):
            return {
                "cve_id": cid,
                "description": cached.get("description") or "",
                "cwes": list(cached.get("cwes") or []),
                "cvss": cached.get("cvss"),
                "source": cached.get("source") or "cache",
            }

    fetched = _fetch_nvd(cid) if fetch_nvd else None
    if fetched and (fetched.get("description") or "").strip():
        with _CACHE_LOCK:
            # Reload under lock so parallel Live cold-misses don't clobber peers.
            cache = _load_cache()
            cache[cid] = {
                "description": fetched.get("description") or "",
                "cwes": list(fetched.get("cwes") or []),
                "cvss": fetched.get("cvss"),
                "source": "nvd",
            }
            try:
                _save_cache(cache)
            except OSError:
                pass
        return fetched

    with _CACHE_LOCK:
        cache = _load_cache()
        cached = cache.get(cid)
    if isinstance(cached, dict):
        return {
            "cve_id": cid,
            "description": cached.get("description") or "",
            "cwes": list(cached.get("cwes") or []),
            "cvss": cached.get("cvss"),
            "source": cached.get("source") or "cache",
        }
    if zenodo:
        return {
            "cve_id": cid,
            "description": zenodo.get("description") or "",
            "cwes": list(zenodo.get("cwes") or []),
            "cvss": zenodo.get("cvss"),
            "source": "zenodo",
        }
    return empty

