"""Disk cache for non-LLM Live ATT&CK prep (enrich context, tech hits, neighbors).

Speeds CIRCL re-runs and ablations: LLM calls stay live; retrieval/ICL prep
is reused across jobs and sessions. Cache lives under ``evals/golden/cache/``
(gitignored).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).with_name("cache") / "live_prep"
_LOCK = threading.Lock()

# Env kill-switch (also exposed as predict_cve_to_attack use_prep_cache=)
_ENV_DISABLE = "LIVE_PREP_CACHE"


def prep_cache_enabled(explicit: bool | None = None) -> bool:
    """Return whether disk prep cache should be used.

    ``explicit`` wins when not None. Otherwise honor env ``LIVE_PREP_CACHE``
    (``0``/``false``/``off`` disables; default on).
    """
    if explicit is not None:
        return bool(explicit)
    raw = (os.environ.get(_ENV_DISABLE) or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def cache_key(parts: dict[str, Any]) -> str:
    """Stable SHA-256 hex key from a JSON-serializable dict."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path_for(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_get(key: str) -> dict[str, Any] | None:
    """Load a cached prep payload, or ``None`` on miss/corruption."""
    path = _path_for(key)
    if not path.is_file():
        return None
    try:
        with _LOCK:
            data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def cache_set(key: str, payload: dict[str, Any]) -> None:
    """Persist a prep payload (best-effort; never raises to callers)."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _path_for(key)
        text = json.dumps(payload, indent=0, sort_keys=True, default=str)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with _LOCK:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
    except OSError:
        return


def clear_prep_cache() -> int:
    """Delete all prep cache files; return count removed (tests)."""
    if not CACHE_DIR.is_dir():
        return 0
    n = 0
    with _LOCK:
        for p in CACHE_DIR.glob("*.json"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n
