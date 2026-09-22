"""Lightweight JSONL / JSON trace logging under traces/."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = _PROJECT_ROOT / "traces"


def new_trace_id() -> str:
    """Return a short hex id for a single agent run."""
    return uuid.uuid4().hex[:12]


class TraceLogger:
    """Collects steps for a single agent run and persists to traces/."""

    def __init__(self, query: str, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.trace_id = new_trace_id()
        self.query = query
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {}

    def add_step(self, kind: str, **payload: Any) -> None:
        """Append a timestamped step (``user``, ``tool_call``, ``tool_result``, …)."""
        if not self.enabled:
            return
        self.steps.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                **payload,
            }
        )

    def set_meta(self, **kwargs: Any) -> None:
        """Merge run-level metadata (model, history turns, …)."""
        self.meta.update(kwargs)

    def fill(
        self,
        dest: dict[str, Any] | None,
        path: Path | None,
        **extra: Any,
    ) -> None:
        """Copy run metadata into ``dest`` (no-op when ``dest`` is None)."""
        if dest is None:
            return
        dest["path"] = str(path) if path is not None else None
        dest["trace_id"] = self.trace_id
        dest["steps"] = list(self.steps)
        dest["meta"] = dict(self.meta)
        dest.update(extra)

    def finalize(self, final: dict[str, Any] | None = None) -> Path | None:
        """Write ``traces/trace_<id>.json`` and return its path (or None if disabled)."""
        if not self.enabled:
            return None
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "trace_id": self.trace_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "meta": self.meta,
            "steps": self.steps,
            "final": final,
        }
        path = TRACES_DIR / f"trace_{self.trace_id}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path
