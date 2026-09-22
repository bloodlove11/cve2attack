"""Tests for Live prep disk cache and refine/jobs wiring."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.golden.live_cache import (
    cache_get,
    cache_key,
    cache_set,
    clear_prep_cache,
    prep_cache_enabled,
)
from evals.golden.live_attack import predict_cve_to_attack
from evals.golden.runner import DEFAULT_JOBS_CAP, resolve_jobs


def test_default_jobs_cap_is_eight() -> None:
    assert DEFAULT_JOBS_CAP == 8
    assert resolve_jobs(121, None) == 8
    assert resolve_jobs(121, 1) == 1
    assert resolve_jobs(5, None) == 5


def test_prep_cache_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.live_cache as lc

    monkeypatch.setattr(lc, "CACHE_DIR", tmp_path / "live_prep")
    key = cache_key({"cve_id": "CVE-2099-1", "v": 1})
    assert cache_get(key) is None
    cache_set(key, {"rag_ctx": "hello", "hits": [{"id": "T1190"}]})
    hit = cache_get(key)
    assert hit is not None
    assert hit["rag_ctx"] == "hello"
    assert hit["hits"][0]["id"] == "T1190"
    assert clear_prep_cache() >= 1


def test_prep_cache_env_disable(monkeypatch) -> None:
    monkeypatch.setenv("LIVE_PREP_CACHE", "0")
    assert prep_cache_enabled() is False
    assert prep_cache_enabled(True) is True
    monkeypatch.delenv("LIVE_PREP_CACHE", raising=False)
    assert prep_cache_enabled() is True


def test_predict_skips_refine_when_disabled(monkeypatch) -> None:
    calls: list[str] = []

    def _chat(messages: list[dict[str, str]]) -> dict[str, Any]:
        blob = "\n".join(m.get("content") or "" for m in messages)
        if "Refine exploitation" in blob or "specialize in CTID exploitation" in blob:
            calls.append("exploit")
            return {"exploitation_techniques": ["T1190"], "_raw": "{}"}
        calls.append("two_head")
        return {
            "exploitation_techniques": ["T1190"],
            "primary_impact": ["T1005"],
            "attack_techniques": ["T1190", "T1005"],
            "evidence": {"T1190": "public-facing application"},
            "_raw": "{}",
        }

    monkeypatch.setattr(
        "evals.golden.live_attack.get_cve_enrichment",
        lambda _c: {
            "description": "Remote code execution in a public-facing application",
            "cwes": ["CWE-94"],
            "source": "test",
        },
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_attack_llm_context",
        lambda *_a, **_k: (
            "candidate set: T1190, T1005\npublic-facing application",
            [{"id": "T1190"}, {"id": "T1005"}],
            {
                "description": "Remote code execution in a public-facing application",
                "cwes": ["CWE-94"],
                "source": "test",
            },
        ),
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.retrieve_labeled_neighbors",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_few_shot_exemplars",
        lambda *_a, **_k: [],
    )

    out = predict_cve_to_attack(
        "CVE-2099-1",
        llm_chat=_chat,
        refine_exploitation=False,
        use_prep_cache=False,
    )
    assert calls == ["two_head"]
    assert out["llm_calls"] == 1
    assert out["refine_exploitation"] is False


def test_predict_prep_cache_hit_skips_retrieval(monkeypatch, tmp_path: Path) -> None:
    import evals.golden.live_cache as lc

    monkeypatch.setattr(lc, "CACHE_DIR", tmp_path / "live_prep")
    builds = {"n": 0}

    def _build(*_a, **_k):
        builds["n"] += 1
        return (
            "public-facing application docs",
            [{"id": "T1190"}, {"id": "T1005"}],
            {
                "description": "Remote code execution in a public-facing application",
                "cwes": ["CWE-94"],
                "source": "test",
            },
        )

    monkeypatch.setattr(
        "evals.golden.live_attack.get_cve_enrichment",
        lambda _c: {
            "description": "Remote code execution in a public-facing application",
            "cwes": ["CWE-94"],
            "source": "test",
        },
    )
    monkeypatch.setattr("evals.golden.live_attack.build_attack_llm_context", _build)
    monkeypatch.setattr(
        "evals.golden.live_attack.retrieve_labeled_neighbors",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_few_shot_exemplars",
        lambda *_a, **_k: [],
    )

    def _chat(messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "exploitation_techniques": [
                {"id": "T1190", "evidence": "public-facing application"}
            ],
            "primary_impact": ["T1005"],
            "attack_techniques": ["T1190", "T1005"],
            "_raw": "{}",
        }

    out1 = predict_cve_to_attack(
        "CVE-2099-2",
        llm_chat=_chat,
        refine_exploitation=False,
        use_prep_cache=True,
    )
    out2 = predict_cve_to_attack(
        "CVE-2099-2",
        llm_chat=_chat,
        refine_exploitation=False,
        use_prep_cache=True,
    )
    assert builds["n"] == 1
    assert out1["prep_cache_hit"] is False
    assert out2["prep_cache_hit"] is True


def test_eval_lab_exposes_speed_controls() -> None:
    text = Path("src/ui/app_pages/eval_lab.py").read_text(encoding="utf-8")
    assert "Parallel jobs" in text
    assert "Exploitation refine LLM" in text
    assert "Cache Live prep" in text
    assert "refine_exploitation=" in text
    assert "use_prep_cache=" in text
    assert "jobs=int(jobs)" in text
