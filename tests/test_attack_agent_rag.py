"""Offline tests for the Live-agent ATT&CK RAG path (NVD + technique docs)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from evals.golden.attack_docs import (
    format_technique_context,
    load_techniques,
    retrieve_techniques,
)
from evals.golden.live_attack import (
    normalize_attack_techniques,
    postprocess_attack_prediction,
)
from evals.golden.nvd_enrich import get_cve_enrichment
from evals.golden.runner import run_eval
from evals.golden.score import load_cases


def test_retrieve_techniques_rce_public_facing() -> None:
    hits = retrieve_techniques("remote code execution public facing", k=12)
    assert hits, "expected technique hits for RCE / public-facing query"
    ids = {h["id"] for h in hits}
    # T1190 (Exploit Public-Facing Application) should rank among results
    assert "T1190" in ids or any(h["id"].startswith("T1190") for h in hits)
    assert all("id" in h and "name" in h for h in hits)
    ctx = format_technique_context(hits)
    assert "Prefer from this candidate set" in ctx
    assert "T" in ctx


def test_load_techniques_nonempty() -> None:
    techs = load_techniques()
    assert len(techs) >= 800
    ids = {t["id"] for t in techs}
    assert "T1189" in ids
    assert "T1485" in ids


def test_enrichment_offline_from_zenodo() -> None:
    # Zenodo CSV is bundled; pick a known row
    enrich = get_cve_enrichment("CVE-2018-11138")
    assert enrich["source"] == "zenodo"
    assert "arbitrary" in enrich["description"].lower() or "command" in enrich[
        "description"
    ].lower()
    assert enrich["cvss"] is not None


def test_zenodo_description_merges_nvd_cache_cwes(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.nvd_enrich as nvd

    cache_path = tmp_path / "nvd_cache.json"
    cache_path.write_text(
        '{"CVE-2099-0001": {"description": "NVD desc unused", '
        '"cwes": ["CWE-78"], "cvss": 9.1, "source": "nvd"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(nvd, "NVD_CACHE_PATH", cache_path)
    monkeypatch.setattr(
        nvd,
        "_zenodo_by_cve",
        lambda: {
            "CVE-2099-0001": {
                "cve_id": "CVE-2099-0001",
                "description": "Zenodo RCE description",
                "cwes": [],
                "cvss": 8.0,
                "source": "zenodo",
            }
        },
    )
    monkeypatch.setattr(
        nvd,
        "_fetch_nvd",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no network")),
    )
    enrich = nvd.get_cve_enrichment("CVE-2099-0001")
    assert enrich["source"] == "zenodo"
    assert enrich["description"] == "Zenodo RCE description"
    assert enrich["cwes"] == ["CWE-78"]


def test_enrichment_uses_cache_without_network(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.nvd_enrich as nvd

    cache_path = tmp_path / "nvd_cache.json"
    cache_path.write_text(
        '{"CVE-2099-0001": {"description": "Cached RCE in demo", '
        '"cwes": ["CWE-94"], "cvss": 9.8, "source": "nvd"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(nvd, "NVD_CACHE_PATH", cache_path)
    # Ensure Zenodo miss for this fake CVE
    monkeypatch.setattr(nvd, "_zenodo_by_cve", lambda: {})
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("network should not be used when cache hits")

    monkeypatch.setattr(nvd, "_fetch_nvd", _boom)
    enrich = nvd.get_cve_enrichment("CVE-2099-0001")
    assert enrich["description"] == "Cached RCE in demo"
    assert enrich["cwes"] == ["CWE-94"]
    assert called["n"] == 0


def test_enrichment_mock_httpx_writes_cache(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.nvd_enrich as nvd

    cache_path = tmp_path / "nvd_cache.json"
    cache_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(nvd, "NVD_CACHE_PATH", cache_path)
    monkeypatch.setattr(nvd, "_zenodo_by_cve", lambda: {})

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "vulnerabilities": [
                    {
                        "cve": {
                            "descriptions": [
                                {"lang": "en", "value": "Mocked buffer overflow RCE"}
                            ],
                            "weaknesses": [
                                {"description": [{"value": "CWE-119"}]}
                            ],
                            "metrics": {
                                "cvssMetricV31": [
                                    {"cvssData": {"baseScore": 9.1}}
                                ]
                            },
                        }
                    }
                ]
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp()

    import httpx as httpx_mod

    monkeypatch.setattr(httpx_mod, "Client", _Client)
    enrich = nvd.get_cve_enrichment("CVE-2099-4242")
    assert "buffer overflow" in enrich["description"].lower()
    assert enrich["source"] == "nvd"
    assert enrich["cvss"] == 9.1
    # Cache written
    raw = cache_path.read_text(encoding="utf-8")
    assert "CVE-2099-4242" in raw


def test_nvd_cache_lock_exists_and_concurrent_writes(tmp_path: Path, monkeypatch) -> None:
    """Parallel Live cold-misses must not corrupt or lose nvd_cache.json entries."""
    import json
    import time

    import evals.golden.nvd_enrich as nvd

    assert hasattr(nvd._CACHE_LOCK, "acquire") and hasattr(nvd._CACHE_LOCK, "release")

    cache_path = tmp_path / "nvd_cache.json"
    cache_path.write_text("{}" + chr(10), encoding="utf-8")
    monkeypatch.setattr(nvd, "NVD_CACHE_PATH", cache_path)
    monkeypatch.setattr(nvd, "_zenodo_by_cve", lambda: {})

    def _fake_fetch(cve_id: str, *, timeout: float = 8.0):
        # Overlap network latency so RMW would race without the lock.
        time.sleep(0.02)
        cid = nvd._norm_cve(cve_id)
        return {
            "cve_id": cid,
            "description": f"desc for {cid}",
            "cwes": ["CWE-79"],
            "cvss": 7.5,
            "source": "nvd",
        }

    monkeypatch.setattr(nvd, "_fetch_nvd", _fake_fetch)

    cves = [f"CVE-2099-{i:04d}" for i in range(1, 13)]

    def _one(cid: str):
        return nvd.get_cve_enrichment(cid)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, cves))

    assert len(results) == len(cves)
    assert all(r["source"] == "nvd" for r in results)

    raw = cache_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert isinstance(data, dict)
    for cid in cves:
        assert cid in data, f"lost cache entry {cid}"
        assert data[cid]["description"] == f"desc for {cid}"


def test_normalize_attack_techniques_filters_junk() -> None:
    raw = ["T1190", "not-a-tech", "T1059.001", "CVE-2021-44228", "t1068", "XSS"]
    out = normalize_attack_techniques(raw, prefer_ids=["T1190", "T1068"])
    assert out[0] == "T1190"
    assert "T1068" in out
    assert "T1059.001" in out
    assert "not-a-tech" not in out
    assert "CVE-2021-44228" not in out
    assert "XSS" not in out


def test_agent_predict_prompt_has_description_or_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    captured: list[list[dict[str, str]]] = []
    desc = (
        "Remote code execution in a public-facing application "
        "via crafted requests allowing data access and denial of service."
    )

    def _fake_chat(messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any) -> Any:
        captured.append(messages)
        mock = MagicMock()
        mock.choices = [MagicMock()]
        # Include junk to exercise postprocess; evidence must be grounded + ≥8 chars
        mock.choices[0].message.content = (
            '{"exploitation_techniques": '
            '[{"id": "T1190", "evidence": "public-facing application"}, "BOGUS"], '
            '"primary_impact": ["T1005", "T1499.004"], '
            '"rationale": "public facing RCE"}'
        )
        return mock

    monkeypatch.setattr("src.llm.client.chat_completion", _fake_chat)

    attack_cases = [c for c in load_cases() if c["task"] == "cve_to_attack"]
    cases = attack_cases[:1]
    assert cases
    cid = str((cases[0].get("input") or {}).get("cve_id") or "")

    import evals.golden.nvd_enrich as nvd

    monkeypatch.setattr(
        nvd,
        "get_cve_enrichment",
        lambda _c: {
            "cve_id": cid,
            "description": desc,
            "cwes": ["CWE-94"],
            "cvss": 9.8,
            "source": "test",
        },
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_attack_llm_context",
        lambda *_a, **_k: (
            "Prefer from this candidate set: T1190, T1005, T1499.004\n"
            "T1190 Exploit Public-Facing Application",
            [{"id": "T1190"}, {"id": "T1005"}, {"id": "T1499.004"}],
            {
                "cve_id": cid,
                "description": desc,
                "cwes": ["CWE-94"],
                "cvss": 9.8,
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
        lambda *_a, **_k: [
            {
                "cve_id": "CVE-2099-1111",
                "techniques": {
                    "exploitation": ["T1190"],
                    "primary": ["T1005"],
                },
                "attack_techniques": ["T1190", "T1005"],
            }
        ],
    )
    monkeypatch.setenv("LIVE_PREP_CACHE", "0")

    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        cases=cases,
        use_rag=True,
        refine_exploitation=False,
        use_prep_cache=False,
    )
    assert captured, "chat_completion was not called"
    blob = "\n".join(m.get("content") or "" for m in captured[0])
    assert (
        "description" in blob.lower()
        or "Prefer from this candidate set" in blob
        or "candidate set" in blob.lower()
        or "ATT&CK technique" in blob
    ), blob[:500]
    # Must not inject exact CTID gold lookup framing as the answer
    assert "kb_lookup" not in blob
    pred = payload["results"][0]["prediction"]
    techs = pred.get("attack_techniques") or []
    assert "BOGUS" not in techs
    assert all(
        isinstance(t, str) and t.startswith("T") for t in techs
    )
    assert pred.get("exploitation_techniques") == ["T1190"]
    assert set(pred.get("primary_impact") or []) == {"T1005", "T1499.004"}
    # Union built when attack_techniques omitted
    assert set(techs) == {"T1190", "T1005", "T1499.004"}
    assert "rationale" not in pred  # stripped from scored prediction
    # Prompt should teach CTID two-head split + evidence gate
    assert "exploitation" in blob.lower()
    assert "primary_impact" in blob.lower() or "primary impact" in blob.lower()
    assert "cite" in blob.lower() or "evidence" in blob.lower()
    assert "T1190" in blob and "T1059" in blob  # high-prior FP list in gate
    assert len(captured) >= 1


def test_agent_no_rag_still_has_candidates(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    captured: list[list[dict[str, str]]] = []

    def _fake_chat(messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any) -> Any:
        captured.append(messages)
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = '{"attack_techniques": ["T1190"]}'
        return mock

    monkeypatch.setattr("src.llm.client.chat_completion", _fake_chat)

    cases = [c for c in load_cases() if c["task"] == "cve_to_attack"][:1]
    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        cases=cases,
        use_rag=False,
    )
    assert payload.get("eval_rag") is False
    blob = "\n".join(m.get("content") or "" for m in captured[0])
    assert "Few-shot exemplars" not in blob
    # Still should mention candidates or technique docs from enrichment path
    assert "candidate set" in blob.lower() or "T1190" in blob or "technique" in blob.lower()


def test_postprocess_attack_prediction_prefer_ids() -> None:
    out = postprocess_attack_prediction(
        {
            "exploitation_techniques": [
                {"id": "T1068", "evidence": "privilege escalation"},
                {"id": "T1190", "evidence": "remote"},
            ],
            "primary_impact": ["T1005"],
        },
        prefer_ids=["T1190", "T1005"],
    )
    assert out["exploitation_techniques"][0] == "T1190"
    assert "T1005" in out["attack_techniques"]


def test_postprocess_fp_drop_without_evidence_agent_path() -> None:
    """High-prior FPs without evidence are stripped; non-FP impact kept."""
    out = postprocess_attack_prediction(
        {
            "exploitation_techniques": ["T1190", "T1059"],
            "primary_impact": ["T1005", "T1499.004"],
        }
    )
    assert out["exploitation_techniques"] == []
    assert out["primary_impact"] == ["T1005", "T1499.004"]
    assert out["attack_techniques"] == ["T1005", "T1499.004"]

