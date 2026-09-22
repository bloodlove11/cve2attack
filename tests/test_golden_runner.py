"""Offline tests for golden eval runner + score helpers (no live API)."""
from __future__ import annotations

from pathlib import Path

from evals.golden.live_attack import postprocess_attack_prediction
from evals.golden.runner import run_eval
from evals.golden.score import (
    canonical_match,
    load_cases,
    score_attack,
    score_attack_with_heads,
    score_technique_head,
    score_triage,
    summarize,
)


def test_score_helpers() -> None:
    assert score_triage("ACT", "ACT")["label_exact_match"] == 1.0
    assert score_triage("track", "TRACK")["label_exact_match"] == 1.0
    assert score_triage("ACT", "ATTEND")["label_exact_match"] == 0.0

    att = score_attack(["T1059", "T1190", "T1005"], ["T1005", "T1190"], k=5)
    assert att["hit"] == 1.0
    assert att["recall_at_k"] == 1.0
    assert 0.0 < att["precision_at_k"] <= 1.0

    empty = score_attack([], ["T1059"], k=5)
    assert empty["hit"] == 0.0
    assert empty["recall_at_k"] == 0.0

    summary = summarize(
        [
            {"label_exact_match": 1.0},
            {"label_exact_match": 0.0},
        ]
    )
    assert summary["label_exact_match"] == 0.5



def _echo_agent(monkeypatch) -> None:
    """Offline stand-in: agent returns gold labels (no API)."""
    import evals.golden.runner as runner_mod

    def fake(case, **_kwargs):  # noqa: ANN001
        task = case.get("task")
        exp = case.get("expected") or {}
        if task == "cve_triage":
            return {
                "triage_label": exp.get("triage_label") or "",
                "mode": "agent_llm",
            }
        return {
            "attack_techniques": list(exp.get("attack_techniques") or []),
            "primary_impact": list(exp.get("primary_impact") or []),
            "exploitation_techniques": list(exp.get("exploitation_techniques") or []),
            "mode": "agent_llm",
        }

    monkeypatch.setattr(runner_mod, "_agent_predict", fake)


def test_agent_mocked_triage_limit_five(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")
    _echo_agent(monkeypatch)

    payload = run_eval(
        mode="agent",
        task="cve_triage",
        limit=5,
        seed=42,
        write_results=True,
    )
    assert payload["mode"] == "agent"
    assert payload["metrics"]["n_scored"] == 5
    assert payload["metrics"]["n_skipped"] == 0
    acc = payload["metrics"].get("accuracy")
    assert acc == 1.0
    assert (tmp_path / "latest.json").is_file()
    assert all(r["task"] == "cve_triage" for r in payload["results"])


def test_load_cases_has_both_tasks() -> None:
    cases = load_cases()
    tasks = {c["task"] for c in cases}
    assert "cve_triage" in tasks
    assert "cve_to_attack" in tasks
    assert len(cases) >= 50


def test_agent_mocked_mixed_split_metrics(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")
    _echo_agent(monkeypatch)

    cases = load_cases()
    triage = [c for c in cases if c["task"] == "cve_triage"][:5]
    attack = [c for c in cases if c["task"] == "cve_to_attack"][:5]
    payload = run_eval(
        mode="agent",
        task="all",
        write_results=False,
        cases=triage + attack,
    )
    m = payload["metrics"]
    assert m["triage"]["accuracy"] == 1.0
    assert m["attack"]["hit_rate"] == 1.0
    assert m["macro_score"] == 1.0
    assert "accuracy" not in m


def test_llm_failure_is_skipped_not_scored_as_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """Provider errors must not deflate hit_rate as empty-pred misses."""
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    n = {"i": 0}

    def fake(case, **_kwargs):  # noqa: ANN001
        n["i"] += 1
        exp = case.get("expected") or {}
        if n["i"] == 1:
            return {
                "attack_techniques": [],
                "primary_impact": [],
                "exploitation_techniques": [],
                "mode": "agent_llm_failed",
                "error": "free_limit_reached",
            }
        return {
            "attack_techniques": list(exp.get("attack_techniques") or []),
            "primary_impact": list(exp.get("primary_impact") or []),
            "exploitation_techniques": list(exp.get("exploitation_techniques") or []),
            "mode": "agent_llm",
        }

    monkeypatch.setattr(runner_mod, "_agent_predict", fake)
    cases = [c for c in load_cases() if c["task"] == "cve_to_attack"][:3]
    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        cases=cases,
        write_results=False,
        jobs=1,
    )
    m = payload["metrics"]
    assert m["n_total"] == 3
    assert m["n_failed"] == 1
    assert m["n_skipped"] == 1
    assert m["n_scored"] == 2
    assert m["attack"]["hit_rate"] == 1.0
    failed_row = next(r for r in payload["results"] if r.get("skipped"))
    assert failed_row["prediction"]["mode"] == "agent_llm_failed"
    assert failed_row.get("correct") is None or failed_row.get("scores") == {}


def test_invalid_mode_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="mode"):
        run_eval(mode="nope", task="cve_triage", limit=1, write_results=False)


def test_postprocess_unions_heads_into_attack_techniques() -> None:
    out = postprocess_attack_prediction(
        {
            "exploitation_techniques": [
                {"id": "T1190", "evidence": "remote code execution"},
                "BOGUS",
            ],
            "primary_impact": ["T1005", "T1499.004"],
        }
    )
    assert out["exploitation_techniques"] == ["T1190"]
    assert out["primary_impact"] == ["T1005", "T1499.004"]
    assert out["attack_techniques"] == ["T1190", "T1005", "T1499.004"]

    # Explicit attack_techniques kept when provided (validated) + evidenced
    kept = postprocess_attack_prediction(
        {
            "exploitation_techniques": [
                {"id": "T1190", "evidence": "public-facing application"}
            ],
            "primary_impact": ["T1005"],
            "attack_techniques": [
                {"id": "T1059", "evidence": "script/command execution"},
                {"id": "T1190", "evidence": "remote"},
            ],
        }
    )
    assert kept["attack_techniques"] == ["T1059", "T1190"]


def test_postprocess_drops_high_prior_fps_without_evidence() -> None:
    dropped = postprocess_attack_prediction(
        {
            "exploitation_techniques": ["T1190", "T1203", "T1068"],
            "primary_impact": ["T1005", "T1059", "T1055"],
        }
    )
    assert dropped["exploitation_techniques"] == []
    assert dropped["primary_impact"] == ["T1005"]
    assert dropped["attack_techniques"] == ["T1005"]

    # Evidence map keeps gated IDs; empty/missing still drops
    kept = postprocess_attack_prediction(
        {
            "exploitation_techniques": ["T1190", "T1203"],
            "primary_impact": ["T1005"],
            "evidence": {"T1190": "remote exploit of public app", "T1203": ""},
        }
    )
    assert kept["exploitation_techniques"] == ["T1190"]
    assert "T1203" not in kept["exploitation_techniques"]
    assert kept["attack_techniques"] == ["T1190", "T1005"]

    # evidence_gate=False preserves legacy behavior
    raw = postprocess_attack_prediction(
        {"exploitation_techniques": ["T1190"], "primary_impact": ["T1005"]},
        evidence_gate=False,
    )
    assert raw["exploitation_techniques"] == ["T1190"]


def test_score_primary_impact_and_exploitation_heads() -> None:
    pi = score_technique_head(["T1005", "T1190"], ["T1005", "T1499.004"], k=5)
    assert pi is not None
    assert pi["hit"] == 1.0
    assert pi["recall_at_k"] == 0.5

    # Empty gold → N/A
    assert score_technique_head(["T1190"], [], k=5) is None
    assert score_technique_head(["T1190"], None, k=5) is None

    miss = score_technique_head(["T1190"], ["T1005"], k=5)
    assert miss is not None
    assert miss["hit"] == 0.0

    full = score_attack_with_heads(
        predicted_techniques=["T1190", "T1005"],
        expected_techniques=["T1190", "T1005", "T1499.004"],
        predicted_primary_impact=["T1005"],
        expected_primary_impact=["T1005", "T1499.004"],
        predicted_exploitation=["T1190"],
        expected_exploitation=["T1190"],
        k=5,
    )
    assert full["hit"] == 1.0
    assert full["primary_impact_hit"] == 1.0
    assert full["exploitation_hit"] == 1.0
    assert "exploitation_hit" in full

    # Empty exploitation gold omits exploitation_* keys
    no_ex = score_attack_with_heads(
        predicted_techniques=["T1005"],
        expected_techniques=["T1005"],
        predicted_primary_impact=["T1005"],
        expected_primary_impact=["T1005"],
        predicted_exploitation=["T1190"],
        expected_exploitation=[],
        k=5,
    )
    assert "primary_impact_hit" in no_ex
    assert "exploitation_hit" not in no_ex


def test_agent_mocked_reports_head_hit_rates(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")
    _echo_agent(monkeypatch)

    cases = [c for c in load_cases() if c["task"] == "cve_to_attack"][:5]
    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        cases=cases,
    )
    m = payload["metrics"]["attack"]
    assert m["hit_rate"] == 1.0
    assert "primary_impact_hit_rate" in m
    scores = [r["scores"] for r in payload["results"]]
    assert any("primary_impact_hit" in s for s in scores)


def test_canonical_match_parent_sub_not_siblings() -> None:
    assert canonical_match("T1204", "T1204.002")
    assert canonical_match("T1204.002", "T1204")
    assert canonical_match("t1204", "T1204.002")
    assert not canonical_match("T1566.001", "T1566.002")
    assert not canonical_match("T1190", "T1203")
    assert canonical_match("T1190", "T1190")


def test_exact_vs_canonical_scoring() -> None:
    # Exact miss, canonical hit (parent ↔ sub)
    soft = score_attack(["T1204"], ["T1204.002"], k=5)
    assert soft["hit"] == 0.0
    assert soft["recall_at_k"] == 0.0
    assert soft["canonical_hit"] == 1.0
    assert soft["canonical_recall_at_k"] == 1.0

    # Sibling subs: neither exact nor canonical
    sib = score_attack(["T1566.001"], ["T1566.002"], k=5)
    assert sib["hit"] == 0.0
    assert sib["canonical_hit"] == 0.0

    # Exact hit implies canonical hit
    exact = score_attack(["T1005", "T1190"], ["T1005"], k=5)
    assert exact["hit"] == 1.0
    assert exact["canonical_hit"] == 1.0

    head = score_technique_head(["T1204"], ["T1204.001", "T1499.004"], k=5)
    assert head is not None
    assert head["hit"] == 0.0
    assert head["canonical_hit"] == 1.0
    assert head["canonical_recall_at_k"] == 0.5  # one of two preds? wait: 1 pred matches 1 of 2 exp → hits=1, recall=1/2

    full = score_attack_with_heads(
        predicted_techniques=["T1204"],
        expected_techniques=["T1204.002"],
        predicted_primary_impact=["T1204"],
        expected_primary_impact=["T1204.002"],
        predicted_exploitation=["T1566.001"],
        expected_exploitation=["T1566.002"],
        k=5,
    )
    assert full["hit"] == 0.0
    assert full["canonical_hit"] == 1.0
    assert full["primary_impact_canonical_hit"] == 1.0
    assert full["exploitation_hit"] == 0.0
    assert full["exploitation_canonical_hit"] == 0.0


def test_resolve_jobs_auto_cap(monkeypatch) -> None:
    from evals.golden.runner import resolve_jobs, default_jobs_cap, DEFAULT_JOBS_CAP

    monkeypatch.delenv("GOLDEN_JOBS_CAP", raising=False)
    assert DEFAULT_JOBS_CAP == 8
    assert default_jobs_cap() == 8
    assert resolve_jobs(20, None) == 8
    assert resolve_jobs(2, None) == 2
    assert resolve_jobs(0, None) == 1
    assert resolve_jobs(20, 1) == 1  # serial repro
    assert resolve_jobs(20, 12) == 12  # explicit override
    assert resolve_jobs(3, 8) == 3  # still capped by n_cases

    monkeypatch.setenv("GOLDEN_JOBS_CAP", "6")
    assert default_jobs_cap() == 6
    assert resolve_jobs(20, None) == 6


def test_jobs_one_matches_serial_semantics(tmp_path: Path, monkeypatch) -> None:
    """--jobs 1 and parallel path produce same ordered results."""
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")
    monkeypatch.delenv("GOLDEN_JOBS_CAP", raising=False)
    _echo_agent(monkeypatch)

    cases = [c for c in load_cases() if c["task"] == "cve_triage"][:8]
    serial = run_eval(
        mode="agent",
        task="cve_triage",
        seed=42,
        write_results=False,
        cases=cases,
        jobs=1,
    )
    parallel = run_eval(
        mode="agent",
        task="cve_triage",
        seed=42,
        write_results=False,
        cases=cases,
        jobs=4,
    )
    assert [r["id"] for r in serial["results"]] == [r["id"] for r in parallel["results"]]
    assert [r.get("scores") for r in serial["results"]] == [
        r.get("scores") for r in parallel["results"]
    ]
    assert serial["metrics"]["accuracy"] == parallel["metrics"]["accuracy"]
    assert serial["jobs"] == 1
    assert parallel["jobs"] == 4


def test_jobs_auto_default_min_n_cap(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")
    monkeypatch.delenv("GOLDEN_JOBS_CAP", raising=False)
    _echo_agent(monkeypatch)

    cases = [c for c in load_cases() if c["task"] == "cve_triage"][:3]
    payload = run_eval(
        mode="agent",
        task="cve_triage",
        write_results=True,
        cases=cases,
        jobs=None,
    )
    assert payload["jobs"] == 3
    assert (tmp_path / "latest.json").is_file()
    import json

    data = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert data["jobs"] == 3
    assert [r["id"] for r in data["results"]] == [c["id"] for c in cases]


def test_cli_jobs_flag() -> None:
    from evals.golden.runner import build_parser

    assert build_parser().parse_args([]).jobs is None
    assert build_parser().parse_args(["--jobs", "1"]).jobs == 1
    assert build_parser().parse_args(["--jobs", "4"]).jobs == 4


def test_cli_defaults_live_agent() -> None:
    from evals.golden.runner import build_parser

    args = build_parser().parse_args([])
    assert args.mode == "agent"
    assert args.task == "cve_to_attack"


def test_agent_predict_triage_does_not_nameerror(monkeypatch) -> None:  # noqa: ANN001
    """Regression: triage RAG still needs build_triage_context imported in runner."""
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "default_llm_chat",
        lambda _messages, model=None: {"triage_label": "ACT", "_raw": "{}"},  # noqa: ARG005
    )
    out = runner_mod._agent_predict(
        {
            "task": "cve_triage",
            "input": {
                "description": "RCE",
                "cvss_score": 9.8,
                "epss": 0.9,
                "in_kev": True,
                "critical_asset": True,
            },
        },
        use_rag=True,
    )
    assert out["triage_label"] == "ACT"
    assert out["mode"] == "agent_llm"
