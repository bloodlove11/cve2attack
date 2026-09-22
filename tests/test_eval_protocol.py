"""P0 eval protocol: held-out freeze load + multi-seed aggregate helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.golden.heldout import (
    ATTACK_METRIC_KEYS,
    HELDOUT_PATH,
    aggregate_seed_metrics,
    extract_attack_metrics,
    format_mean_std,
    heldout_case_ids,
    load_heldout,
    parse_seeds,
)
from evals.golden.runner import _filter_cases, build_parser, run_eval
from evals.golden.score import load_cases


def test_load_heldout_freeze() -> None:
    data = load_heldout()
    assert data["protocol"] == "deepseek_live_n20"
    assert data["task"] == "cve_to_attack"
    assert data["n"] == 20
    ids = heldout_case_ids()
    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert ids[0] == "attack-CVE-2019-6538"
    assert HELDOUT_PATH.is_file()
    # Frozen list matches first 20 attack cases in dataset order
    attack = [c["id"] for c in load_cases() if c["task"] == "cve_to_attack"]
    assert ids == attack[:20]
    assert "circl_import_future" in data


def test_parse_seeds() -> None:
    assert parse_seeds("42,43,44") == [42, 43, 44]
    assert parse_seeds([7, 8]) == [7, 8]
    with pytest.raises(ValueError):
        parse_seeds("")
    with pytest.raises(ValueError):
        parse_seeds([])


def test_aggregate_seed_metrics_mean_std() -> None:
    payloads = [
        {
            "seed": 42,
            "metrics": {
                "attack": {
                    "hit_rate": 0.8,
                    "primary_impact_hit_rate": 0.75,
                    "exploitation_hit_rate": 0.2,
                    "precision_at_k": 0.7,
                    "recall_at_k": 0.1,
                }
            },
        },
        {
            "seed": 43,
            "metrics": {
                "attack": {
                    "hit_rate": 0.6,
                    "primary_impact_hit_rate": 0.65,
                    "exploitation_hit_rate": 0.1,
                    "precision_at_k": 0.5,
                    "recall_at_k": 0.2,
                }
            },
        },
    ]
    agg = aggregate_seed_metrics(payloads)
    assert agg["n_seeds"] == 2
    assert agg["seeds"] == [42, 43]
    hit = agg["metrics"]["hit_rate"]
    assert hit["mean"] == pytest.approx(0.7)
    assert hit["std"] == pytest.approx(0.1414213562373095)
    for key in ATTACK_METRIC_KEYS:
        assert key in agg["metrics"]
    text = format_mean_std(agg)
    assert "hit_rate" in text
    assert "±" in text


def test_extract_attack_metrics_flat_aliases() -> None:
    payload = {
        "metrics": {
            "hit_rate": 0.5,
            "precision_at_k": 0.4,
            "recall_at_k": 0.3,
            "attack_primary_impact_hit_rate": 0.55,
            "attack_exploitation_hit_rate": 0.15,
        }
    }
    mets = extract_attack_metrics(payload)
    assert mets["hit_rate"] == 0.5
    assert mets["primary_impact_hit_rate"] == 0.55
    assert mets["exploitation_hit_rate"] == 0.15


def test_filter_cases_by_heldout_ids() -> None:
    cases = load_cases()
    ids = heldout_case_ids()
    selected = _filter_cases(
        cases, task="cve_to_attack", limit=99, seed=42, case_ids=ids
    )
    assert [c["id"] for c in selected] == ids
    # seed ignored when case_ids set; limit larger than freeze keeps full list
    assert len(selected) == 20


def test_filter_cases_limit_is_prefix_of_frozen_ids() -> None:
    cases = load_cases()
    ids = heldout_case_ids()
    selected = _filter_cases(
        cases, task="cve_to_attack", limit=3, seed=42, case_ids=ids
    )
    assert [c["id"] for c in selected] == ids[:3]


def test_run_eval_heldout_agent_mocked(tmp_path: Path, monkeypatch) -> None:
    """Held-out selection works offline with mocked Live agent."""
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    def fake(case, **_kwargs):  # noqa: ANN001
        exp = case.get("expected") or {}
        return {
            "attack_techniques": list(exp.get("attack_techniques") or []),
            "primary_impact": list(exp.get("primary_impact") or []),
            "exploitation_techniques": list(exp.get("exploitation_techniques") or []),
            "mode": "agent_llm",
        }

    monkeypatch.setattr(runner_mod, "_agent_predict", fake)

    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        heldout=True,
        write_results=True,
        jobs=1,
    )
    assert payload["heldout"] is True
    assert payload["metrics"]["n_scored"] == 20
    assert payload["case_ids"] == heldout_case_ids()
    assert payload["seed"] is None
    from evals.golden.live_attack import LIVE_PIPELINE_VERSION

    assert payload["pipeline_version"] == LIVE_PIPELINE_VERSION


def test_cli_seeds_and_heldout_flags() -> None:
    p = build_parser()
    assert p.parse_args([]).seeds is None
    assert p.parse_args([]).heldout is False
    args = p.parse_args(["--seeds", "42,43,44", "--heldout", "--jobs", "1"])
    assert args.seeds == "42,43,44"
    assert args.heldout is True
    assert args.jobs == 1


def test_multi_seed_aggregate_with_offline_runs(tmp_path: Path, monkeypatch) -> None:
    """multi_seed orchestration over mocked agent (no API) + aggregate file write."""
    import evals.golden.multi_seed as ms
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ms, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ms, "MULTI_LATEST_JSON", tmp_path / "multi_seed_latest.json")
    monkeypatch.setattr(ms, "MULTI_LATEST_MD", tmp_path / "multi_seed_latest.md")

    def fake(case, **_kwargs):  # noqa: ANN001
        exp = case.get("expected") or {}
        return {
            "attack_techniques": list(exp.get("attack_techniques") or []),
            "primary_impact": list(exp.get("primary_impact") or []),
            "exploitation_techniques": list(exp.get("exploitation_techniques") or []),
            "mode": "agent_llm",
        }

    monkeypatch.setattr(runner_mod, "_agent_predict", fake)

    agg = ms.run_multi_seed(
        mode="agent",
        task="cve_to_attack",
        limit=5,
        seeds="42,43",
        write_results=True,
        jobs=1,
        heldout=False,
    )
    assert agg["n_seeds"] == 2
    assert "hit_rate" in agg["metrics"]
    assert (tmp_path / "multi_seed_latest.json").is_file()
    data = json.loads((tmp_path / "multi_seed_latest.json").read_text())
    assert data["n_seeds"] == 2
    assert len(data["runs"]) == 2
