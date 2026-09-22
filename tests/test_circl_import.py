"""CIRCL HF gold import: pin revision, schema mapping, train-only neighbors, heldout."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals.golden.circl_import import (
    HF_REVISION,
    PROTOCOL_CIRCL_TEST,
    case_id_for_cve,
    circl_test_cve_ids,
    clear_circl_caches,
    load_circl_cases,
    load_circl_heldout,
    load_circl_rows,
    map_circl_row,
    row_to_golden_case,
)
from evals.golden.eval_rag import retrieve_labeled_neighbors
from evals.golden.heldout import heldout_case_ids, load_heldout
from evals.golden.runner import _filter_cases, build_parser, run_eval
from evals.golden.score import score_attack_with_heads


def _sample_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": "CVE-2099-0001",
        "title": "Test vuln",
        "description": "Remote code execution via crafted input.",
        "exploitation_techniques": ["T1190"],
        "primary_impact": ["T1005"],
        "secondary_impact": ["T1499.004"],
        "techniques": ["T1190", "T1005", "T1499.004"],
        "techniques_derived": ["T1574.007", "T1059"],  # must never become gold
        "label_sources": ["ctid_cve"],
        "attack_version": "19.1",
        "cwes": ["CWE-94"],
    }
    row.update(overrides)
    return row


def test_pin_revision_constant_and_local_meta() -> None:
    meta = Path("evals/golden/raw/circl/REVISION.json")
    assert meta.is_file()
    data = __import__("json").loads(meta.read_text())
    assert data["hf_revision"] == HF_REVISION
    assert HF_REVISION == "319c3e324e7561592cf7e51ec003deb5dd90e61b"


def test_load_circl_rows_pinned_revision_mock() -> None:
    """Pin revision is passed to datasets.load_dataset when local cache absent."""
    fake_rows = [_sample_row()]

    class _FakeSplit:
        def __iter__(self):
            return iter(fake_rows)

    class _FakeDS(dict):
        pass

    fake_ds = _FakeDS(train=_FakeSplit(), test=_FakeSplit())
    captured: dict[str, Any] = {}

    def _fake_load(name: str, revision: str | None = None, **kwargs: Any):
        captured["name"] = name
        captured["revision"] = revision
        return fake_ds

    with patch("evals.golden.circl_import._parquet_path") as pp:
        pp.return_value = Path("/nonexistent/nope.parquet")
        with patch.dict("sys.modules", {"datasets": type("M", (), {})()}):
            # Inject load_dataset on a fake module via import inside function:
            # patch the import site by providing datasets with load_dataset.
            import types

            mod = types.ModuleType("datasets")
            mod.load_dataset = _fake_load  # type: ignore[attr-defined]
            with patch.dict("sys.modules", {"datasets": mod}):
                loaded = load_circl_rows(
                    "test", revision=HF_REVISION, prefer_local=False
                )
    assert captured["revision"] == HF_REVISION
    assert captured["name"] == "CIRCL/vulnerability-attack-techniques"
    assert len(loaded["test"]) == 1


def test_map_no_invented_heads_and_ignores_derived() -> None:
    mapped = map_circl_row(_sample_row())
    assert mapped["expected"]["attack_techniques"] == [
        "T1190",
        "T1005",
        "T1499.004",
    ]
    assert mapped["expected"]["exploitation_techniques"] == ["T1190"]
    assert mapped["expected"]["primary_impact"] == ["T1005"]
    # derived never appears in gold
    blob = str(mapped["expected"])
    assert "T1574.007" not in blob
    assert "T1059" not in blob
    assert mapped["techniques_derived_ignored"] is True

    # Empty CIRCL heads stay empty; scorers skip them (N/A) rather than invent
    empty_heads = map_circl_row(
        _sample_row(
            exploitation_techniques=[],
            primary_impact=[],
            techniques=["T1005"],
            techniques_derived=["T1059"],
        )
    )
    assert empty_heads["expected"]["exploitation_techniques"] == []
    assert empty_heads["expected"]["primary_impact"] == []
    assert empty_heads["expected"]["attack_techniques"] == ["T1005"]
    scores = score_attack_with_heads(
        predicted_techniques=["T1005"],
        expected_techniques=empty_heads["expected"]["attack_techniques"],
        predicted_primary_impact=["T1005"],
        expected_primary_impact=empty_heads["expected"]["primary_impact"],
        predicted_exploitation=["T1190"],
        expected_exploitation=empty_heads["expected"]["exploitation_techniques"],
    )
    assert "hit" in scores
    assert "primary_impact_hit" not in scores  # empty gold → N/A
    assert "exploitation_hit" not in scores


def test_row_to_golden_case_schema() -> None:
    case = row_to_golden_case(_sample_row(), split="test")
    assert case["id"] == "circl-CVE-2099-0001"
    assert case["task"] == "cve_to_attack"
    assert case["input"]["cve_id"] == "CVE-2099-0001"
    assert case["input"]["cwes"] == ["CWE-94"]
    assert case["protocol"] == PROTOCOL_CIRCL_TEST
    assert "techniques_derived" not in case["expected"]


def test_heldout_circl_freeze() -> None:
    data = load_circl_heldout()
    assert data["protocol"] == "circl_test"
    assert data["n"] == 121
    assert data["selection"]["hf_revision"] == HF_REVISION
    ids = heldout_case_ids(protocol="circl_test")
    assert len(ids) == 121
    assert len(set(ids)) == 121
    assert ids[0].startswith("circl-CVE-")
    # Matches load_heldout protocol path
    assert load_heldout(protocol="circl_test")["case_ids"] == ids
    # deepseek remains separate
    ds = load_heldout(protocol="deepseek_live_n20")
    assert ds["protocol"] == "deepseek_live_n20"
    assert ds["n"] == 20
    assert set(ds["case_ids"]).isdisjoint(set(ids))


def test_local_circl_cases_load_and_filter() -> None:
    cases = load_circl_cases(split="test")
    assert len(cases) == 121
    ids = heldout_case_ids(protocol="circl_test")
    selected = _filter_cases(
        cases, task="cve_to_attack", limit=999, seed=1, case_ids=ids
    )
    assert [c["id"] for c in selected] == ids
    # Every case scores on flat techniques; no derived in expected
    for c in selected[:5]:
        exp = c["expected"]
        assert exp["attack_techniques"]
        assert "techniques_derived" not in exp


def test_train_only_neighbors_hard_exclude_test_ids() -> None:
    clear_circl_caches()
    test_ids = circl_test_cve_ids()
    assert len(test_ids) == 121
    # Pick a test CVE as target
    target = sorted(test_ids)[0]
    enrich = {
        "description": (
            "Remote code execution in a public-facing web application via "
            "crafted HTTP requests. Command injection."
        ),
        "cwes": ["CWE-78"],
        "source": "test",
    }
    neighbors = retrieve_labeled_neighbors(
        target,
        enrichment=enrich,
        k=5,
        use_embeddings=False,
        protocol="circl_test",
    )
    assert neighbors, "expected some CIRCL-train neighbors"
    for n in neighbors:
        cid = str(n.get("cve_id") or "").upper()
        assert cid != target
        assert cid not in test_ids
        assert n.get("source") == "circl_train"


def test_circl_neighbors_do_not_fall_back_to_ctid_corpus(monkeypatch) -> None:
    """BM25 miss / empty query must not pull CTID-labeled neighbors on CIRCL."""
    from evals.golden import eval_rag as rag

    def boom(*_a, **_k):  # noqa: ANN001
        raise AssertionError("ctid retrieve must not run under protocol=circl_test")

    monkeypatch.setattr(rag, "retrieve", boom)
    monkeypatch.setattr(rag, "_tokenize", lambda _q: [])
    neighbors = retrieve_labeled_neighbors(
        "CVE-2099-4242",
        enrichment={"description": "", "cwes": [], "source": "test"},
        k=3,
        use_embeddings=False,
        protocol="circl_test",
    )
    assert neighbors == []


def test_protocol_circl_test_without_heldout_uses_official_test_not_train(
    tmp_path: Path, monkeypatch
) -> None:
    """``--protocol circl_test`` must never score CIRCL train as if it were test."""
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
        protocol="circl_test",
        heldout=False,
        limit=10,
        write_results=False,
        jobs=1,
    )
    assert payload["protocol"] == "circl_test"
    assert payload["metrics"]["n_scored"] == 10
    freeze = heldout_case_ids(protocol="circl_test")
    assert payload["case_ids"] == freeze[:10]
    for row in payload["results"]:
        # Frozen CIRCL test cases carry split=test in the gold adapter
        assert row["id"] in freeze
        assert not str(row.get("id") or "").startswith("unused")


def test_heldout_circl_limit_is_frozen_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    def fake(case, **_kwargs):  # noqa: ANN001
        exp = case.get("expected") or {}
        return {
            "attack_techniques": list(exp.get("attack_techniques") or []),
            "mode": "agent_llm",
        }

    monkeypatch.setattr(runner_mod, "_agent_predict", fake)

    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        heldout=True,
        protocol="circl_test",
        limit=3,
        write_results=False,
        jobs=1,
    )
    freeze = heldout_case_ids(protocol="circl_test")
    assert payload["case_ids"] == freeze[:3]
    assert payload["metrics"]["n_scored"] == 3


def test_cli_protocol_flag() -> None:
    p = build_parser()
    args = p.parse_args(
        ["--heldout", "--protocol", "circl_test", "--task", "cve_to_attack"]
    )
    assert args.heldout is True
    assert args.protocol == "circl_test"


def test_run_eval_circl_heldout_agent_mocked(tmp_path: Path, monkeypatch) -> None:
    """Held-out circl_test selection works offline (mocked agent; no Live API)."""
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
        protocol="circl_test",
        write_results=True,
        jobs=1,
    )
    assert payload["heldout"] is True
    assert payload["protocol"] == "circl_test"
    assert payload["metrics"]["n_scored"] == 121
    assert payload["case_ids"] == heldout_case_ids(protocol="circl_test")
    # Dual metrics present when any case scored
    attack = payload["metrics"].get("attack") or {}
    assert "hit_rate" in attack or "hit" in attack or payload["metrics"].get("hit_rate") is not None
    assert "recall_at_k" in attack or payload["metrics"].get("recall_at_k") is not None


def test_case_id_helpers() -> None:
    assert case_id_for_cve("cve-2020-1234") == "circl-CVE-2020-1234"
