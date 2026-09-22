"""Offline tests for CTID index (Live neighbor ICL scaffolding)."""
from __future__ import annotations

from evals.golden.ctid_index import (
    build_few_shot_exemplars,
    get_index,
    load_ctid_records,
    lookup_ctid_mapping,
    retrieve,
)


def test_ctid_index_loads_many_cves() -> None:
    records = load_ctid_records()
    assert len(records) >= 100
    by_id, cached = get_index()
    assert len(by_id) >= 100
    assert len(cached) == len(records)


def test_exact_lookup_known_cve() -> None:
    hit = lookup_ctid_mapping("CVE-2019-15243")
    assert hit is not None
    assert hit["cve_id"] == "CVE-2019-15243"
    techs = hit["attack_techniques"]
    assert "T1059" in techs
    assert "T1190" in techs or "T1078" in techs
    assert hit["match"] == "exact"


def test_lookup_missing_cve() -> None:
    assert lookup_ctid_mapping("CVE-1900-0001") is None


def test_retrieve_exact_first() -> None:
    rows = retrieve(cve_id="CVE-2019-15243", k=3)
    assert rows
    assert rows[0]["cve_id"] == "CVE-2019-15243"
    assert rows[0]["match"] == "exact"


def test_retrieve_heldout_excludes_target() -> None:
    rows = retrieve(
        cve_id="CVE-2019-15243",
        k=5,
        exclude_cve_ids={"CVE-2019-15243"},
        allow_exact=False,
    )
    assert all(r["cve_id"] != "CVE-2019-15243" for r in rows)


def test_few_shot_excludes_target() -> None:
    target = "CVE-2019-15243"
    exemplars = build_few_shot_exemplars(target, n=3, seed=7)
    assert len(exemplars) == 3
    assert all(e["cve_id"] != target for e in exemplars)
    assert all(e.get("attack_techniques") for e in exemplars)
