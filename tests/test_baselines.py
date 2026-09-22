"""Contamination / protocol bookkeeping (non-LLM predictors removed)."""
from __future__ import annotations

import json
from pathlib import Path

from evals.golden.circl_import import circl_test_cve_ids
from evals.golden.heldout import heldout_case_ids
from evals.golden.score import load_cases


def test_contamination_file_matches_known_overlap() -> None:
    data = json.loads(Path("evals/golden/contamination.json").read_text())
    internal = {
        str((c.get("input") or {}).get("cve_id") or "").upper()
        for c in load_cases()
    }
    internal.discard("")
    circl_test = set(circl_test_cve_ids())
    computed = internal & circl_test
    assert set(data["circl_test_overlap_cves"]) == computed
    n20_cves: set[str] = set()
    by_id = {c.get("id"): c for c in load_cases()}
    for case_id in heldout_case_ids():
        inp = (by_id.get(case_id) or {}).get("input") or {}
        cid = str(inp.get("cve_id") or "").upper()
        if cid:
            n20_cves.add(cid)
    assert set(data["heldout_n20_overlap_cves"]) == (n20_cves & circl_test)
    assert len(data["heldout_n20_overlap_cves"]) == 3
