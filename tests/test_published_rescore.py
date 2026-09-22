"""Re-score published CIRCL artifacts with the current scorer.

Protects EVAL_REPORT.md headline floats from silent score.py drift.
Of-record JSON is a frozen older pipeline; this checks scoring math only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.golden.score import score_attack_with_heads, summarize

_PUB = Path("evals/golden/published")


def _rescore(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = []
    for row in payload.get("results") or []:
        if row.get("skipped"):
            continue
        pred = row.get("prediction") or {}
        exp = row.get("expected") or {}
        scores.append(
            score_attack_with_heads(
                predicted_techniques=list(pred.get("attack_techniques") or []),
                expected_techniques=list(exp.get("attack_techniques") or []),
                predicted_primary_impact=list(pred.get("primary_impact") or []),
                expected_primary_impact=(
                    list(exp["primary_impact"])
                    if exp.get("primary_impact") is not None
                    else None
                ),
                predicted_exploitation=list(pred.get("exploitation_techniques") or []),
                expected_exploitation=(
                    list(exp["exploitation_techniques"])
                    if exp.get("exploitation_techniques") is not None
                    else None
                ),
                k=5,
            )
        )
    return summarize(scores)


@pytest.mark.parametrize(
    "filename,hit,recall,precision",
    [
        (
            "circl_test_live_refine.json",
            0.5041322314049587,
            0.28771102634739,
            0.28374655647382924,
        ),
        (
            "circl_test_live_norefine.json",
            0.19834710743801653,
            0.10663452708907253,
            0.16115702479338842,
        ),
    ],
)
def test_published_artifact_rescores(
    filename: str, hit: float, recall: float, precision: float
) -> None:
    path = _PUB / filename
    assert path.is_file()
    att = _rescore(path)
    stored = json.loads(path.read_text(encoding="utf-8"))["metrics"]["attack"]
    assert att["hit"] == pytest.approx(stored["hit"], abs=1e-12)
    assert att["recall_at_k"] == pytest.approx(stored["recall_at_k"], abs=1e-12)
    assert att["precision_at_k"] == pytest.approx(stored["precision_at_k"], abs=1e-12)
    assert att["hit"] == pytest.approx(hit, abs=1e-12)
    assert att["recall_at_k"] == pytest.approx(recall, abs=1e-12)
    assert att["precision_at_k"] == pytest.approx(precision, abs=1e-12)
