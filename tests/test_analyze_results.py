"""Smoke test for published-result error analysis helper."""
from __future__ import annotations

from evals.golden.analyze_results import analyze


def test_analyze_counts_empty_and_fn() -> None:
    payload = {
        "mode": "agent",
        "protocol": "circl_test",
        "refine_exploitation": False,
        "results": [
            {
                "id": "a",
                "skipped": False,
                "correct": False,
                "prediction": {"attack_techniques": []},
                "expected": {"attack_techniques": ["T1190"]},
                "scores": {"hit": 0.0, "recall_at_k": 0.0},
            },
            {
                "id": "b",
                "skipped": False,
                "correct": True,
                "prediction": {"attack_techniques": ["T1190", "T1499.004"]},
                "expected": {
                    "attack_techniques": ["T1190"],
                    "exploitation_techniques": ["T1190"],
                },
                "scores": {
                    "hit": 1.0,
                    "recall_at_k": 1.0,
                    "exploitation_hit": 1.0,
                },
            },
        ],
    }
    report = analyze(payload)
    assert report["n_scored"] == 2
    assert report["hit_rate"] == 0.5
    assert report["empty_prediction_rate"] == 0.5
    assert report["top_false_negatives"][0][0] == "T1190"
    assert report["top_false_positives"][0][0] == "T1499.004"
