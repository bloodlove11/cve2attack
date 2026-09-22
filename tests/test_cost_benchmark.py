"""Unit tests for LLM usage / cost helpers and eval cost aggregation."""

from __future__ import annotations

from typing import Any

from evals.golden.cost_benchmark import format_table
from evals.golden.cost_metrics import aggregate_cost_metrics, cost_row_from_payload
from evals.golden.live_attack import default_llm_chat
from evals.golden.runner import _aggregate, markdown_report
from src.llm.usage import add_usage, empty_usage, extract_usage


class _Usage:
    def __init__(
        self,
        *,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int = 15,
        cost: float | None = 0.0012,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cost = cost


class _Msg:
    content = '{"exploitation_techniques":["T1190"],"primary_impact":["T1059"]}'
    tool_calls = None


class _Choice:
    message = _Msg()


class _Resp:
    def __init__(self, usage: Any = None) -> None:
        self.choices = [_Choice()]
        self.usage = usage


def test_extract_usage_with_cost() -> None:
    u = extract_usage(_Resp(_Usage(cost=0.002)))
    assert u is not None
    assert u["prompt_tokens"] == 10
    assert u["completion_tokens"] == 5
    assert u["total_tokens"] == 15
    assert u["cost"] == 0.002
    assert u["n_calls"] == 1
    assert u["n_calls_with_cost"] == 1


def test_extract_usage_free_tier_zero_cost() -> None:
    u = extract_usage(_Resp(_Usage(cost=0.0)))
    assert u is not None
    assert u["cost"] == 0.0
    assert u["n_calls_with_cost"] == 1


def test_extract_usage_missing() -> None:
    assert extract_usage(_Resp(None)) is None
    assert extract_usage({"choices": []}) is None


def test_add_usage_sums() -> None:
    a = extract_usage(_Resp(_Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12, cost=0.01)))
    b = extract_usage(_Resp(_Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8, cost=0.02)))
    s = add_usage(a, b)
    assert s is not None
    assert s["prompt_tokens"] == 15
    assert s["completion_tokens"] == 5
    assert s["total_tokens"] == 20
    assert abs(s["cost"] - 0.03) < 1e-9
    assert s["n_calls"] == 2


def test_add_usage_none_identity() -> None:
    a = empty_usage()
    a["prompt_tokens"] = 3
    a["n_calls"] = 1
    assert add_usage(None, a) == a
    assert add_usage(a, None) == a
    assert add_usage(None, None) is None


def test_default_llm_chat_attaches_usage(monkeypatch: Any) -> None:
    def fake_chat_completion(*_a: Any, **_k: Any) -> Any:
        return _Resp(_Usage(cost=0.004))

    monkeypatch.setattr("src.llm.client.chat_completion", fake_chat_completion)
    monkeypatch.setattr("src.llm.client.get_model", lambda: "test-model")
    parsed = default_llm_chat([{"role": "user", "content": "hi"}])
    assert parsed.get("_usage", {}).get("cost") == 0.004
    assert parsed.get("_model") == "test-model"
    assert "exploitation_techniques" in parsed or "_raw" in parsed


def test_aggregate_cost_metrics_and_markdown() -> None:
    rows = [
        {
            "id": "a",
            "task": "cve_to_attack",
            "skipped": False,
            "correct": True,
            "scores": {"hit": 1.0, "recall_at_k": 0.5, "precision_at_k": 0.5},
            "prediction": {
                "attack_techniques": ["T1190"],
                "mode": "agent_llm",
                "llm_calls": 2,
                "model": "deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                    "cost": 0.01,
                    "n_calls": 2,
                    "n_calls_with_cost": 2,
                },
            },
        },
        {
            "id": "b",
            "task": "cve_to_attack",
            "skipped": False,
            "correct": False,
            "scores": {"hit": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0},
            "prediction": {
                "attack_techniques": [],
                "mode": "agent_llm",
                "llm_calls": 2,
                "model": "deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                    "cost": 0.005,
                    "n_calls": 2,
                    "n_calls_with_cost": 2,
                },
            },
        },
    ]
    metrics = _aggregate(rows, mode="agent")
    cost = metrics["cost"]
    assert cost["total_cost_usd"] == 0.015
    assert cost["total_tokens"] == 240
    assert cost["n_cases_with_usage"] == 2
    assert cost["avg_cost_per_case_usd"] == 0.0075
    assert abs(cost["cost_per_hit_usd"] - 0.015) < 1e-9  # 0.015 / 1 hit
    md = markdown_report(
        {
            "mode": "agent",
            "task": "cve_to_attack",
            "model": "deepseek-v4-flash",
            "metrics": metrics,
            "results": rows,
        }
    )
    assert "Cost / tokens" in md
    assert "total_cost_usd" in md


def test_cost_benchmark_table() -> None:
    payload = {
        "model": "qwen3.8-27b",
        "protocol": "circl_test",
        "refine_exploitation": True,
        "metrics": {
            "n_scored": 2,
            "hit_rate": 0.5,
            "attack": {
                "hit_rate": 0.5,
                "recall_at_k": 0.25,
                "precision_at_k": 0.4,
                "n": 2,
            },
            "cost": {
                "model": "qwen3.8-27b",
                "total_cost_usd": 0.0,
                "avg_cost_per_case_usd": 0.0,
                "cost_per_hit_usd": 0.0,
                "total_tokens": 200,
                "avg_tokens_per_case": 100.0,
                "total_llm_calls": 4,
                "n_cases_with_usage": 2,
            },
        },
        "results": [],
    }
    row = cost_row_from_payload(payload)
    assert row["model"] == "qwen3.8-27b"
    assert row["total_cost_usd"] == 0.0
    table = format_table([row])
    assert "qwen3.8-27b" in table
    assert "$0.0000" in table


def test_aggregate_without_usage() -> None:
    rows = [
        {
            "id": "legacy",
            "task": "cve_to_attack",
            "skipped": False,
            "scores": {"hit": 1.0},
            "prediction": {"attack_techniques": ["T1190"], "llm_calls": 2},
        }
    ]
    cost = aggregate_cost_metrics(rows, hit_rate=1.0)
    assert cost["n_cases_with_usage"] == 0
    assert cost["total_cost_usd"] is None
    assert cost["total_llm_calls"] == 2
