"""Tests for model × RAG cost grid helpers (no network)."""

from evals.golden.model_grid import format_grid_table, list_price_cost


def test_list_price_cost_deepseek() -> None:
    # 3366 in + 222 out at catalog rates
    c = list_price_cost(3366, 222, "deepseek-v4-flash")
    assert c is not None
    assert 0.00015 < c < 0.00018


def test_list_price_cost_unknown() -> None:
    assert list_price_cost(100, 10, "no-such-model") is None


def test_format_grid_table() -> None:
    md = format_grid_table(
        [
            {
                "model": "deepseek-v4-flash",
                "rag": True,
                "n_ok": 20,
                "hit": 0.5,
                "recall_at_k": 0.3,
                "precision_at_k": 0.28,
                "total_tokens": 70000,
                "list_avg_cost_per_case_usd": 0.00016,
                "observed_total_cost_usd": 0.0,
            }
        ]
    )
    assert "deepseek-v4-flash" in md
    assert "on" in md
    assert "$0.0002" in md or "$0.0001" in md
