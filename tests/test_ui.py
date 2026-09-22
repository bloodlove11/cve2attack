"""Optional Streamlit UI smoke tests (streamlit not required for core suite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui import APP_MODULE


def test_ui_package_exports() -> None:
    assert APP_MODULE == "src.ui.app"
    assert Path("src/ui/app.py").is_file()
    assert Path("src/ui/helpers.py").is_file()
    assert Path("src/ui/app_pages/chat.py").is_file()
    assert Path("src/ui/app_pages/eval_lab.py").is_file()


def test_eval_lab_is_primary_surface() -> None:
    app = Path("src/ui/app.py").read_text(encoding="utf-8")
    eval_lab = Path("src/ui/app_pages/eval_lab.py").read_text(encoding="utf-8")
    # Golden eval is the default Streamlit page
    assert 'title="Golden eval"' in app
    assert "default=True" in app
    assert app.index("eval_lab.py") < app.index("chat.py")
    # Live agent defaults for the focused workflow
    assert "Live agent" in eval_lab
    assert '"cve_to_attack"' in eval_lab
    assert eval_lab.index('"cve_to_attack"') < eval_lab.index('"cve_triage"')
    assert "CIRCL test (n=121)" in eval_lab
    assert "Internal golden" in eval_lab
    # Internal first so Live smokes don't auto-holdout CIRCL n=121
    assert eval_lab.index("Internal golden") < eval_lab.index("CIRCL test")


def test_eval_lab_is_live_agent_only() -> None:
    text = Path("src/ui/app_pages/eval_lab.py").read_text(encoding="utf-8")
    assert "Live agent" in text
    assert "circl_test" in text
    assert '"Prior"' not in text
    assert '"kNN"' not in text
    assert '"Policy"' not in text
    assert '"KB"' not in text
    assert '"Hybrid"' not in text


def test_chat_page_is_secondary() -> None:
    text = Path("src/ui/app_pages/chat.py").read_text(encoding="utf-8")
    assert "secondary" in text.lower()
    assert "Enable RAG" not in text
    assert "Enable rerank" not in text
    assert "STRIDE" not in text
    assert "Live" in text or "live" in text
    assert "lookup_ctid_mapping" not in text
    assert "CTID KB lookup" not in text

def test_ui_helpers_without_streamlit() -> None:
    """Agent helpers import without requiring streamlit at import time."""
    from src.ui.helpers import api_key_configured, format_trace_summary, run_query

    assert isinstance(api_key_configured(), bool)
    assert "No trace" in format_trace_summary({})
    assert "tool call" in format_trace_summary(
        {"trace_id": "abc", "path": "/tmp/t.json", "steps": [], "meta": {}}
    )
    assert callable(run_query)


def test_ui_app_main_importable_when_streamlit_present() -> None:
    pytest.importorskip("streamlit")
    from src.ui import app as ui_app
    from src.ui import helpers as ui_helpers

    assert ui_app is not None
    assert callable(ui_helpers.run_query)
    assert callable(ui_helpers.api_key_configured)

def test_results_dataframe_stringifies_technique_lists():
    """Lists in predicted/expected must be strings for Arrow/Streamlit."""
    from src.ui.helpers import results_dataframe

    payload = {
        "results": [
            {
                "id": "attack-CVE-1",
                "task": "cve_to_attack",
                "skipped": False,
                "correct": True,
                "prediction": {
                    "attack_techniques": ["T1190", "T1059"],
                    "primary_impact": ["T1005"],
                    "exploitation_techniques": ["T1190"],
                },
                "expected": {"attack_techniques": ["T1190"]},
                "scores": {"hit": 1.0, "recall_at_k": 1.0, "primary_impact_hit": 1.0},
            },
            {
                "id": "triage-1",
                "task": "cve_triage",
                "skipped": False,
                "correct": True,
                "prediction": {"triage_label": "ACT"},
                "expected": {"triage_label": "ACT"},
                "scores": {"label_exact_match": 1.0},
            },
        ]
    }
    rows = results_dataframe(payload)
    assert rows[0]["predicted"] == "T1190, T1059"
    assert rows[0]["expected"] == "T1190"
    assert isinstance(rows[0]["predicted"], str)
    assert rows[0]["predicted_primary_impact"] == "T1005"
    assert rows[0]["predicted_exploitation"] == "T1190"
    assert isinstance(rows[0]["predicted_primary_impact"], str)
    assert rows[1]["predicted"] == "ACT"
    assert "predicted_primary_impact" not in rows[1]



def test_session_history_from_messages() -> None:
    from src.ui.helpers import session_history_from_messages
    from src.schemas.answer import AgentAnswer

    history = session_history_from_messages(
        [
            {"role": "user", "content": "Map CVE-1 to ATT&CK"},
            {
                "role": "assistant",
                "answer": AgentAnswer(
                    answer="T1190", confidence=0.8, tools_used=[], status="ok"
                ),
            },
            {"role": "user", "content": "and exploitation?"},
        ]
    )
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "T1190"
    assert history[2]["content"] == "and exploitation?"


def test_resolve_ui_model_presets(monkeypatch) -> None:
    monkeypatch.delenv("EXPLABS_MODEL", raising=False)
    from src.ui.helpers import resolve_ui_model, STARTER_MODELS

    assert resolve_ui_model("deepseek-v4-flash") == "deepseek-v4-flash"
    assert resolve_ui_model("gpt-6-astra") == "gpt-6-astra"
    assert STARTER_MODELS[0] == "deepseek-v4-flash"
    for mid in (
        "deepseek-v4-flash",
        "gpt-6-astra",
        "gpt-5.6-luna",
        "claude-fable-5.1",
        "qwen3.8-27b",
        "gpt-5.6-sol",
        "claude-sonnet-5",
    ):
        assert mid in STARTER_MODELS


def test_resolve_ui_model_custom(monkeypatch) -> None:
    monkeypatch.setenv("EXPLABS_MODEL", "env-model")
    from src.ui.helpers import apply_runtime_model, resolve_ui_model

    assert resolve_ui_model("Custom / EXPLABS_MODEL", "my-custom") == "my-custom"
    assert resolve_ui_model("Custom / EXPLABS_MODEL", "") == "env-model"
    assert apply_runtime_model("gpt-6-astra") == "gpt-6-astra"
    import os

    assert os.environ["EXPLABS_MODEL"] == "gpt-6-astra"
