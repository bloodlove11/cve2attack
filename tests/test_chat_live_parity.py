"""Chat product path shares Live ATT&CK helper with golden mode=agent."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import src.api.server as api_server
from fastapi.testclient import TestClient

from evals.golden.live_attack import predict_cve_to_attack
from src.agent.dispatch import run_chat_query
from src.attack.detect import extract_cve_to_attack_intent
from src.schemas.answer import AgentAnswer


def test_detect_cve_to_attack_intent() -> None:
    assert (
        extract_cve_to_attack_intent(
            "Map CVE-2021-44228 to MITRE ATT&CK techniques"
        )
        == "CVE-2021-44228"
    )
    assert extract_cve_to_attack_intent("Explain ACT vs ATTEND vs TRACK.") is None
    # Triage without ATT&CK language → not Live ATT&CK
    assert (
        extract_cve_to_attack_intent(
            "Triage CVE-2024-1234 for severity and recommended action."
        )
        is None
    )


def test_detect_cve_from_history_followup() -> None:
    prior = "user: Map CVE-2021-44228 to MITRE ATT&CK techniques\nassistant: T1190"
    assert (
        extract_cve_to_attack_intent(
            "What are the exploitation techniques for that CVE?",
            prior_text=prior,
        )
        == "CVE-2021-44228"
    )
    assert (
        extract_cve_to_attack_intent(
            "What's the CVSS and KEV severity for that CVE?",
            prior_text=prior,
        )
        is None
    )


def test_predict_cve_to_attack_is_shared_entrypoint() -> None:
    """Eval + Chat both call evals.golden.live_attack.predict_cve_to_attack."""
    import evals.golden.runner as runner_mod

    assert runner_mod.predict_cve_to_attack is predict_cve_to_attack


def test_chat_dispatch_uses_live_pipeline(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    def _fake_predict(cve_id: str, **kwargs: Any) -> dict[str, Any]:
        called["cve_id"] = cve_id
        called["kwargs"] = kwargs
        return {
            "cve_id": cve_id,
            "exploitation_techniques": ["T1190"],
            "primary_impact": ["T1005"],
            "attack_techniques": ["T1190", "T1005"],
            "mode": "agent_llm",
            "enrichment_source": "test",
            "pipeline": "live_attack",
        }

    monkeypatch.setattr(
        "evals.golden.live_attack.predict_cve_to_attack", _fake_predict
    )
    # Also patch the import site used inside chat_live
    monkeypatch.setattr(
        "src.attack.chat_live.predict_cve_to_attack",
        _fake_predict,
        raising=False,
    )

    answer = run_chat_query(
        "Map CVE-2019-15243 to MITRE ATT&CK techniques",
        trace=False,
    )
    assert called["cve_id"] == "CVE-2019-15243"
    assert "live_attack_pipeline" in answer.tools_used
    assert "T1190" in answer.answer
    assert "T1005" in answer.answer
    assert "Live ATT&CK" in answer.answer


def test_chat_dispatch_non_mapping_uses_agent(monkeypatch: Any) -> None:
    def _boom(*_a: Any, **_k: Any) -> AgentAnswer:
        raise AssertionError("Live pipeline must not run for a non-mapping query")

    monkeypatch.setattr("src.agent.dispatch.run_live_attack_answer", _boom)

    def chat_fn(messages, **kwargs):  # noqa: ANN001
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=(
                            '{"answer":"ACT remediates now; ATTEND investigates; '
                            'TRACK monitors.","confidence":1.0,'
                            '"tools_used":[],"citations":[],"status":"ok"}'
                        ),
                        tool_calls=[],
                    )
                )
            ]
        )

    ans = run_chat_query("Explain ACT vs ATTEND vs TRACK.", chat_fn=chat_fn, trace=False)
    assert ans.status == "ok"
    assert "ACT" in ans.answer


def test_api_chat_optional_key_gate(monkeypatch: Any) -> None:
    def fake_run(query, **kwargs):  # noqa: ANN001
        return AgentAnswer(
            answer=f"echo:{query}",
            confidence=0.8,
            tools_used=[],
            status="ok",
        )

    monkeypatch.setattr(api_server, "run_chat_query", fake_run)
    client = TestClient(api_server.app)

    # Gate off
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 200

    monkeypatch.setenv("CHAT_API_KEY", "secret-demo")
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 401

    r = client.post(
        "/chat",
        json={"query": "hello"},
        headers={"X-API-Key": "secret-demo"},
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "echo:hello"
