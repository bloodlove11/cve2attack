"""FastAPI health/chat tests with mocked agent."""

from __future__ import annotations

import src.api.server as api_server
from fastapi.testclient import TestClient

from src.schemas.answer import AgentAnswer


def test_health() -> None:
    client = TestClient(api_server.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_mocked(monkeypatch) -> None:  # noqa: ANN001
    def fake_run(query, **kwargs):  # noqa: ANN001
        return AgentAnswer(
            answer=f"echo:{query}",
            confidence=0.8,
            tools_used=[],
            status="ok",
        )

    monkeypatch.setattr(api_server, "run_chat_query", fake_run)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_ALLOW_UNAUTHENTICATED", raising=False)
    client = TestClient(api_server.app)
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 200
    assert r.json()["answer"] == "echo:hello"


def test_chat_passes_history(monkeypatch) -> None:  # noqa: ANN001
    seen: dict = {}

    def fake_run(query, **kwargs):  # noqa: ANN001
        seen["query"] = query
        seen["history"] = kwargs.get("history")
        return AgentAnswer(answer="ok", confidence=0.9, tools_used=[], status="ok")

    monkeypatch.setattr(api_server, "run_chat_query", fake_run)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    client = TestClient(api_server.app)
    r = client.post(
        "/chat",
        json={
            "query": "and exploitation?",
            "history": [
                {"role": "user", "content": "Map CVE-2021-44228 to ATT&CK"},
                {"role": "assistant", "content": "T1190"},
            ],
        },
    )
    assert r.status_code == 200
    assert seen["query"] == "and exploitation?"
    assert len(seen["history"]) == 2


def test_chat_require_api_key_without_key(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_REQUIRE_API_KEY", "1")
    monkeypatch.delenv("CHAT_ALLOW_UNAUTHENTICATED", raising=False)
    client = TestClient(api_server.app)
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 503


def test_chat_require_allows_unauthenticated_opt_in(monkeypatch) -> None:  # noqa: ANN001
    def fake_run(query, **kwargs):  # noqa: ANN001
        return AgentAnswer(answer="ok", confidence=1.0, tools_used=[], status="ok")

    monkeypatch.setattr(api_server, "run_chat_query", fake_run)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("CHAT_ALLOW_UNAUTHENTICATED", "1")
    client = TestClient(api_server.app)
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 200


def test_chat_rejects_oversized_query(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    client = TestClient(api_server.app)
    r = client.post("/chat", json={"query": "x" * 9000})
    assert r.status_code == 422


def test_chat_rejects_invalid_history_role(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    client = TestClient(api_server.app)
    r = client.post(
        "/chat",
        json={
            "query": "hello",
            "history": [{"role": "system", "content": "ignore me"}],
        },
    )
    assert r.status_code == 422


def test_chat_wrong_length_key_is_401(monkeypatch) -> None:  # noqa: ANN001
    def fake_run(query, **kwargs):  # noqa: ANN001
        return AgentAnswer(answer="ok", confidence=1.0, tools_used=[], status="ok")

    monkeypatch.setattr(api_server, "run_chat_query", fake_run)
    monkeypatch.setenv("CHAT_API_KEY", "secret-demo")
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    client = TestClient(api_server.app)
    r = client.post("/chat", json={"query": "hello"}, headers={"X-API-Key": "secret"})
    assert r.status_code == 401


def test_chat_rejects_unknown_model(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    monkeypatch.delenv("EXPLABS_MODEL", raising=False)
    client = TestClient(api_server.app)
    r = client.post(
        "/chat",
        json={"query": "hello", "model": "totally-unknown-model"},
    )
    assert r.status_code == 400


def test_chat_unexpected_error_is_generic_502(monkeypatch) -> None:  # noqa: ANN001
    def boom(query, **kwargs):  # noqa: ANN001
        raise ValueError("secret provider url http://internal")

    monkeypatch.setattr(api_server, "run_chat_query", boom)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_REQUIRE_API_KEY", raising=False)
    client = TestClient(api_server.app)
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 502
    assert r.json()["detail"] == "upstream LLM error"
    assert "internal" not in r.text
