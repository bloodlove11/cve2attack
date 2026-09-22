"""Agent loop tests with mocked chat_fn."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.agent.graph import run_agent
from src.agent.tracing import TraceLogger, TRACES_DIR


def _msg(content: str = "", tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


def _tc(name: str, args: dict[str, Any], cid: str = "c1") -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_agent_llm_only_no_tools() -> None:
    def chat_fn(messages, **kwargs):  # noqa: ANN001
        assert "tools" not in kwargs or not kwargs.get("tools")
        return _msg(
            content=json.dumps(
                {
                    "answer": "ACT for KEV-listed CVEs; otherwise weigh CVSS/EPSS.",
                    "confidence": 0.8,
                    "tools_used": [],
                    "citations": [],
                    "status": "ok",
                }
            )
        )

    ans = run_agent(
        "How should I triage a KEV CVE?",
        chat_fn=chat_fn,
        trace=False,
    )
    assert ans.status == "ok"
    assert "ACT" in ans.answer
    assert ans.tools_used == []


def test_agent_includes_history_in_messages() -> None:
    seen: dict[str, Any] = {}

    def chat_fn(messages, **kwargs):  # noqa: ANN001
        seen["messages"] = messages
        return _msg(
            content=json.dumps(
                {
                    "answer": "follow-up ok",
                    "confidence": 0.9,
                    "tools_used": [],
                    "citations": [],
                    "status": "ok",
                }
            )
        )

    ans = run_agent(
        "remind me what I asked",
        history=[
            {"role": "user", "content": "Look up CVE-2019-15243 in CTID."},
            {"role": "assistant", "content": "T1059"},
        ],
        chat_fn=chat_fn,
        trace=False,
    )
    assert ans.status == "ok"
    roles = [m["role"] for m in seen["messages"]]
    assert roles[:4] == ["system", "user", "assistant", "user"]
    assert "CVE-2019-15243" in seen["messages"][1]["content"]


def test_agent_partial_on_bad_json() -> None:
    def chat_fn(messages, **kwargs):  # noqa: ANN001
        return _msg(content="not json at all")

    ans = run_agent("hi", chat_fn=chat_fn, trace=False)
    assert ans.status in {"partial", "error"}


def test_trace_logger_writes(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("src.agent.tracing.TRACES_DIR", tmp_path)
    t = TraceLogger("q", enabled=True)
    t.add_step("user", content="q")
    path = t.finalize({"answer": "a"})
    assert path is not None and path.exists()
    data = json.loads(path.read_text())
    assert data["query"] == "q"
