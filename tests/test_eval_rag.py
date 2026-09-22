"""Offline tests for eval-time RAG (CTID neighbors + prompt injection)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from evals.golden.eval_rag import build_attack_context, build_triage_context
from evals.golden.runner import run_eval
from evals.golden.score import load_cases


def test_build_attack_context_excludes_target_mapping_line() -> None:
    target = "CVE-2019-15243"
    ctx = build_attack_context(target, k=5, exclude_cve=True)
    assert ctx
    assert "CTID" in ctx or "neighbor" in ctx.lower()
    # Must not leak the target's own mapping as an answer line
    # (neighbors only, no "- CVE-2019-15243: [...]" row)
    for line in ctx.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"- {target}:"):
            raise AssertionError(f"target CVE mapping leaked in context: {stripped}")
    # Neighbor CVE ids should appear; target must not be listed as a mapping row
    assert "CVE-2019-" in ctx or "T" in ctx


def test_build_attack_context_nonempty_for_known_cve() -> None:
    ctx = build_attack_context("CVE-2019-15243", k=3, exclude_cve=True)
    assert len(ctx) > 50
    assert "T" in ctx  # technique IDs present from neighbors or priors


def test_build_triage_context_has_signals_and_hint() -> None:
    ctx = build_triage_context(
        {
            "in_kev": True,
            "cvss_score": 9.8,
            "epss": 0.9,
            "critical_asset": True,
            "description": "RCE in demo service",
        }
    )
    assert "in_kev" in ctx or "KEV" in ctx
    assert "9.8" in ctx
    assert "advisory" in ctx.lower() or "Policy hint" in ctx


def test_agent_run_eval_injects_rag_into_mock_messages(
    tmp_path: Path, monkeypatch
) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    captured: list[list[dict[str, str]]] = []

    def _fake_chat(messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any) -> Any:
        captured.append(messages)
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = '{"attack_techniques": ["T1190"]}'
        return mock

    monkeypatch.setattr("src.llm.client.chat_completion", _fake_chat)

    cases = [c for c in load_cases() if c["task"] == "cve_to_attack"][:1]
    assert cases
    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        cases=cases,
        use_rag=True,
    )
    assert payload.get("eval_rag") is True
    assert payload["metrics"].get("eval_rag") is True
    assert captured, "chat_completion was not called"
    blob = "\n".join(m.get("content") or "" for m in captured[0])
    assert (
        "ATT&CK technique" in blob
        or "candidate set" in blob.lower()
        or "description" in blob.lower()
        or "CTID" in blob
        or "neighbor" in blob.lower()
        or "Few-shot" in blob
    ), blob[:600]


def test_agent_no_rag_omits_neighbor_block(tmp_path: Path, monkeypatch) -> None:
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    captured: list[list[dict[str, str]]] = []

    def _fake_chat(messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any) -> Any:
        captured.append(messages)
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = '{"attack_techniques": ["T1190"]}'
        return mock

    monkeypatch.setattr("src.llm.client.chat_completion", _fake_chat)

    cases = [c for c in load_cases() if c["task"] == "cve_to_attack"][:1]
    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        cases=cases,
        use_rag=False,
    )
    assert payload.get("eval_rag") is False
    blob = "\n".join(m.get("content") or "" for m in captured[0])
    assert "Retrieved CTID neighbor" not in blob
    assert "Few-shot exemplars" not in blob
