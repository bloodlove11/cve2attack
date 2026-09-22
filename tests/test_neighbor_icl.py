"""Offline tests for Live-safe similar-labeled CVE neighbor ICL."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from evals.golden.eval_rag import (
    neighbor_technique_ids,
    neighbors_as_few_shot,
    query_text_from_enrichment,
    retrieve_labeled_neighbors,
)
from evals.golden.live_attack import format_few_shot, postprocess_attack_prediction
from evals.golden.runner import run_eval
from evals.golden.score import load_cases


def test_neighbor_retrieval_excludes_target_cve() -> None:
    target = "CVE-2019-15243"
    enrich = {
        "description": (
            "Remote code execution in a public-facing web application via "
            "crafted HTTP requests. Command injection and privilege escalation."
        ),
        "cwes": ["CWE-78", "CWE-94"],
        "cvss": 9.8,
        "source": "test",
    }
    neighbors = retrieve_labeled_neighbors(
        target, enrichment=enrich, k=3, use_embeddings=False
    )
    assert neighbors, "expected BM25/lexical neighbors for RCE-style query"
    assert len(neighbors) <= 3
    assert all(n.get("cve_id") != target for n in neighbors)
    assert all(str(n.get("cve_id") or "").upper() != target for n in neighbors)
    # Must carry two-head CTID technique splits for few-shot formatting
    assert all(n.get("techniques") for n in neighbors)
    assert all(n.get("attack_techniques") for n in neighbors)


def test_embeddings_are_opt_in(monkeypatch) -> None:
    from evals.golden import eval_rag as rag

    def boom(*_a, **_k):  # noqa: ANN001
        raise AssertionError("sentence-transformers retrieval must be opt-in")

    monkeypatch.setattr(rag, "_embed_retrieve", boom)
    neighbors = retrieve_labeled_neighbors(
        "CVE-2019-15243",
        enrichment={
            "description": "Remote code execution in a public-facing web application.",
            "cwes": ["CWE-94"],
            "source": "test",
        },
        k=3,
        use_embeddings=None,
    )
    assert neighbors, "BM25 path should still return neighbors"


def test_neighbor_few_shot_two_head_format() -> None:
    target = "CVE-2019-15243"
    enrich = {
        "description": (
            "SQL injection in internet-facing application allowing remote attackers."
        ),
        "cwes": ["CWE-89"],
        "source": "test",
    }
    neighbors = retrieve_labeled_neighbors(
        target, enrichment=enrich, k=3, use_embeddings=False
    )
    exemplars = neighbors_as_few_shot(neighbors)
    blob = format_few_shot(exemplars)
    assert "Few-shot exemplars" in blob
    assert "exploitation=" in blob
    assert "primary_impact=" in blob
    assert target not in blob  # target id must not appear as an exemplar row
    assert "Soft bias" in blob or "do not copy" in blob.lower()


def test_candidate_constrain_drops_off_list_ids(
    tmp_path: Path, monkeypatch
) -> None:
    """Live postprocess keeps LLM IDs only when they match the candidate set."""
    import evals.golden.runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "LATEST_JSON", tmp_path / "latest.json")
    monkeypatch.setattr(runner_mod, "LATEST_MD", tmp_path / "latest.md")

    captured: list[list[dict[str, str]]] = []
    desc = (
        "Remote code execution in a public-facing application "
        "via crafted requests."
    )

    def _fake_chat(messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any) -> Any:
        captured.append(messages)
        mock = MagicMock()
        mock.choices = [MagicMock()]
        # T9999 is off-catalog / off-list; T1190 should survive via candidates + quote
        mock.choices[0].message.content = (
            '{"exploitation_techniques": '
            '[{"id": "T1190", "evidence": "public-facing application"}, '
            '{"id": "T9999", "evidence": "public-facing application"}], '
            '"primary_impact": ["T1005"], '
            '"attack_techniques": ["T1190", "T9999", "T1005"]}'
        )
        return mock

    monkeypatch.setattr("src.llm.client.chat_completion", _fake_chat)

    cases = [c for c in load_cases() if c["task"] == "cve_to_attack"][:1]
    assert cases
    cid = str((cases[0].get("input") or {}).get("cve_id") or "")

    enrich = {
        "cve_id": cid,
        "description": desc,
        "cwes": ["CWE-94"],
        "cvss": 9.8,
        "source": "test",
    }
    monkeypatch.setattr(
        "evals.golden.live_attack.get_cve_enrichment",
        lambda _c: dict(enrich),
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_attack_llm_context",
        lambda *_a, **_k: (
            "Prefer from this candidate set: T1190, T1005\n"
            "T1190 Exploit Public-Facing Application",
            [{"id": "T1190"}, {"id": "T1005"}],
            dict(enrich),
        ),
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.retrieve_labeled_neighbors",
        lambda *_a, **_k: [
            {
                "cve_id": "CVE-2099-1111",
                "attack_techniques": ["T1190", "T1005"],
                "techniques": {
                    "exploitation": ["T1190"],
                    "primary": ["T1005"],
                },
            }
        ],
    )

    payload = run_eval(
        mode="agent",
        task="cve_to_attack",
        write_results=False,
        cases=cases,
        use_rag=True,
        use_prep_cache=False,
    )
    assert captured
    blob = "\n".join(m.get("content") or "" for m in captured[0])
    assert "Few-shot exemplars" in blob
    assert "exploitation=" in blob
    assert "candidate set" in blob.lower()
    assert "ONLY emit" in blob or "only emit" in blob.lower()
    # Two-head + exploitation refine → at least one LLM call (typically 2)
    assert len(captured) >= 1
    pred = payload["results"][0]["prediction"]
    techs = set(pred.get("attack_techniques") or [])
    assert "T9999" not in techs
    assert "T1190" in techs or "T1005" in techs
    # Target CVE must not be listed as a few-shot exemplar row
    for line in blob.splitlines():
        if line.strip().startswith(f"- {cid}:"):
            raise AssertionError(f"target CVE leaked as few-shot: {line}")


def test_evidence_gate_still_drops_ungated_fps() -> None:
    out = postprocess_attack_prediction(
        {
            "exploitation_techniques": ["T1190", "T1059"],
            "primary_impact": ["T1005"],
        },
        prefer_ids=["T1190", "T1059", "T1005", "T1499.004"],
    )
    assert out["exploitation_techniques"] == []
    assert out["primary_impact"] == ["T1005"]
    assert out["attack_techniques"] == ["T1005"]


def test_query_text_uses_description_and_cwes() -> None:
    q = query_text_from_enrichment(
        "CVE-2099-0001",
        {
            "description": "Buffer overflow allows arbitrary code execution",
            "cwes": ["CWE-119", "CWE-787"],
        },
    )
    assert "Buffer overflow" in q
    assert "CWE-119" in q
    assert "CVE-2099-0001" in q


def test_neighbor_technique_ids_unique_ordered() -> None:
    ids = neighbor_technique_ids(
        [
            {
                "cve_id": "CVE-1",
                "techniques": {
                    "exploitation": ["T1190"],
                    "primary": ["T1005", "T1190"],
                },
                "attack_techniques": ["T1190", "T1005", "T1499.004"],
            }
        ]
    )
    assert ids[0] == "T1190"
    assert ids.count("T1190") == 1
    assert "T1005" in ids
    assert "T1499.004" in ids


def test_neighbor_search_docs_omit_gold_technique_ids() -> None:
    """Neighbor BM25/embed docs must not contain gold T-IDs (cleaner ICL ablation)."""
    from evals.golden.eval_rag import _neighbor_doc_text, clear_neighbor_corpus_cache

    clear_neighbor_corpus_cache()
    text = _neighbor_doc_text(
        {
            "cve_id": "CVE-2099-0001",
            "attack_techniques": ["T1190", "T1059.001"],
            "techniques": {"exploitation": ["T1190"], "primary": ["T1059.001"]},
            "phase": "exploitation",
        },
        {
            "description": "Remote code execution via crafted HTTP request",
            "cwes": ["CWE-94"],
        },
    )
    assert "Remote code execution" in text
    assert "CWE-94" in text
    assert "T1190" not in text
    assert "T1059" not in text
