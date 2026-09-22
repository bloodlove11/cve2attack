"""Unit tests for Live LLM accuracy helpers (gate, constrain, exploit refine)."""
from __future__ import annotations

from typing import Any

from evals.golden.live_attack import (
    build_exploitation_prompt,
    build_grounding_text,
    filter_to_candidates,
    id_in_candidate_set,
    postprocess_attack_prediction,
    predict_cve_to_attack,
    quote_is_grounded,
)


def test_quote_is_grounded_requires_verbatim_substring() -> None:
    text = "Remote code execution in a public-facing application via crafted requests."
    assert quote_is_grounded("public-facing application", text)
    assert quote_is_grounded("Public-Facing   Application", text)  # normalized
    assert not quote_is_grounded(".", text)
    assert not quote_is_grounded("remote", text)  # too short
    assert not quote_is_grounded("completely invented paraphrase here", text)


def test_filter_to_candidates_parent_sub() -> None:
    assert filter_to_candidates(
        ["T1190", "T1059.001", "T1005"],
        ["T1190", "T1059"],
    ) == ["T1190", "T1059.001"]
    assert id_in_candidate_set("T1059", {"T1059.001"})
    assert id_in_candidate_set("T1059.001", {"T1059"})
    assert not id_in_candidate_set("T1005", {"T1190", "T1059"})


def test_grounded_evidence_gate_drops_ungrounded_high_prior() -> None:
    grounding = (
        "Remote code execution in a public-facing application "
        "via crafted HTTP requests."
    )
    kept = postprocess_attack_prediction(
        {
            "exploitation_techniques": [
                {"id": "T1190", "evidence": "public-facing application"},
                {"id": "T1059", "evidence": "script interpreter fantasy"},
            ],
            "primary_impact": ["T1005"],
        },
        prefer_ids=["T1190", "T1059", "T1005"],
        grounding_text=grounding,
        constrain_to_candidates=True,
    )
    assert kept["exploitation_techniques"] == ["T1190"]
    assert kept["primary_impact"] == ["T1005"]


def test_constrain_drops_off_candidate_ids() -> None:
    out = postprocess_attack_prediction(
        {
            "exploitation_techniques": ["T1190"],
            "primary_impact": ["T1005", "T1485"],
            "evidence": {
                "T1190": "public-facing application",
            },
        },
        prefer_ids=["T1190", "T1005"],
        grounding_text="Exploit of a public-facing application allows data theft.",
        constrain_to_candidates=True,
        evidence_gate=True,
    )
    assert out["exploitation_techniques"] == ["T1190"]
    assert out["primary_impact"] == ["T1005"]
    assert "T1485" not in out["attack_techniques"]


def test_build_grounding_text_defaults_to_cve_only() -> None:
    g = build_grounding_text(
        enrichment={"description": "Hello CVE", "cwes": ["CWE-94"]},
        rag_context="T1190 docs",
    )
    assert "Hello CVE" in g
    assert "CWE-94" in g
    assert "T1190 docs" not in g
    g_rag = build_grounding_text(
        enrichment={"description": "Hello CVE"},
        rag_context="T1190 docs",
        include_rag=True,
    )
    assert "T1190 docs" in g_rag


def test_build_exploitation_prompt_is_llm_only_head() -> None:
    msgs = build_exploitation_prompt(
        {"cve_id": "CVE-2099-0001"},
        candidate_ids=["T1190", "T1203"],
        enrichment={"description": "RCE in public-facing app"},
        draft_primary_impact=["T1005"],
        draft_exploitation=["T1190"],
    )
    blob = "\n".join(m["content"] for m in msgs)
    assert "exploitation_techniques" in blob
    assert "ONLY emit" in blob
    assert "T1005" in blob
    assert "CVE-2099-0001" in blob
    assert "keep/drop" in blob.lower() or "Draft exploitation" in blob
    assert "CVE description" in blob
    assert "never from att&ck" in blob.lower() or "not att&ck technique docs" in blob.lower()


def test_exploit_refine_veto_filters_draft() -> None:
    from evals.golden.live_attack import apply_exploit_refine_veto

    assert apply_exploit_refine_veto(
        ["T1190", "T1203"], ["T1190", "T1210"], max_n=2
    ) == ["T1190"]
    # Empty draft → refine may fill
    assert apply_exploit_refine_veto([], ["T1190", "T1203", "T1068"], max_n=2) == [
        "T1190",
        "T1203",
    ]


def test_exploit_refine_empty_honors_veto_all() -> None:
    from evals.golden.live_attack import apply_exploit_refine_veto

    assert apply_exploit_refine_veto(["T1190", "T1203"], [], max_n=2) == []
    # Refine did not run / parse miss → keep full draft, do not cap at max_n
    assert apply_exploit_refine_veto(
        ["T1190", "T1203", "T1059"], [], max_n=2, refine_ok=False
    ) == ["T1190", "T1203", "T1059"]


def test_high_prior_gate_applies_to_subtechniques() -> None:
    grounding = "Remote code execution in a public-facing application."
    dropped = postprocess_attack_prediction(
        {
            "exploitation_techniques": [
                {"id": "T1059.004", "evidence": "script interpreter fantasy"},
            ],
            "primary_impact": ["T1005"],
        },
        prefer_ids=["T1059.004", "T1005"],
        grounding_text=grounding,
        constrain_to_candidates=True,
    )
    assert dropped["exploitation_techniques"] == []
    kept = postprocess_attack_prediction(
        {
            "exploitation_techniques": [
                {"id": "T1059.004", "evidence": "public-facing application"},
            ],
            "primary_impact": ["T1005"],
        },
        prefer_ids=["T1059.004", "T1005"],
        grounding_text=grounding,
        constrain_to_candidates=True,
    )
    assert kept["exploitation_techniques"] == ["T1059.004"]


def test_filter_enterprise_drops_ics_mobile() -> None:
    from evals.golden.live_attack import filter_enterprise_technique_ids

    out = filter_enterprise_technique_ids(
        ["T1190", "T0819", "T1404", "T0888", "T1005"]
    )
    assert "T1190" in out
    assert "T1005" in out
    assert "T0819" not in out
    assert "T1404" not in out
    assert "T0888" not in out


def test_predict_runs_exploitation_refine_second_llm_call(monkeypatch) -> None:
    """LLM is obligatory: two-head call + exploitation refine call."""
    calls: list[str] = []

    def _chat(messages: list[dict[str, str]]) -> dict[str, Any]:
        blob = "\n".join(m.get("content") or "" for m in messages)
        if "Refine exploitation" in blob or "specialize in CTID exploitation" in blob:
            calls.append("exploit")
            return {
                "exploitation_techniques": [
                    {
                        "id": "T1190",
                        "evidence": "public-facing application",
                    }
                ],
                "_raw": '{"exploitation_techniques":["T1190"]}',
            }
        calls.append("two_head")
        return {
            "exploitation_techniques": [],
            "primary_impact": ["T1005"],
            "attack_techniques": ["T1005"],
            "_raw": '{"primary_impact":["T1005"]}',
        }

    monkeypatch.setattr(
        "evals.golden.live_attack.get_cve_enrichment",
        lambda _c: {
            "description": (
                "Remote code execution in a public-facing application "
                "via crafted requests."
            ),
            "cwes": ["CWE-94"],
            "cvss": 9.8,
            "source": "test",
        },
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_attack_llm_context",
        lambda *_a, **_k: (
            "Prefer from this candidate set: T1190, T1005\n"
            "T1190 Exploit Public-Facing Application",
            [{"id": "T1190"}, {"id": "T1005"}],
            {
                "description": (
                    "Remote code execution in a public-facing application "
                    "via crafted requests."
                ),
                "cwes": ["CWE-94"],
                "source": "test",
            },
        ),
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.retrieve_labeled_neighbors",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_few_shot_exemplars",
        lambda *_a, **_k: [],
    )

    out = predict_cve_to_attack(
        "CVE-2099-4242",
        use_rag=True,
        llm_chat=_chat,
        refine_exploitation=True,
    )
    assert calls == ["two_head", "exploit"]
    assert out["llm_calls"] == 2
    assert out["mode"] == "agent_llm"
    assert out["exploitation_techniques"] == ["T1190"]
    assert out["primary_impact"] == ["T1005"]
    assert "T1190" in out["attack_techniques"]


def test_predict_refine_cannot_add_ids_beyond_draft(monkeypatch) -> None:
    """When the draft has exploit IDs, refine is a veto and cannot invent T1210."""
    calls: list[str] = []

    def _chat(messages: list[dict[str, str]]) -> dict[str, Any]:
        blob = "\n".join(m.get("content") or "" for m in messages)
        if "Refine exploitation" in blob or "specialize in CTID exploitation" in blob:
            calls.append("exploit")
            return {
                "exploitation_techniques": [
                    {"id": "T1190", "evidence": "public-facing application"},
                    {"id": "T1210", "evidence": "public-facing application"},
                ],
                "_raw": "{}",
            }
        calls.append("two_head")
        return {
            "exploitation_techniques": [
                {"id": "T1190", "evidence": "public-facing application"}
            ],
            "primary_impact": ["T1005"],
            "attack_techniques": ["T1190", "T1005"],
            "_raw": "{}",
        }

    monkeypatch.setattr(
        "evals.golden.live_attack.get_cve_enrichment",
        lambda _c: {
            "description": (
                "Remote code execution in a public-facing application "
                "via crafted requests."
            ),
            "cwes": [],
            "source": "test",
        },
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_attack_llm_context",
        lambda *_a, **_k: (
            "candidates",
            [{"id": "T1190"}, {"id": "T1210"}, {"id": "T1005"}],
            {
                "description": (
                    "Remote code execution in a public-facing application "
                    "via crafted requests."
                ),
                "source": "test",
            },
        ),
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.retrieve_labeled_neighbors",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_few_shot_exemplars",
        lambda *_a, **_k: [],
    )

    out = predict_cve_to_attack(
        "CVE-2099-4243",
        use_rag=True,
        llm_chat=_chat,
        refine_exploitation=True,
        use_prep_cache=False,
    )
    assert calls == ["two_head", "exploit"]
    assert out["exploitation_techniques"] == ["T1190"]
    assert "T1210" not in out["attack_techniques"]


def test_format_few_shot_does_not_park_unlabeled_gold_on_exploit_head() -> None:
    from evals.golden.live_attack import format_few_shot

    blob = format_few_shot(
        [
            {
                "cve_id": "CVE-2010-0001",
                "attack_techniques": ["T1005", "T1485"],
                "techniques": {"exploitation": [], "primary": []},
            }
        ]
    )
    assert "CVE-2010-0001" in blob
    assert "attack_techniques=[T1005, T1485]" in blob
    assert "exploitation=[T1005, T1485]" not in blob


def test_predict_never_looks_up_target_ctid_gold(monkeypatch) -> None:
    def boom(*_a, **_k):  # noqa: ANN001
        raise AssertionError("lookup_ctid_mapping must not run on the Live path")

    monkeypatch.setattr("evals.golden.ctid_index.lookup_ctid_mapping", boom)
    monkeypatch.setattr(
        "evals.golden.live_attack.get_cve_enrichment",
        lambda _c: {
            "description": "Remote code execution in a public-facing application.",
            "cwes": [],
            "source": "test",
        },
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_attack_llm_context",
        lambda *_a, **_k: (
            "docs",
            [{"id": "T1190"}],
            {
                "description": "Remote code execution in a public-facing application.",
                "source": "test",
            },
        ),
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.retrieve_labeled_neighbors",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "evals.golden.live_attack.build_few_shot_exemplars",
        lambda *_a, **_k: [],
    )

    def _chat(messages):  # noqa: ANN001
        return {
            "exploitation_techniques": [
                {"id": "T1190", "evidence": "public-facing application"}
            ],
            "primary_impact": [],
            "_raw": "{}",
        }

    out = predict_cve_to_attack(
        "CVE-2019-15243",
        use_rag=True,
        llm_chat=_chat,
        refine_exploitation=False,
        use_prep_cache=False,
    )
    assert out["mode"] == "agent_llm"
    assert "T1190" in out["attack_techniques"]
