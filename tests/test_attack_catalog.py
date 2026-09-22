"""STIX-derived ATT&CK catalog coverage + revoked-by resolution."""
from __future__ import annotations

from evals.golden.attack_catalog import (
    catalog_ids,
    load_current_techniques,
    load_revision,
    load_revoked_by,
    resolve_technique_id,
)
from evals.golden.attack_docs import retrieve_techniques
from evals.golden.build_attack_catalog import (
    clean_description,
    merge_domains,
    parse_stix_bundle,
)
from evals.golden.circl_import import load_circl_cases
from evals.golden.live_attack import tech_id_from_item
from evals.golden.score import (
    load_cases,
    normalize_technique_id,
    score_attack,
)


def _mini_bundle() -> dict:
    return {
        "objects": [
            {
                "type": "x-mitre-collection",
                "name": "Test",
                "x_mitre_version": "19.2",
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--aaaa",
                "name": "Drive-by Compromise",
                "description": "Adversaries [drive-by](https://example) (Citation: Foo) a browser.",
                "revoked": False,
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1189"}
                ],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--old",
                "name": "Install Insecure Config",
                "description": "revoked mobile technique",
                "revoked": True,
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1478"}
                ],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--new",
                "name": "Code Signing Policy Modification",
                "description": "replacement",
                "revoked": False,
                "x_mitre_is_subtechnique": True,
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1632.001"}
                ],
            },
            {
                "type": "relationship",
                "relationship_type": "revoked-by",
                "source_ref": "attack-pattern--old",
                "target_ref": "attack-pattern--new",
            },
        ]
    }


def test_parse_stix_bundle_drops_revoked_and_maps() -> None:
    current, edges, meta = parse_stix_bundle(_mini_bundle(), domain="enterprise")
    ids = {r["id"] for r in current}
    assert "T1189" in ids
    assert "T1478" not in ids
    assert "T1632.001" in ids
    assert edges["T1478"] == "T1632.001"
    assert meta["n_revoked"] == 1
    drive = next(r for r in current if r["id"] == "T1189")
    assert "Citation" not in drive["description"]
    assert "https://" not in drive["description"]
    assert "drive-by" in drive["description"]


def test_merge_chains_revoked_by() -> None:
    current, edges, meta = parse_stix_bundle(_mini_bundle(), domain="mobile")
    techs, chained, _metas = merge_domains([(current, edges, meta)])
    assert chained["T1478"] == "T1632.001"
    assert {t["id"] for t in techs} == {"T1189", "T1632.001"}


def test_clean_description_strips_citation_and_truncates() -> None:
    text = clean_description("Hello (Citation: X) [world](http://x)." + (" y" * 400), max_len=40)
    assert "Citation" not in text
    assert "http" not in text
    assert text.endswith("…")
    assert len(text) <= 40


def test_revision_is_pinned_19_2() -> None:
    rev = load_revision()
    assert rev.get("attack_version") == "19.2"
    assert rev.get("n_current", 0) >= 800
    assert "enterprise-attack-19.2.json" in str(rev.get("urls") or {})


def test_catalog_covers_circl_test_gold() -> None:
    ids = catalog_ids()
    assert "T1189" in ids
    assert "T1485" in ids
    assert "T1685" in ids
    assert "T0860" in ids
    assert "T1477" in ids  # deprecated, not revoked
    assert "T1478" not in ids  # revoked → T1632.001
    missing: list[str] = []
    for case in load_circl_cases(split="test"):
        for raw in (case.get("expected") or {}).get("attack_techniques") or []:
            tid = resolve_technique_id(raw)
            assert tid, f"unnormalizable CIRCL gold {raw!r} in {case.get('id')}"
            if tid not in ids:
                missing.append(tid)
    assert not missing, f"CIRCL test gold missing from catalog: {sorted(set(missing))}"


def test_revoked_t1478_resolves_and_scores() -> None:
    assert load_revoked_by().get("T1478") == "T1632.001"
    assert resolve_technique_id("T1478") == "T1632.001"
    assert normalize_technique_id("T1478") == "T1478"  # format-only
    assert tech_id_from_item("T1478") == "T1632.001"
    scores = score_attack(["T1632.001"], ["T1478"], k=5)
    assert scores["hit"] == 1.0
    assert scores["recall_at_k"] == 1.0


def test_t873_stays_unnormalizable() -> None:
    assert normalize_technique_id("T873") is None
    assert resolve_technique_id("T873") is None
    cases = [c for c in load_cases() if c.get("id") == "attack-CVE-2019-10980"]
    assert cases
    junk = (cases[0].get("expected") or {}).get("attack_techniques") or []
    assert "T873" in junk


def test_retrieve_drive_by_finds_t1189() -> None:
    hits = retrieve_techniques("drive-by compromise via malicious website in the browser", k=12)
    ids = {h["id"] for h in hits}
    assert "T1189" in ids


def test_catalog_size_and_domains() -> None:
    techs = load_current_techniques()
    domains = {t["domain"] for t in techs}
    assert domains >= {"enterprise", "mobile", "ics"}
    assert len(techs) == len({t["id"] for t in techs})
    assert all(t["id"] for t in techs)
