# Golden dataset sources

We do not scrape random web pages for labels. Ground truth comes from public, citable corpora:

| Dataset | What it gives us | URL |
|---|---|---|
| MITRE CTID `attack_to_cve` | Expert CVE-to-ATT&CK technique mappings | https://github.com/center-for-threat-informed-defense/attack_to_cve |
| CIRCL `vulnerability-attack-techniques` | Curated CVE-to-ATT&CK gold (CTID + KEV), official train/test splits | https://huggingface.co/datasets/CIRCL/vulnerability-attack-techniques (pin: `evals/golden/raw/circl/REVISION.json`) |
| Zenodo CVE Triage Stability Benchmark | 320 CVEs with policy labels `ACT` / `ATTEND` / `TRACK` from CVSS+EPSS+KEV+asset | https://doi.org/10.5281/zenodo.20665128 |
| CISA KEV | Known-exploited membership used as an input feature | https://www.cisa.gov/known-exploited-vulnerabilities-catalog |

Raw downloads live under `evals/golden/raw/` (regenerate with the build script). Curated eval cases: `golden_dataset.json`.

## Scoring
- cve_triage: exact match on `gold_label`
- cve_to_attack: recall@k over predicted technique IDs vs CTID set; bonus if any `primary_impact` hits

## Live-agent enrichment (not gold labels)
- Zenodo descriptions / CVSS reused offline for CVE enrichment prompts
- NVD API 2.0 (optional, cached in `raw/nvd_cache.json`) when Zenodo misses
- ATT&CK technique docs (`raw/attack_techniques_min.json`) for lexical technique RAG, not CTID CVE-to-technique gold for the target. Regenerated from official STIX (enterprise + mobile + ICS, pinned v19.2) via `python -m evals.golden.build_attack_catalog`. Revoked IDs map through `raw/attack_revoked_by.json` (e.g. `T1478` → `T1632.001`). See `raw/attack_REVISION.json`.

## CIRCL gold import

- Adapter: `evals.golden.circl_import` (HF revision pinned; local parquet under `raw/circl/`).
- Held-out protocol: `circl_test` → `heldout_circl_test_ids.json` (official test IDs only).
- Schema: score on flat `techniques`; pass through CIRCL `exploitation_techniques` /
  `primary_impact` when present; do not invent heads; never use
  `techniques_derived` / CAPEC / LLM-expanded labels as gold.
- Neighbor ICL under `circl_test`: train pool only, hard-exclude test CVE ids.
- `deepseek_live_n20` is a smoke protocol, not the headline. Overlap with CIRCL test: `evals/golden/contamination.json`.
- How to run: see [`docs/EVAL_PROTOCOL.md`](../../docs/EVAL_PROTOCOL.md).
