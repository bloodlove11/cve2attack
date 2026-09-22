# CVE-to-ATT&CK evaluation protocol (P0)

This document aligns our golden / Live evaluation practice with the CIRCL
paper's normal practice for CVE-to-ATT&CK eval
([arXiv:2607.25572](https://arxiv.org/abs/2607.25572), *Mapping CVEs to MITRE
ATT&CK Techniques: A Curated Gold-Set Classifier and the Limits of LLM-Assisted
Label Expansion*). Scope here is protocol only: no RoBERTa retrain, no
CAPEC weak-label chain, no LLM gold expansion.

## Gold and Live honesty

| Path | Gold | What we claim |
|------|------|----------------|
| Curated CTID gold | Expert MITRE CTID `attack_to_cve` mappings in `evals/golden/golden_dataset.json` | Labels for scoring only. `deepseek_live_n20` (n=20) is a smoke list, not the headline. |
| CIRCL HF gold | `CIRCL/vulnerability-attack-techniques` via `evals.golden.circl_import` (pinned revision; protocol `circl_test`) | Headline labels (official test, n=121); train-only neighbor / ICL pool |
| Live agent (`mode=agent`) | Target CVE's CTID / CIRCL mapping is never looked up as a feature or few-shot | Honest LLM (+ enrich / tech-doc RAG / neighbor ICL) skill |
| Chat CVE-to-ATT&CK | Same shared helper | Calls `evals.golden.live_attack.predict_cve_to_attack`, one Live path, no dual stack |

Non-LLM predictors (Prior / kNN / Policy / KB / Hybrid answer modes) were
removed. The harness is Live LLM only.

Neighbor ICL may retrieve other labeled CVEs as soft few-shots; the
target CVE is held out of that retrieval (`retrieve_heldout`-style). There
is no default CTID lookup of the target on the Live or Chat CVE-to-ATT&CK path.

Analyst-suggest, not ground truth. Predictions are ranked technique
suggestions for an analyst. CTID / CIRCL gold is the scoring reference; model
output is not a substitute for expert mapping.

Dev vs test. Iterate prompt / evidence-gate / synonym work on CIRCL
train (or a train-only slice). Do not tune on `circl_test` or on the
frozen n=20 list after seeing scores. The published 0.504 refine table did
use CIRCL-test error analysis (post-hoc); further gate work must not. Five
internal gold CVEs also appear in CIRCL test, listed in
[`evals/golden/contamination.json`](../evals/golden/contamination.json)
(`CVE-2018-8111`, `CVE-2019-0926`, `CVE-2019-16784`, `CVE-2020-1495`,
`CVE-2020-4068`; three of those are in `deepseek_live_n20`). Treat those IDs
as lightly seen, not fully unseen.

CIRCL gold-only RoBERTa recall@5 (0.673 ± 0.019 in arXiv:2607.25572) is
literature context for the ambition bar, not something we claim without a
shared protocol.

## Never cherry-pick; fixed seeds; serial Live A/Bs

- Do not drop hard cases, re-order after seeing scores, or tune prompts on
  the reported test list.
- Use fixed seeds when sampling a subset (`--seed N` or `--seeds 42,43,44`).
- For reproducible Live A/Bs use `--jobs 1` (serial case execution). Parallel
  `--jobs > 1` is fine for throughput but is not the default for published
  comparisons.
- Primary headline numbers use CIRCL `--heldout --protocol circl_test`
  (n=121), not a freshly shuffled draw. `--heldout` without `--protocol`
  still defaults to the internal n=20 smoke list for backward compatibility.

## Frozen held-out IDs (smoke)

File: [`evals/golden/heldout_ids.json`](../evals/golden/heldout_ids.json)

- Protocol: `deepseek_live_n20`
- Task: `cve_to_attack`
- n: 20
- Role: smoke / regression list, not the number of record
- Selection: first 20 `cve_to_attack` cases in `golden_dataset.json` order
  (equivalent to `--limit 20` with `seed=null`). Frozen in-repo so DeepSeek
  Live n=20 stays comparable across runs.

```bash
python -m evals.golden.runner --mode agent --task cve_to_attack --heldout --jobs 1
```

### CIRCL test held-out (`circl_test`)

Imported from Hugging Face
[CIRCL/vulnerability-attack-techniques](https://huggingface.co/datasets/CIRCL/vulnerability-attack-techniques)
pinned to revision `319c3e324e7561592cf7e51ec003deb5dd90e61b` (see
`evals/golden/raw/circl/REVISION.json` + adapter `evals.golden.circl_import`).

| Item | Value |
|------|--------|
| Protocol | `circl_test` (separate from `deepseek_live_n20`) |
| Freeze file | [`evals/golden/heldout_circl_test_ids.json`](../evals/golden/heldout_circl_test_ids.json) |
| n | 121 (official HF test split) |
| Gold mapping | flat `techniques` → `expected.attack_techniques` for scoring; `exploitation_techniques` / `primary_impact` only when CIRCL provides them (no invented heads; never `techniques_derived`) |
| Neighbor ICL | train split only; all test CVE ids hard-excluded |

```bash
# CIRCL held-out Live (API cost — operators only; not CI)
# ``--protocol circl_test`` always uses the frozen official test IDs.
# ``--limit N`` takes a prefix of that freeze (does not score CIRCL train).
python -m evals.golden.runner --mode agent --task cve_to_attack \
  --heldout --protocol circl_test --jobs 1

# Smoke: default --heldout is still DeepSeek Live n=20 (not headline)
python -m evals.golden.runner --mode agent --task cve_to_attack --heldout --jobs 1
```

Do not train/tune on the CIRCL test list. Do not use CAPEC /
LLM-expanded / `techniques_derived` labels as gold. Prompt/gate changes
iterate on CIRCL train.

## Dual metrics

We report both our operational suite and CIRCL-style ranking recall:

| Metric | Meaning (how we compute it) |
|--------|------------------------------|
| hit / `hit_rate` | 1.0 if any predicted top-k technique id is in the gold technique list (exact `T####` / `T####.###` after normalize) |
| primary_impact_hit | Same hit rule on the `primary_impact` head vs gold `primary_impact` |
| exploitation_hit | Same hit rule on `exploitation_techniques` vs gold exploitation |
| P@k / `precision_at_k` | (# predicted top-k ids that appear in gold) / (\|top-k predicted\|), variable-length: one correct ID → P@5 = 1.0, empty pred → 0. Not CIRCL ranking P@5. |
| recall@k / `recall_at_k` | (# gold techniques found in predicted top-k) / (\|gold techniques\|), analyst-oriented ranking sense used by CIRCL / VulnTrain |

Implementation: `evals.golden.score.score_attack` /
`score_attack_with_heads` (default k=5). Canonical (parent↔sub) soft-match
variants are also recorded (`canonical_*`) but exact scores remain the lead
metrics.

How recall@k is computed from our predictions vs gold: for each case,
normalize predicted technique ids, take the first *k*, count how many distinct
gold ids appear in that prefix, divide by the number of gold ids for that case.
Dataset `recall_at_k` is the mean of per-case recalls (empty gold → skipped for
head metrics; union gold empty → 0 recall). This matches the usual
multi-label ranking definition CIRCL reports for analyst suggestion quality; it
is not the same as hit rate.

## Multi-seed reporting

Keep the single-seed path unchanged (`--seed N`). For mean±std across seeds:

```bash
# Script (default jobs=1)
python -m evals.golden.multi_seed \
  --mode agent --task cve_to_attack --limit 20 --seeds 42,43,44 --jobs 1

# Same via runner
python -m evals.golden.runner \
  --mode agent --task cve_to_attack --limit 20 --seeds 42,43,44 --jobs 1

# Variance over LLM sampling on the frozen held-out list
python -m evals.golden.multi_seed --mode agent --heldout --seeds 42,43,44 --jobs 1
```

Aggregates mean±std for: `hit_rate`, `primary_impact_hit_rate`,
`exploitation_hit_rate`, `precision_at_k`, `recall_at_k`
(`evals.golden.heldout.aggregate_seed_metrics`). Writes
`evals/golden/results/multi_seed_latest.{json,md}` (gitignored with other
results).

Live multi-seed runs cost API calls, so leave full Live sweeps to operators;
CI covers offline helpers only.

### Caveat: ``temperature=0`` and held-out multi-seed variance

Live chat completions currently use ``temperature=0``. The frozen
``--heldout`` path also fixes the case list (``seed`` only affects subset
sampling when not using held-out). Until a seed reaches LLM sampling or
neighbor-draw stochasticity, `multi_seed --heldout` mean±std will look flat
(near-zero std). That is expected, not a bug in the aggregator.
Wire seed into neighbor retrieval / sampling later if you need meaningful
variance on the frozen list; until then prefer multi-seed on ``--limit``
subsets with ``--seed`` per run, or treat held-out multi-seed as a
reproducibility check.

## Related work (cite; don't overclaim)

- Paper: arXiv:2607.25572 (CIRCL / VulnTrain CVE-to-ATT&CK gold-set classifier).
- Gold dataset: [CIRCL/vulnerability-attack-techniques](https://huggingface.co/datasets/CIRCL/vulnerability-attack-techniques)
  on Hugging Face (CTID-curated; DOI in paper artifacts).
- Code / validators: [vulnerability-lookup/VulnTrain](https://github.com/vulnerability-lookup/VulnTrain).

We cite their curated-gold + recall@k analyst-ranking framing and their
finding that LLM label expansion is unreliable. We do not claim we beat
their reported recall@5 (e.g. gold-only ≈ 0.673 ± 0.019 under their corrected
protocol) without a shared dataset split, label vocabulary, and eval
protocol. Live n=20 is an internal smoke snapshot, not a head-to-head with
their RoBERTa classifier and not of record. Headline Live numbers use
`circl_test` under this protocol.

## ATT&CK catalog (STIX)

Live technique-doc RAG and T-ID resolution use a compact catalog built from
official MITRE ATT&CK STIX 2.1 (enterprise + mobile + ICS), pinned to
v19.2. The old hand-subset (`~90` parents / `398` IDs) dropped gold IDs
such as `T1189` (14 CIRCL test cases) and `T1485`; 38 / 121 CIRCL test
cases had at least one gold ID missing from that subset.

- Current techniques: `evals/golden/raw/attack_techniques_min.json`
- `revoked-by` map: `evals/golden/raw/attack_revoked_by.json`
- Pin metadata: `evals/golden/raw/attack_REVISION.json`

Scoring and Live `normalize_attack_techniques` follow `revoked-by` to the
current ID (`T1478` → `T1632.001`). Format-only `normalize_technique_id` does
not. Deprecated-but-current IDs (e.g. mobile `T1477`) stay in the catalog.
Unnormalizable junk (`T873`) stays invalid; do not invent a T-ID for it.

Do not commit the 50MB+ STIX bundles. Regen from
[attack-stix-data](https://github.com/mitre-attack/attack-stix-data).

Out of scope for this P0 protocol doc: RoBERTa / VulnTrain retrain, CAPEC
derivation labels, LLM gold expansion.

## Cost / token benchmarking (per model)

Every Live eval run records the resolved ``EXPLABS_MODEL`` on the payload and
sums provider ``usage`` (prompt/completion tokens + optional USD ``cost``) into
``metrics.cost``:

| Field | Meaning |
|-------|---------|
| `total_cost_usd` | Sum of `usage.cost` across cases (free-tier often `0.0`) |
| `avg_cost_per_case_usd` | Mean cost on cases that reported usage |
| `cost_per_hit_usd` | `total_cost / (hit_rate * n_scored)` when hits > 0 |
| `total_tokens` / `avg_tokens_per_case` | Token volume (useful when cost is $0) |
| `total_llm_calls` | Count of chat completions (1-2 per case with refine) |

Compare two result JSONs (quality × cost):

```bash
python -m evals.golden.cost_benchmark path/to/run_a.json path/to/run_b.json
```

Older published artifacts without per-case `usage` show `—` for cost columns;
re-run under the current harness to populate them.

## Pointers

- Sources: [`evals/golden/SOURCES.md`](../evals/golden/SOURCES.md)
- CIRCL import: `evals.golden.circl_import` + `evals/golden/heldout_circl_test_ids.json`
- Contamination: [`evals/golden/contamination.json`](../evals/golden/contamination.json)
- Runner: `python -m evals.golden.runner` (`--heldout --protocol circl_test`)
- Cost compare: `python -m evals.golden.cost_benchmark RESULTS.json...`
- Shared Live predict: `evals.golden.live_attack.predict_cve_to_attack` (`protocol=`)
- ATT&CK catalog: `evals.golden.attack_catalog` (STIX v19.2; `revoked-by` on score + Live normalize)
- Regen catalog: `python -m evals.golden.build_attack_catalog --from-dir DIR` (or `--download`)
- Demo overview: [`docs/OVERVIEW.md`](OVERVIEW.md)
