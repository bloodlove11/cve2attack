# Eval report: CIRCL Live agent

Protocol: `--heldout --protocol circl_test` (n=121 official CIRCL test).  
Model: `deepseek-v4-flash` (ExperientialLabs). Target CTID/CIRCL gold is never looked up on Live.  
Harness: Live LLM only (non-LLM Prior / kNN / Policy / KB / Hybrid predictors removed).  
Status: post-hoc on the frozen test list (prompt/gate/refine iteration used CIRCL-test errors, contrary to the "iterate on train" rule below). Artifacts predate `pipeline_version=3` (HEAD strips ICS/mobile candidates and CVE-only-grounds quotes). Re-running on this tree is a new experiment.

Artifacts: `evals/golden/published/`.

## Headline table (CIRCL test n=121)

| Mode | hit | recall@5 | P@5 | exploit hit* | impact hit* | Wall (jobs=8) |
|------|-----|----------|-----|--------------|-------------|----------------|
| Live agent (no exploit-refine) | 0.198 | 0.107 | 0.161 | 0.107 | 0.149 | ~682s |
| Live agent (**+ exploit-refine**) | **0.504** | **0.288** | **0.284** | **0.393** | 0.162 | ~735s |

\* Head metrics only when CIRCL provides head gold (many rows are flat `techniques` only).

### Verdict

Exploit-refine is a large positive ablation (+0.31 hit, +0.29 exploit hit) vs no-refine on the same frozen CIRCL test list. Headline Live with refine: hit 0.504, recall@5 0.288. This is an honest frozen-list measurement of an older pipeline, not a pre-registered unseen-test claim, and not HEAD `pipeline_version=3`.

Cost × accuracy: list-price estimates for a full CIRCL pass are ~$0.01 (no-refine) vs ~$0.02 (refine) for `deepseek-v4-flash` at catalog rates, see [`CIRCL_COST_ACCURACY.md`](CIRCL_COST_ACCURACY.md). Model × RAG grid (n=20 slice): [`CIRCL_MODEL_RAG_COST.md`](CIRCL_MODEL_RAG_COST.md).

## Precision-refine attempt (partial, not of record)

Tried CVE-only evidence grounding + exploit-refine veto + Enterprise-only candidates on the same CIRCL test list.

### DeepSeek slice (credits exhausted early)

`deepseek-v4-flash` full n=121 did not finish (`insufficient_credits`). On 34 completed cases only (same IDs vs prior published refine):

| | hit | recall@5 | P@5 | empty% | pred len | notes |
|--|-----|----------|-----|--------|----------|--------|
| Published refine (same 34) | 0.559 | 0.319 | 0.338 | 0.0 | 2.32 | T1210×6, T0819×4 FPs |
| Precision-refine (34 OK) | 0.412 | 0.216 | **0.397** | 11.8 | 1.00 | T1210/ICS FPs gone; shorter lists |

Artifact: `evals/golden/published/circl_test_live_refine_precision_partial.json`.

### Qwen3.8-27b slice (free daily $5 cap)

Switched to free-tier `qwen3.8-27b` (same precision-refine pipeline). Completed 87/121 before `free_limit_reached` (resets 00:00 UTC). Metrics below are completed-only, not full CIRCL, not DeepSeek of-record:

| | hit | recall@5 | P@5 | exploit hit* | impact hit* | empty% | mean pred len |
|--|-----|----------|-----|--------------|-------------|---------|---------------|
| qwen3.8-27b precision-refine (87 OK) | 0.471 | 0.255 | **0.388** | 0.541 | 0.192 | 0.0 | 1.61 |

Read: vs DeepSeek of-record refine (hit 0.504 / P@5 0.284), this incomplete Qwen run shows higher P@5 and shorter lists, with hit slightly lower on the completed slice, directionally consistent with the precision goal, but do not replace the of-record table until a full n=121 finishes. Artifact: `evals/golden/published/circl_test_live_refine_qwen38_27b_partial.json`.

## Ablation: exploit-refine

| Setting | hit | recall@5 | exploit hit | LLM calls/case |
|---------|-----|----------|-------------|----------------|
| `--no-exploit-refine` | 0.198 | 0.107 | 0.107 | 1 |
| refine on (default) | 0.504 | 0.288 | 0.393 | 2 |

Takeaway: the second LLM pass is high-leverage for exploitation; keep it for of-record Live runs. Use `--no-exploit-refine` only for fast plumbing sweeps.

## Error taxonomy

See [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) (no-refine detail) and refine analysis JSON under `evals/golden/published/`.

1. Over-gating of high-prior techniques: empty preds ~33% without refine; top gold FNs are `T1190` / `T1203` / `T1059` / `T1068` (the gate's high-prior set).
2. Impact head still weak: refine helps exploit more than primary_impact (0.16).
3. Not contamination: Live never reads target gold; losses are skill + gating.

Next experiments (priority): finish remaining 34 CIRCL cases after Qwen free-limit reset (00:00 UTC) or with funded DeepSeek credits; if full-n recall stays down vs 0.504, soften CVE-only gate (high-prior CVE-only, others may use RAG) while keeping Enterprise filter + T1210 boost removal.

## How to reproduce

```bash
pip install -e ".[dev,ui,circl]"

python -m evals.golden.runner --mode agent --task cve_to_attack \
  --heldout --protocol circl_test --jobs 1 --no-exploit-refine
python -m evals.golden.runner --mode agent --task cve_to_attack \
  --heldout --protocol circl_test --jobs 1

python -m evals.golden.analyze_results \
  evals/golden/published/circl_test_live_refine.json \
  --md docs/ERROR_ANALYSIS.md
```

## CV one-liner

> Built an evaluated CVE-to-ATT&CK Live agent with frozen CIRCL held-out protocol. Published Live (`deepseek-v4-flash`) hit 0.50 with exploit-refine on CIRCL test n=121 is a post-hoc older-pipeline ablation; it does not claim SOTA or HEAD reproducibility.

Ambition bar (literature context, not claimed): CIRCL gold-only RoBERTa recall@5 ≈ 0.673 ± 0.019 ([arXiv:2607.25572](https://arxiv.org/abs/2607.25572)).
