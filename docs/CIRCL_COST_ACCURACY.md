# CIRCL golden eval: cost × accuracy comparison

Protocol: `--heldout --protocol circl_test` (n=121), Live agent.  
Token sample: 3 CIRCL test cases measured on `deepseek-v4-flash` (2026-09-08) with the current harness (`usage` attached).  
List prices: ExperientialLabs catalog `GET /api/models/<slug>` → `input_micro_usd_per_million` / `output_micro_usd_per_million` (host-managed waterfall).  
Observed gateway charge: the same DeepSeek sample returned `usage.cost=0.0` (promo / free allotment). Estimates below are list-price, not necessarily what the org was billed.

## Measured tokens / case (DeepSeek sample)

| Setting | LLM calls | Avg prompt | Avg completion | Avg total |
|---------|----------:|-----------:|---------------:|----------:|
| No exploit-refine | 1 | 1 789 | 97 | 1 886 |
| + exploit-refine | 2 | 3 366 | 222 | 3 588 |

Qwen was free-tier capped (`free_limit_reached`) during re-measure; token estimate reuses the DeepSeek prompt/completion averages (same prompts). Qwen reasoning can inflate completion tokens, so a 5× completion sensitivity column is included.

## Comparative table (CIRCL test)

| Model | Pipeline | n scored | hit | recall@5 | P@5 | Est. $/case (list) | Est. full n=121 | Est. $/hit (list) | Notes |
|-------|----------|--------:|----:|---------:|----:|-------------------:|----------------:|------------------:|-------|
| `deepseek-v4-flash` | Live, no refine | 121 | **0.198** | 0.107 | 0.161 | ~$0.000084 | **~$0.010** | ~$0.00043 | Of-record ablation |
| `deepseek-v4-flash` | Live + exploit-refine | 121 | **0.504** | **0.288** | 0.284 | ~$0.000162 | **~$0.020** | ~$0.00032 | **Of-record headline** |
| `qwen3.8-27b` | Precision-refine (partial) | **87**/121 | 0.471 | 0.255 | **0.388** | ~$0.00161* | ~$0.195* (full) | ~$0.0034* | Free-tier until $5/day; *same-token estimate |
| `qwen3.8-27b` | same, 5× completion sensitivity | 87 | 0.471 | 0.255 | 0.388 | ~$0.00374* | ~$0.453* (full) | ~$0.0079* | If reasoning blows up output tokens |

\*Qwen list price: $0.32 / $2.40 per 1M input/output tokens. DeepSeek Flash list: ~$0.042 / ~$0.085 per 1M. Gateway often reported $0.00 charged on DeepSeek during this session.

### How to read it

- Best quality / $ (list): DeepSeek + exploit-refine: hit 0.50 for ~2¢ per full CIRCL pass at list rates (or ~$0 if still on promo).
- Best precision on incomplete slice: Qwen precision-refine P@5 **0.388** vs DeepSeek of-record **0.284**, but list cost is ~10× higher per case at equal tokens, and the run is only 87/121.
- Do not mix Qwen partial metrics into the of-record DeepSeek table until a full n=121 finishes.

## Catalog rates used

| Model | Input $/1M | Output $/1M | Source |
|-------|-----------:|------------:|--------|
| `deepseek-v4-flash` | 0.042448 | 0.084896 | ExperientialLabs `experiential_cloud` provider |
| `qwen3.8-27b` | 0.320000 | 2.400000 | ExperientialLabs `experiential_cloud` provider |

## Reproduce estimates

```bash
# Measure tokens/cost on a fresh case (after cost-benchmark harness)
EXPLABS_MODEL=deepseek-v4-flash python -m evals.golden.runner \
  --mode agent --task cve_to_attack --protocol circl_test --jobs 1 --limit 3

# Compare result JSONs (cost columns populate on new runs)
python -m evals.golden.cost_benchmark \
  evals/golden/published/circl_test_live_refine.json \
  evals/golden/results/latest.json
```

Machine-readable copy: [`evals/golden/published/circl_cost_accuracy_compare.json`](../evals/golden/published/circl_cost_accuracy_compare.json).
