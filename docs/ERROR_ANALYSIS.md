# Live error analysis: CIRCL test n=121 (`deepseek-v4-flash`)

Compare two Live settings on the same frozen held-out list. Target gold is never looked up.

| Setting | hit | recall@5 | empty pred | exploit miss* | impact miss* |
|---------|-----|----------|------------|---------------|--------------|
| no exploit-refine | 0.198 | 0.107 | **33.1%** | 0.893 (n=56) | 0.851 (n=74) |
| **+ exploit-refine** | **0.504** | **0.288** | **0.8%** | **0.607** (n=56) | 0.838 (n=74) |

\* Miss rate among cases that have that head in CIRCL gold.

## Failure taxonomy

### 1. Over-gating (dominant without refine)

High-prior IDs (`T1190`, `T1203`, `T1059`, `T1068`, `T1055`) require a ≥8-char verbatim quote in CVE/RAG text. Without the refine pass, the model often emits IDs without copy-paste evidence, so the gate empties the prediction.

Top FNs without refine: T1190×36, T1203×20, T1059×20, T1068×15.

### 2. Refine recovers exploit IDs (but adds FPs)

With refine, empty preds nearly vanish and exploit hit rises 0.11 → 0.39. Top FPs become T1190×25, T1203×23, T1210×19; the model over-fires public-facing / RCE techniques vs CIRCL gold.

### 3. Primary impact remains weak

Impact miss stays ~0.84 even with refine. Refine is exploit-specialized by design; impact needs its own pass or better candidates.

### 4. Not a protocol leak

Losses are gating + head skill, not target-gold contamination (Live never reads CTID/CIRCL labels for the target CVE).

## Example misses (+ refine)

- `circl-CVE-2013-1493`: pred=`T1190, T1499.004` · gold=`T1189, T1203`
- `circl-CVE-2024-24919`: pred=`T1190, T1133, T0888` · gold=`T1003.*, T1005, T1059.004, T1202`
- `circl-CVE-2010-0817`: pred=`[]` · gold=`T1190`

## Next experiments (priority)

1. Soft-gate: keep top-1 high-prior candidate without quote, mark `partial`
2. Quote-repair LLM: force verbatim span attachment before gate
3. Gate-off upper bound on CIRCL (measure raw LLM+RAG ceiling)
4. Impact-refine pass (mirror exploit refine)

## Regenerate

```bash
python -m evals.golden.analyze_results \
  evals/golden/published/circl_test_live_norefine.json \
  --json-out evals/golden/published/circl_test_live_norefine_analysis.json
python -m evals.golden.analyze_results \
  evals/golden/published/circl_test_live_refine.json \
  --json-out evals/golden/published/circl_test_live_refine_analysis.json \
  --md docs/ERROR_ANALYSIS.md
```

See also [`EVAL_REPORT.md`](EVAL_REPORT.md).
