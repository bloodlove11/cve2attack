# Published CIRCL eval artifacts

Frozen `--heldout --protocol circl_test` (n=121) results for CV / README claims.

| File | Description |
|------|-------------|
| `circl_test_live_norefine.json` | Live agent, 1 LLM call/case, jobs=8 |
| `circl_test_live_refine.json` | Live agent + exploit-refine, 2 LLM calls/case |
| `circl_test_live_refine_precision_partial.json` | Precision-refine: 34/121 on `deepseek-v4-flash` (credits exhausted); not of record |
| `circl_test_live_refine_qwen38_27b_partial.json` | Precision-refine: 87/121 on `qwen3.8-27b` (free daily $5 cap); completed-only metrics; not of record |
| `circl_cost_accuracy_compare.json` | Cost × accuracy comparison (measured tokens + catalog list prices) |
| `circl_model_rag_cost_grid.json` | 3-model × RAG on/off grid (CIRCL prefix n=20; not of-record) |
| `*_analysis.json` | Error taxonomy from `python -m evals.golden.analyze_results` |

Of-record model: `deepseek-v4-flash` (artifacts do not pin `model` / `pipeline_version`; they predate HEAD `pipeline_version=3`). Headline 0.504 is a post-hoc CIRCL-test ablation of that older pipeline; do not claim HEAD reproduces it. Narrative: [`docs/EVAL_REPORT.md`](../../docs/EVAL_REPORT.md) · Cost table: [`docs/CIRCL_COST_ACCURACY.md`](../../docs/CIRCL_COST_ACCURACY.md) · Model × RAG: [`docs/CIRCL_MODEL_RAG_COST.md`](../../docs/CIRCL_MODEL_RAG_COST.md).
