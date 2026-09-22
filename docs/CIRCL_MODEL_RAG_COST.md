# CIRCL model × RAG × cost comparison

Slice: first 20 IDs of frozen `circl_test` (same list every cell). Not of-record n=121.  
Pipeline: Live agent + exploit-refine (2 LLM calls/case).  
RAG on = neighbor ICL + ATT&CK-doc retrieval. RAG off = `--no-rag` (no neighbor few-shot; ATT&CK-doc candidates still in the prompt).  
List $ = ExperientialLabs catalog rates × measured tokens. Observed $ = provider `usage.cost` (promo/free often `$0`).

## Grid (measured 2026-09-08)

| model | RAG | n OK | hit | recall@5 | P@5 | tok/case | list $/case | list ×121 | observed $ (n=20) |
|-------|-----|-----:|----:|---------:|----:|---------:|------------:|----------:|------------------:|
| `deepseek-v4-flash` | on | 20 | **0.450** | 0.241 | 0.300 | 3 538 | **$0.00016** | **$0.019** | $0.00 |
| `deepseek-v4-flash` | off | 20 | 0.300 | 0.198 | 0.217 | 2 746 | $0.00013 | $0.016 | $0.00 |
| `gpt-5.6-luna` | on | 20 | **0.550** | **0.291** | **0.425** | 3 955 | $0.00144 | $0.174 | $0.00 |
| `gpt-5.6-luna` | off | 20 | 0.350 | 0.198 | 0.325 | 3 031 | $0.00117 | $0.142 | $0.00 |
| `qwen3.8-27b` | on | 0 | — | — | — | — | ~$0.0016* | ~$0.19* | blocked |
| `qwen3.8-27b` | off | 0 | — | — | — | — | — | — | blocked |

\*Qwen list-price estimate using DeepSeek RAG token mix (no live Qwen tokens this run). Free daily $5 cap (`free_limit_reached`). Prior incomplete CIRCL (RAG on, n=87): hit 0.471, P@5 0.388; not this n=20 slice.

## Catalog rates (USD / 1M tokens)

| model | input | output |
|-------|------:|-------:|
| `deepseek-v4-flash` | 0.042 | 0.085 |
| `gpt-5.6-luna` | 0.20 | 1.20 |
| `qwen3.8-27b` | 0.32 | 2.40 |

## How to read it

- RAG helps all models that finished: DeepSeek hit +0.15, Luna +0.20 vs no-RAG on the same 20 CVEs.
- Best quality (this slice): `gpt-5.6-luna` + RAG (hit 0.55 / P@5 0.43).
- Best list-price quality/$: `deepseek-v4-flash` + RAG: ~9× cheaper than Luna at catalog rates for ~0.10 less hit.
- Qwen could not be scored here; at list rates it would sit near Luna on $/case if token mix is similar, with a higher output rate ($2.40/1M).

Reproduce:

```bash
python -m evals.golden.model_grid \
  --models deepseek-v4-flash,gpt-5.6-luna,qwen3.8-27b \
  --n 20 --jobs 3 --write-docs
```

Machine-readable: [`evals/golden/published/circl_model_rag_cost_grid.json`](../evals/golden/published/circl_model_rag_cost_grid.json).
