# Walkthrough: Golden eval Live agent

Assumes Quickstart from the [README](../README.md) (`pip install -e ".[ui,dev]"`, `.env` with `EXPLABS_API_KEY`).

## 0-1. Launch Eval Lab

```bash
source .venv/bin/activate
streamlit run src/ui/app.py
```

Open http://localhost:8501, then open Golden eval (default page). Chat is secondary.

## 1-4. Live agent (main path)

1. Task `cve_to_attack`. Gold defaults to Internal (cheap smoke); switch to CIRCL test (n=121) for headline numbers.
2. Model `deepseek-v4-flash`. Needs `EXPLABS_API_KEY`.
3. Keep Eval RAG on. Start with Limit 10 before a full held-out CIRCL run.
4. Published CIRCL numbers: [`docs/EVAL_REPORT.md`](EVAL_REPORT.md).

```bash
python -m evals.golden.runner --mode agent --task cve_to_attack --limit 10 --jobs 8
python -m evals.golden.runner --mode agent --task cve_to_attack \
  --heldout --protocol circl_test --jobs 8
```

Results: `evals/golden/results/latest.md` and (for published runs) `evals/golden/published/`.

Summary: project focus = golden eval Live agent (LLM-only). Chat/API = secondary demos on the same Live helper.

More: [README](../README.md) · eval protocol: [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) · sources: [`evals/golden/SOURCES.md`](../evals/golden/SOURCES.md).
