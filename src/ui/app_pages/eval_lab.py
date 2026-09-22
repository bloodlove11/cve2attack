"""Golden eval lab: Live LLM agent scoring only."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from evals.golden.runner import LATEST_JSON, LATEST_MD, run_eval
from src.ui.helpers import (
    CUSTOM_MODEL_LABEL,
    api_key_configured,
    model_picker_options,
    resolve_ui_model,
    results_dataframe,
)

st.title("Golden eval")
st.caption(
    "Primary surface. Score Live agent CVE→ATT&CK against internal gold or "
    "CIRCL `circl_test`. LLM-only: no Prior / kNN / Policy / KB / Hybrid modes."
)

key_ok = api_key_configured()

with st.form("golden_eval_form"):
    st.caption(
        "Live agent: LLM inference with NVD/Zenodo enrichment + ATT&CK "
        "technique-doc RAG, neighbor ICL, two-head + exploitation-refine "
        "LLM calls, then candidate-constrain + grounded evidence-gate "
        "(same shared path as CLI golden `mode=agent`). Does not look up "
        "CTID/CIRCL gold for the target CVE."
    )
    protocol_label = st.selectbox(
        "Gold protocol",
        ["Internal golden", "CIRCL test (n=121)"],
        help=(
            "Start with Internal + Limit for Live smokes. "
            "Headline ATT&CK numbers use CIRCL official test "
            "(n=121; always held-out). Internal frozen list is a 20-case smoke set."
        ),
    )
    use_heldout = st.checkbox(
        "Frozen held-out list",
        value=False,
        help=(
            "Use the frozen ID list (n=20 internal or n=121 CIRCL). "
            "CIRCL protocol always uses official test IDs; Limit takes a "
            "prefix of that freeze for cheap smokes."
        ),
    )
    task = st.selectbox(
        "Task",
        ["cve_to_attack", "cve_triage", "all"],
        index=0,
        help="Primary focus is CVE→ATT&CK. Triage uses the Live LLM (no rule policy).",
    )
    limit = st.slider(
        "Limit",
        min_value=5,
        max_value=121,
        value=10,
        step=5,
        help=(
            "Start small for Live (API cost). On CIRCL this is a prefix of "
            "the frozen official test list (n=121), not CIRCL train."
        ),
    )
    seed = st.number_input("Seed", min_value=0, value=0, step=1, help="0 means no shuffle")
    use_seed = st.checkbox("Shuffle with seed before limit", value=False)
    use_eval_rag = st.toggle(
        "Eval RAG",
        value=True,
        help=(
            "NVD + ATT&CK technique docs (+ optional leave-one-out neighbor "
            "few-shot). Never injects the target CVE CTID gold mapping."
        ),
    )
    refine_exploitation = st.toggle(
        "Exploitation refine LLM",
        value=True,
        help=(
            "Second LLM pass for the exploitation head (better exploit metrics, "
            "~2× Live API calls). Turn OFF for fast CIRCL sweeps / ablations."
        ),
    )
    use_prep_cache = st.toggle(
        "Cache Live prep",
        value=True,
        help=(
            "Disk-cache enrich / tech-doc / neighbor prep under "
            "evals/golden/cache/ (LLM calls still live). Speeds CIRCL re-runs."
        ),
    )
    jobs = st.slider(
        "Parallel jobs",
        min_value=1,
        max_value=16,
        value=8,
        step=1,
        help=(
            "Case-level workers for Live. Default 8. "
            "Use 1 for serial repro; lower if you hit API 429s."
        ),
    )
    st.markdown("**Live model**")
    _opts = model_picker_options()
    model_choice = st.selectbox(
        "Model",
        _opts,
        index=0,
        help="Starter list. Applied via EXPLABS_MODEL for Live calls in this run.",
    )
    custom_model = ""
    if model_choice == CUSTOM_MODEL_LABEL:
        from src.llm.client import get_model

        custom_model = st.text_input(
            "Custom model id",
            value=get_model(),
            help="Overrides EXPLABS_MODEL for this eval run.",
        )
    selected_model = resolve_ui_model(model_choice, custom_model)
    submitted = st.form_submit_button(
        "Run eval",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )

mode = "agent"
protocol = (
    "circl_test" if (protocol_label or "").startswith("CIRCL") else None
)
# CIRCL gold is the official test split; never score train as if it were test.
heldout = bool(use_heldout) or protocol == "circl_test"
if heldout and task == "cve_triage":
    st.info("Held-out lists are ATT&CK-only, so the run will use `cve_to_attack`.")
    task = "cve_to_attack"
run_blocked = not key_ok

if not key_ok:
    st.warning(
        "Live agent needs `EXPLABS_API_KEY`. Set the key in `.env` and restart."
    )
else:
    st.caption(
        "Live agent ATT&CK scoring is LLM inference over enrichment + technique docs. "
        "It does not look up CTID/CIRCL gold for the target CVE."
    )

st.caption(
    "Live agent = LLM inference over NVD/Zenodo descriptions and ATT&CK technique "
    "documentation, not CTID answer lookup. Eval RAG (default ON) adds ATT&CK-doc "
    "candidates and optional leave-one-out neighbor few-shot; the target CVE's own "
    "CTID row is never injected."
)

if submitted:
    if run_blocked:
        st.error("Cannot run Live agent without `EXPLABS_API_KEY`.")
    else:
        progress_bar = st.progress(0.0, text="Starting…")
        status = st.empty()

        def _on_progress(done: int, total: int, row: dict[str, Any] | None) -> None:
            frac = done / total if total else 1.0
            progress_bar.progress(min(frac, 1.0), text=f"{done}/{total} cases")
            if row:
                status.caption(
                    f"Last: `{row.get('id')}` · "
                    f"{'skipped' if row.get('skipped') else ('ok' if row.get('correct') else 'miss')}"
                )

        try:
            with st.spinner("Running golden eval…"):
                payload = run_eval(
                    mode=mode,
                    task=task,
                    limit=int(limit),
                    seed=int(seed) if use_seed else None,
                    write_results=True,
                    progress=_on_progress,
                    use_rag=bool(use_eval_rag),
                    heldout=heldout,
                    protocol=protocol,
                    jobs=int(jobs),
                    refine_exploitation=bool(refine_exploitation),
                    use_prep_cache=bool(use_prep_cache),
                    model=selected_model,
                )
            st.session_state["golden_eval_payload"] = payload
            progress_bar.progress(1.0, text="Done")
            status.success(
                "Eval finished. Results written to `evals/golden/results/latest.*`"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Eval failed: {exc}")

payload = st.session_state.get("golden_eval_payload")
if not payload and LATEST_JSON.is_file():
    try:
        payload = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
        st.caption("Showing last saved `evals/golden/results/latest.json`.")
    except (OSError, json.JSONDecodeError):
        payload = None

if not payload:
    st.info("Configure options in the form and click **Run eval**.")
else:
    metrics = payload.get("metrics") or {}
    tri = metrics.get("triage") or {}
    att = metrics.get("attack") or {}
    triage_acc = tri.get("accuracy", metrics.get("triage_accuracy", metrics.get("accuracy")))
    attack_hit = att.get("hit_rate", metrics.get("attack_hit_rate", metrics.get("hit_rate")))
    attack_recall = att.get(
        "recall_at_k", metrics.get("attack_recall_at_k", metrics.get("recall_at_k"))
    )
    macro = metrics.get("macro_score")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("Scored", str(metrics.get("n_scored", 0)))
    with m2:
        with st.container(border=True):
            st.metric(
                "Triage accuracy",
                f"{triage_acc:.1%}" if triage_acc is not None else "—",
            )
    with m3:
        with st.container(border=True):
            st.metric(
                "ATT&CK hit rate",
                f"{attack_hit:.1%}" if attack_hit is not None else "—",
            )
    with m4:
        with st.container(border=True):
            if macro is not None:
                st.metric("Macro score", f"{macro:.1%}")
            elif attack_recall is not None:
                st.metric("ATT&CK recall@k", f"{attack_recall:.1%}")
            elif triage_acc is not None and attack_hit is None:
                st.metric("Accuracy (triage)", f"{triage_acc:.1%}")
            else:
                st.metric("Macro score", "—")

    cost = metrics.get("cost") or {}
    if cost:
        model_label = payload.get("model") or cost.get("model") or "—"
        st.caption(f"Model: `{model_label}`")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.container(border=True):
                total_cost = cost.get("total_cost_usd")
                st.metric(
                    "Total cost (USD)",
                    f"${total_cost:.4f}" if total_cost is not None else "—",
                )
        with c2:
            with st.container(border=True):
                avg_cost = cost.get("avg_cost_per_case_usd")
                st.metric(
                    "Avg $/case",
                    f"${avg_cost:.4f}" if avg_cost is not None else "—",
                )
        with c3:
            with st.container(border=True):
                st.metric("Total tokens", str(cost.get("total_tokens", "—")))
        with c4:
            with st.container(border=True):
                cph = cost.get("cost_per_hit_usd")
                st.metric(
                    "Cost / hit",
                    f"${cph:.4f}" if cph is not None else "—",
                )

    st.dataframe(results_dataframe(payload), width="stretch")

    json_bytes = json.dumps(payload, indent=2).encode("utf-8")
    md_text = ""
    if LATEST_MD.is_file():
        try:
            md_text = LATEST_MD.read_text(encoding="utf-8")
        except OSError:
            md_text = ""
    if not md_text:
        from evals.golden.runner import markdown_report

        md_text = markdown_report(payload)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="golden_eval_latest.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Download Markdown",
            data=md_text.encode("utf-8"),
            file_name="golden_eval_latest.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )
