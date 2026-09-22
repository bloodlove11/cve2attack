"""Chat analyst page: conversational security tool agent."""

from __future__ import annotations

import streamlit as st

from src.ui.helpers import (
    CUSTOM_MODEL_LABEL,
    api_key_configured,
    format_trace_summary,
    model_picker_options,
    render_answer_card,
    render_tool_timeline,
    resolve_ui_model,
    run_query,
    session_history_from_messages,
)

st.title("Chat (secondary)")
st.caption(
    "Optional demo page. The **primary** workflow is **Golden eval → Live agent**. "
    "CVE→ATT&CK here still calls the same `predict_cve_to_attack` helper as "
    "`mode=agent`. Neighbor ICL uses the CTID corpus (not CIRCL train), so Chat "
    "numbers will not match `--protocol circl_test`. Follow-ups can omit the "
    "CVE id when ATT&CK language remains; they re-run the mapper rather than "
    "answering from the last prediction. Other questions are JSON-only LLM chat "
    "(no tools)."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace_info" not in st.session_state:
    st.session_state.last_trace_info = {}

key_ok = api_key_configured()

with st.sidebar:
    st.subheader("Model")
    _opts = model_picker_options()
    model_choice = st.selectbox(
        "Live model",
        _opts,
        index=0,
        help=(
            "Starter list only. Default deepseek-v4-flash keeps temperature=0; "
            "gpt-6-astra omits temperature automatically."
        ),
        key="chat_model_choice",
    )
    custom_model = ""
    if model_choice == CUSTOM_MODEL_LABEL:
        from src.llm.client import get_model

        custom_model = st.text_input(
            "Custom model id",
            value=get_model(),
            help="Sent as EXPLABS_MODEL for this run.",
            key="chat_custom_model",
        )
    selected_model = resolve_ui_model(model_choice, custom_model)
    st.caption(f"Active: `{selected_model}`")
    st.divider()
    if key_ok:
        st.badge("API key configured", icon=":material/check_circle:", color="green")
    else:
        st.badge("API key missing", icon=":material/error:", color="red")
    st.caption("Env: `EXPLABS_API_KEY` · `EXPLABS_BASE_URL` · `EXPLABS_MODEL`")
    st.divider()
    last = st.session_state.get("last_trace_info") or {}
    st.subheader("Last trace")
    if last:
        st.markdown(format_trace_summary(last))
        with st.expander("Trace steps (raw)", expanded=False):
            st.json(
                {
                    "trace_id": last.get("trace_id"),
                    "path": last.get("path"),
                    "meta": last.get("meta"),
                    "steps": last.get("steps"),
                }
            )
    else:
        st.caption("Run a query to write a trace under `traces/`.")
    st.divider()
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.session_state.last_trace_info = {}
        st.session_state.pop("chat_suggestions", None)
        st.rerun()

if not key_ok:
    st.warning(
        "EXPLABS_API_KEY is missing. Copy `.env.example` to `.env`, set your "
        "ExperientialLabs key, then restart Streamlit. Chat and Golden eval "
        "both require a Live API key (LLM-only)."
    )

SUGGESTIONS = {
    ":material/bug_report: Triage a CVE": (
        "Triage CVE-2024-1234 for severity and recommended action."
    ),
    ":material/hub: Live ATT&CK map": (
        "Map CVE-2021-44228 to MITRE ATT&CK techniques "
        "(Live helper: same function as golden mode=agent; CTID ICL, not CIRCL)."
    ),
}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            answer = msg.get("answer")
            if answer is not None:
                render_answer_card(answer)
            else:
                st.markdown(msg.get("content", ""))
            steps = msg.get("steps") or []
            if steps:
                render_tool_timeline(steps, live=False)

prompt: str | None = None
if not st.session_state.messages:
    selected = st.pills(
        "Try asking",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
        disabled=not key_ok,
        key="chat_suggestions",
    )
    if selected:
        prompt = SUGGESTIONS[selected]

chat_value = st.chat_input(
    "Ask the security tool agent…",
    disabled=not key_ok,
)
if chat_value:
    prompt = chat_value

if prompt and key_ok:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status(":shimmer[Running agent]", type="compact") as status:
                with st.status("Calling Live / CTID agent", type="step"):
                    st.write(
                        "Same path as CLI / FastAPI `/chat` "
                        "(Live ATT&CK when mapping a CVE; history for follow-ups)."
                    )
                    history = session_history_from_messages(
                        st.session_state.messages[:-1]
                    )
                    answer, trace_info = run_query(
                        prompt,
                        model=selected_model,
                        history=history,
                    )
                status.update(label="Agent finished", state="complete")

            st.session_state.last_trace_info = trace_info
            steps = trace_info.get("steps") or []
            render_answer_card(answer)
            if steps:
                render_tool_timeline(steps, live=False)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "answer": answer,
                    "steps": steps,
                    "trace_info": trace_info,
                }
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent run failed: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {exc}"}
            )
