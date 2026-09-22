"""Streamlit multipage entry: Golden eval lab (primary) + optional Chat.

Run::

    streamlit run src/ui/app.py

Loads project-root ``.env`` for ExperientialLabs. Golden eval / Live agent is
the main product surface (needs ``EXPLABS_API_KEY``). Chat is a secondary demo
page on the same Live helper.

Install::

    pip install -e ".[ui]"
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402 (after .env load + sys.path setup above)

st.set_page_config(
    page_title="CVE-to-ATT&CK Eval Agent · Eval Lab",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.caption("Eval Lab · Live CVE→ATT&CK pipeline (shared function, not LangGraph)")

_PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

page = st.navigation(
    [
        st.Page(
            str(_PAGES_DIR / "eval_lab.py"),
            title="Golden eval",
            icon=":material/science:",
            default=True,
        ),
        st.Page(
            str(_PAGES_DIR / "chat.py"),
            title="Chat (secondary)",
            icon=":material/chat:",
        ),
    ]
)

page.run()
