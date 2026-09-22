"""FastAPI surface: ``GET /health`` and ``POST /chat``.

Local-only: ``/chat`` is intended for localhost / private demos. Do not
expose it on the public internet without a reverse proxy and authentication.

Auth:
- When ``CHAT_API_KEY`` is set, require matching ``X-API-Key`` or Bearer.
- When ``CHAT_REQUIRE_API_KEY=1`` (Compose default), refuse requests if the key
  env is unset (default-deny for published binds).
- Set ``CHAT_ALLOW_UNAUTHENTICATED=1`` only for trusted local demos without a key.

``/chat`` mirrors the CLI / Streamlit Chat path: CVE→ATT&CK uses the shared
Live ATT&CK pipeline (same function as golden ``mode=agent``;
Chat neighbor ICL uses the CTID corpus unless a CIRCL protocol is passed).
Other queries use a JSON-only LLM chat (no tools).
"""

from __future__ import annotations

import os
import secrets
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.agent.dispatch import run_chat_query
from src.schemas.answer import AgentAnswer

app = FastAPI(
    title="CVE-to-ATT&CK Eval Agent",
    description=(
        "Measured CVE→ATT&CK Live pipeline with JSON Schema outputs. "
        "Of-record path is a shared function (`predict_cve_to_attack`), not LangGraph. "
        "`/chat` is local-demo only: do not expose publicly without auth. "
        "Optional `CHAT_API_KEY` + `X-API-Key` header; Compose sets "
        "`CHAT_REQUIRE_API_KEY=1` by default."
    ),
    version="0.1.0",
)

MAX_QUERY_CHARS = 8_000
MAX_HISTORY_TURNS = 12
ALLOWED_CHAT_MODELS = frozenset(
    {
        "deepseek-v4-flash",
        "gpt-5.6-luna",
        "qwen3.8-27b",
    }
)


class HistoryMessage(BaseModel):
    """One prior Chat turn. Invalid roles are rejected at the API boundary (422)."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)


class ChatRequest(BaseModel):
    """Chat body: user query plus optional multi-turn history."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    model: str | None = Field(default=None, max_length=128)


class HealthResponse(BaseModel):
    """Liveness payload for orchestration / compose healthchecks."""

    status: str
    service: str


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _check_chat_api_key(
    x_api_key: str | None,
    authorization: str | None,
) -> None:
    """Enforce optional / required API key gate with constant-time compare."""
    expected = (os.environ.get("CHAT_API_KEY") or "").strip()
    require = _env_flag("CHAT_REQUIRE_API_KEY")
    allow_unauth = _env_flag("CHAT_ALLOW_UNAUTHENTICATED")

    if not expected:
        if require and not allow_unauth:
            raise HTTPException(
                status_code=503,
                detail=(
                    "CHAT_API_KEY is required (CHAT_REQUIRE_API_KEY=1). "
                    "Set CHAT_API_KEY or CHAT_ALLOW_UNAUTHENTICATED=1 for local demos."
                ),
            )
        return

    provided = (x_api_key or "").strip()
    if not provided and authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        else:
            provided = auth
    # compare_digest is constant-time and returns False when lengths differ.
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail=(
                "Unauthorized: set header X-API-Key (or Authorization: Bearer) "
                "to match CHAT_API_KEY. /chat is local-demo only."
            ),
        )


def _model_allowed(model: str | None) -> bool:
    """Allow empty (env default), catalog models, or the current EXPLABS_MODEL."""
    if not model or not str(model).strip():
        return True
    name = str(model).strip()
    if name in ALLOWED_CHAT_MODELS:
        return True
    env_model = (os.environ.get("EXPLABS_MODEL") or "").strip()
    return bool(env_model) and name == env_model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness (no LLM call)."""
    return HealthResponse(status="ok", service="sec-tool-agent")


@app.post("/chat", response_model=AgentAnswer)
def chat(
    body: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> AgentAnswer:
    """Run the product Chat path once; return a validated ``AgentAnswer``.

    Local-only: meant for localhost demos. Optional / required
    ``CHAT_API_KEY`` gate via ``X-API-Key`` / Bearer. CVE→ATT&CK uses the shared
    Live pipeline. Pass ``history`` for multi-turn follow-ups.

    Raises 401/503 if API key gate fails; 500 if the LLM client cannot start
    (e.g. missing ``EXPLABS_API_KEY``).
    """
    _check_chat_api_key(x_api_key, authorization)
    if not _model_allowed(body.model):
        raise HTTPException(
            status_code=400,
            detail="model is not allowlisted for /chat",
        )

    history = [m.model_dump() for m in body.history]
    try:
        return run_chat_query(
            body.query,
            history=history,
            model=body.model,
            trace=False,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="upstream LLM error",
        ) from None
