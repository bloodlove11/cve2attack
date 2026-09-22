"""OpenAI-compatible ExperientialLabs LLM client."""

from __future__ import annotations

import os
import time
from typing import Any

from openai import APIStatusError, BadRequestError, OpenAI, RateLimitError

DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

# Extensible denylist: routes that reject sampling params (400: temperature not supported).
NO_TEMPERATURE_MODELS: frozenset[str] = frozenset(
    {
        "gpt-6-astra",
    }
)

_client: OpenAI | None = None
_client_key: tuple[str, str] | None = None
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 3


def _is_retryable_status(exc: BaseException) -> bool:
    """True for 429 / 5xx provider errors worth a short backoff retry."""
    if isinstance(exc, RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    return status in _RETRYABLE_STATUS


def _omit_temperature_env() -> bool:
    """True when ``EXPLABS_OMIT_TEMPERATURE`` is set to a truthy flag (1/true/yes/on)."""
    raw = os.environ.get("EXPLABS_OMIT_TEMPERATURE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def model_supports_temperature(model: str) -> bool:
    """Return False for denylisted models or when omit-temperature env is set."""
    if _omit_temperature_env():
        return False
    name = (model or "").strip().lower()
    if not name:
        return True
    return name not in NO_TEMPERATURE_MODELS


def _is_temperature_unsupported_error(exc: BaseException) -> bool:
    """Detect API 400 for unsupported ``temperature`` (retry without it once)."""
    if not isinstance(exc, BadRequestError):
        return False
    code = getattr(exc, "code", None)
    param = getattr(exc, "param", None)
    if code == "unsupported_parameter" and param == "temperature":
        return True
    body = getattr(exc, "body", None)
    blob = f"{exc!s} {body!s}".lower()
    if "temperature" not in blob:
        return False
    return any(
        token in blob
        for token in (
            "unsupported_parameter",
            "unsupported parameter",
            "not supported",
            "unknown parameter",
        )
    )


def get_model() -> str:
    """Resolve chat model id from ``EXPLABS_MODEL`` or the package default."""
    return os.environ.get("EXPLABS_MODEL", DEFAULT_MODEL)


def get_base_url() -> str:
    """Resolve OpenAI-compatible base URL from ``EXPLABS_BASE_URL`` or default."""
    return os.environ.get("EXPLABS_BASE_URL", DEFAULT_BASE_URL)


def clear_client_cache() -> None:
    """Drop the cached OpenAI client (tests / key rotation)."""
    global _client, _client_key
    _client = None
    _client_key = None


def get_client() -> OpenAI:
    """Return an ExperientialLabs OpenAI-compatible client.

    Requires EXPLABS_API_KEY in the environment. Never hardcode secrets.
    Rebuilds the client when base URL or API key changes (not a sticky
    ``lru_cache`` of the first key seen).
    """
    global _client, _client_key
    api_key = os.environ.get("EXPLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "EXPLABS_API_KEY is not set. Copy .env.example to .env and set your key."
        )
    base_url = get_base_url()
    key = (base_url, api_key)
    if _client is None or _client_key != key:
        _client = OpenAI(base_url=base_url, api_key=api_key)
        _client_key = key
    return _client


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    temperature: float = 0.0,
    model: str | None = None,
    client: OpenAI | None = None,
) -> Any:
    """Call ``chat.completions.create`` for the agent / eval loops.

    Args:
        messages: OpenAI-style message list.
        tools: Optional function-tool schemas.
        tool_choice: ``auto`` / forced tool / None.
        temperature: Sampling temperature (default 0 for demos/evals).
            Omitted for ``NO_TEMPERATURE_MODELS``, when
            ``EXPLABS_OMIT_TEMPERATURE=1``, or after one automatic retry if the
            API rejects temperature as an unsupported parameter.
        model: Override env model.
        client: Injectable client (tests); else cached ``get_client()``.

    Returns:
        Provider completion object (``choices[0].message`` used by callers).
    """
    c = client or get_client()
    resolved_model = model or get_model()
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
    }
    include_temperature = model_supports_temperature(resolved_model)
    if include_temperature:
        kwargs["temperature"] = temperature
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    attempt = 0
    while True:
        try:
            return c.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if (
                include_temperature
                and _is_temperature_unsupported_error(exc)
                and "temperature" in kwargs
            ):
                # Reshaping the request, not a transient failure: retry right
                # away without spending transient-retry budget. Only reachable
                # once, since include_temperature is cleared here.
                kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
                include_temperature = False
                continue
            raise
        except (RateLimitError, APIStatusError) as exc:
            attempt += 1
            if not _is_retryable_status(exc) or attempt >= _MAX_TRANSIENT_RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
