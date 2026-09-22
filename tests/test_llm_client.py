"""LLM client config tests (no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from openai import BadRequestError

from src.llm import client as llm_client


def _fake_response_ok() -> Any:
    class _Msg:
        content = "ok"
        tool_calls = None

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return _Resp()


def _make_fake_client(create_side_effect: Any) -> Any:
    completions = MagicMock()
    completions.create.side_effect = create_side_effect
    client = MagicMock()
    client.chat.completions = completions
    return client


def _bad_request_temperature() -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
    err = BadRequestError(
        "Error code: 400 - temperature is not supported",
        response=response,
        body={
            "error": {
                "message": "temperature is not supported",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_parameter",
            }
        },
    )
    # openai may set code/param from body depending on version; force for tests
    object.__setattr__(err, "code", "unsupported_parameter")
    object.__setattr__(err, "param", "temperature")
    return err


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXPLABS_MODEL", raising=False)
    monkeypatch.delenv("EXPLABS_BASE_URL", raising=False)
    monkeypatch.delenv("EXPLABS_OMIT_TEMPERATURE", raising=False)
    assert llm_client.get_model() == "deepseek-v4-flash"
    assert "experientiallabs.ai" in llm_client.get_base_url()


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPLABS_MODEL", "custom-model")
    monkeypatch.setenv("EXPLABS_BASE_URL", "http://localhost:9999/v1")
    assert llm_client.get_model() == "custom-model"
    assert llm_client.get_base_url() == "http://localhost:9999/v1"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    llm_client.clear_client_cache()
    with pytest.raises(RuntimeError, match="EXPLABS_API_KEY"):
        llm_client.get_client()


def test_get_client_rebuilds_when_key_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_client.clear_client_cache()
    monkeypatch.setenv("EXPLABS_API_KEY", "key-one")
    monkeypatch.setenv("EXPLABS_BASE_URL", "http://localhost:9999/v1")
    c1 = llm_client.get_client()
    monkeypatch.setenv("EXPLABS_API_KEY", "key-two")
    c2 = llm_client.get_client()
    assert c1 is not c2
    llm_client.clear_client_cache()


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-6-astra", False),
        ("GPT-6-ASTRA", False),
        ("deepseek-v4-flash", True),
        ("glm-5.3-flash", True),
        ("custom-model", True),
    ],
)
def test_model_supports_temperature(
    model: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXPLABS_OMIT_TEMPERATURE", raising=False)
    assert llm_client.model_supports_temperature(model) is expected


def test_omit_temperature_env_forces_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPLABS_OMIT_TEMPERATURE", "1")
    assert llm_client.model_supports_temperature("deepseek-v4-flash") is False


def test_chat_completion_omits_temperature_for_astra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXPLABS_OMIT_TEMPERATURE", raising=False)
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_response_ok()

    client = _make_fake_client(create)
    llm_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="gpt-6-astra",
        temperature=0.0,
        client=client,
    )
    assert len(calls) == 1
    assert "temperature" not in calls[0]
    assert calls[0]["model"] == "gpt-6-astra"


def test_chat_completion_includes_temperature_for_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXPLABS_OMIT_TEMPERATURE", raising=False)
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_response_ok()

    client = _make_fake_client(create)
    llm_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="deepseek-v4-flash",
        temperature=0.0,
        client=client,
    )
    assert len(calls) == 1
    assert calls[0].get("temperature") == 0.0
    assert calls[0]["model"] == "deepseek-v4-flash"


def test_chat_completion_omits_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPLABS_OMIT_TEMPERATURE", "true")
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_response_ok()

    client = _make_fake_client(create)
    llm_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="deepseek-v4-flash",
        temperature=0.0,
        client=client,
    )
    assert "temperature" not in calls[0]


def test_chat_completion_retries_without_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXPLABS_OMIT_TEMPERATURE", raising=False)
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise _bad_request_temperature()
        return _fake_response_ok()

    client = _make_fake_client(create)
    resp = llm_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="some-new-model",
        temperature=0.0,
        client=client,
    )
    assert resp.choices[0].message.content == "ok"
    assert len(calls) == 2
    assert calls[0].get("temperature") == 0.0
    assert "temperature" not in calls[1]


def test_chat_completion_does_not_retry_other_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXPLABS_OMIT_TEMPERATURE", raising=False)
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
    other = BadRequestError(
        "Error code: 400 - model not found",
        response=response,
        body={"error": {"message": "model not found", "code": "model_not_found"}},
    )
    object.__setattr__(other, "code", "model_not_found")
    object.__setattr__(other, "param", None)

    client = _make_fake_client(other)
    with pytest.raises(BadRequestError, match="model not found"):
        llm_client.chat_completion(
            [{"role": "user", "content": "hi"}],
            model="deepseek-v4-flash",
            temperature=0.0,
            client=client,
        )
    assert client.chat.completions.create.call_count == 1


def test_chat_completion_retries_429(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import RateLimitError

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    calls: list[int] = []

    def create(**kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            response = httpx.Response(
                429, request=httpx.Request("POST", "https://example.test")
            )
            err = RateLimitError(
                "Error code: 429 - rate limited",
                response=response,
                body={"error": {"message": "rate limited"}},
            )
            object.__setattr__(err, "status_code", 429)
            raise err
        return _fake_response_ok()

    client = _make_fake_client(create)
    resp = llm_client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="deepseek-v4-flash",
        client=client,
    )
    assert resp.choices[0].message.content == "ok"
    assert len(calls) == 2
