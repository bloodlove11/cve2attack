"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from src.schemas.answer import AGENT_ANSWER_JSON_SCHEMA, AgentAnswer


def test_valid_answer():
    a = AgentAnswer(
        answer="hello",
        confidence=0.5,
        tools_used=["live_attack_pipeline", "live_attack_pipeline", " other "],
        status="ok",
    )
    assert a.tools_used == ["live_attack_pipeline", "other"]
    assert AGENT_ANSWER_JSON_SCHEMA["title"] == "AgentAnswer"


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        AgentAnswer(answer="x", confidence=1.5)


def test_empty_answer_rejected():
    with pytest.raises(ValidationError):
        AgentAnswer(answer="", confidence=0.1)
