"""Pydantic / JSON Schema validated agent answer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """Optional source citation (RAG or tool)."""

    source: str = Field(..., description="Document id, enrichment source, or tool name")
    snippet: str | None = Field(None, description="Short supporting excerpt")


class AgentAnswer(BaseModel):
    """Structured final answer produced by the agent."""

    answer: str = Field(..., min_length=1, description="Final natural-language answer")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Self-reported confidence in [0, 1]"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Names of tools invoked during the run"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Optional supporting citations"
    )
    status: Literal["ok", "partial", "error"] = Field(
        "ok", description="Outcome status for downstream consumers"
    )

    @field_validator("tools_used")
    @classmethod
    def _normalize_tools(cls, v: list[str]) -> list[str]:
        return sorted({t.strip() for t in v if t and t.strip()})


AGENT_ANSWER_JSON_SCHEMA: dict[str, Any] = AgentAnswer.model_json_schema()
