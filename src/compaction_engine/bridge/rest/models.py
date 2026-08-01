"""
REST layer Data Transfer Objects.

Alias strategy — why we do NOT use Field(alias=...) or Field(validation_alias=...)
on direct FastAPI request body models:
─────────────────────────────────────────────────────────────────────────────────
In Pydantic 2.13.4, FastAPI's OpenAPI schema generation calls
TypeAdapter(RequestModel).json_schema() for every top-level request body model.
This triggers a second schema-generation pass where each field's FieldInfo is
processed as a free-standing type-annotation object rather than as a bound
model-field definition.  In that context, validation_alias/alias on a FieldInfo
has no representable meaning in the JSON Schema spec, so Pydantic fires
UnsupportedFieldAttributeWarning — regardless of whether the alias was placed
inside Annotated or in the Field() default assignment.

serialization_alias is NOT affected because JSON Schema generation only cares
about the validation/parsing direction; serialization metadata is ignored.

Nested models (e.g. RawTurnDTO) are also unaffected: FastAPI resolves them via
a cached $ref and never triggers the TypeAdapter re-pass on their fields.

FIX for request models:
  Strip every alias attribute from FieldInfo completely.
  Use model_validator(mode='before') to remap camelCase JSON keys to snake_case
  Python field names before Pydantic validates them.  This keeps the parsing
  behaviour identical (callers send {"workflowId": ...}) while keeping FieldInfo
  objects alias-free and therefore silent during schema generation.

FIX for response models:
  serialization_alias= is safe — use it as before.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from compaction_engine.schemas.extraction_schema import RawDialogueTurn


# ── Request models ────────────────────────────────────────────────────────────

class RawTurnDTO(BaseModel):
    """
    One raw dialogue turn received from an agent or orchestrator.
    Nested model — FastAPI resolves it via $ref, no TypeAdapter re-pass,
    so Annotated[T, Field(alias=...)] is safe here.
    """
    model_config = ConfigDict(populate_by_name=True)

    turn_index: Annotated[int, Field(ge=0,                        alias="turnIndex")]
    agent_name: Annotated[str, Field(min_length=1, max_length=120, alias="agentName")]
    role:        str = Field(min_length=1, max_length=40)
    content:     str = Field(min_length=1)
    timestamp:   datetime | None = None

    def to_domain(self) -> RawDialogueTurn:
        kwargs: dict[str, Any] = {
            "turn_index": self.turn_index,
            "agent_name": self.agent_name,
            "role":       self.role,
            "content":    self.content,
        }
        if self.timestamp is not None:
            kwargs["timestamp"] = self.timestamp
        return RawDialogueTurn(**kwargs)


class CompressRequest(BaseModel):
    """
    Direct FastAPI request body model — NO alias attributes on FieldInfo.
    camelCase JSON keys are remapped to snake_case by the model_validator
    before Pydantic validation runs, so FieldInfo objects carry zero alias
    metadata and the TypeAdapter schema-generation pass stays warning-free.
    """
    model_config = ConfigDict(populate_by_name=True)

    workflow_id: str             = Field(min_length=1)
    turns:       list[RawTurnDTO] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _remap_camel_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        mapping = {"workflowId": "workflow_id"}
        return {mapping.get(k, k): v for k, v in data.items()}


class ValidateRequest(BaseModel):
    """
    Direct FastAPI request body model — same alias-free strategy as CompressRequest.
    """
    model_config = ConfigDict(populate_by_name=True)

    gold_standard:   dict[str, Any]
    compressed:      dict[str, Any]
    match_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _remap_camel_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        mapping = {
            "goldStandard":   "gold_standard",
            "matchThreshold": "match_threshold",
        }
        return {mapping.get(k, k): v for k, v in data.items()}


# ── Response models ───────────────────────────────────────────────────────────
# These are never passed through TypeAdapter for request-body schema generation.
# serialization_alias= on Field() is safe — no warning.

class CompressResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_id:          str            = Field(serialization_alias="workflowId")
    route:                str
    compression_ratio:    float          = Field(serialization_alias="compressionRatio")
    latency_ms:           float          = Field(serialization_alias="latencyMs")
    fact_count:           int            = Field(serialization_alias="factCount")
    source_turn_count:    int            = Field(serialization_alias="sourceTurnCount")
    token_count_estimate: int            = Field(serialization_alias="tokenCountEstimate")
    warnings:             list[str]
    fallback_reason:      str | None     = Field(default=None,
                                                 serialization_alias="fallbackReason")
    state:                dict[str, Any]

    @classmethod
    def from_result(cls, result: "CompressionResult") -> "CompressResponse":  # type: ignore[name-defined]
        from compaction_engine.bridge.result import CompressionResult
        assert isinstance(result, CompressionResult)
        return cls(
            workflow_id          = result.state.workflow_id,
            route                = result.route.value,
            compression_ratio    = round(result.compression_ratio, 4),
            latency_ms           = round(result.latency_ms, 2),
            fact_count           = len(result.state.facts),
            source_turn_count    = result.state.source_turn_count,
            token_count_estimate = result.state.token_count_estimate,
            warnings             = result.warnings,
            fallback_reason      = result.fallback_reason,
            state                = result.state.model_dump(mode="json"),
        )


class FidelityReportDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_id:            str       = Field(serialization_alias="workflowId")
    fact_recall:            float     = Field(serialization_alias="factRecall")
    fact_precision:         float     = Field(serialization_alias="factPrecision")
    hallucination_rate:     float     = Field(serialization_alias="hallucinationRate")
    constraint_recall:      float     = Field(serialization_alias="constraintRecall")
    open_question_recall:   float     = Field(serialization_alias="openQuestionRecall")
    overall_fidelity_score: float     = Field(serialization_alias="overallFidelityScore")
    passed:                 bool
    notes:                  list[str]


class ValidateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    report: FidelityReportDTO
    passed: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status:   str
    provider: str
    model:    str
    version:  str = "0.1.0"


class ErrorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    error:       str
    error_type:  str       = Field(serialization_alias="errorType")
    workflow_id: str | None = Field(default=None, serialization_alias="workflowId")
