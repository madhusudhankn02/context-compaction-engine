"""
FastAPI route handlers for the compaction bridge REST API.

Serialization note: response models use Field(serialization_alias="camelCase").
FastAPI only honours serialization_alias when response_model_by_alias=True is
set on the route decorator — that flag is set on every endpoint below.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from compaction_engine.bridge.container import BridgeContainer
from compaction_engine.bridge.rest.models import (
    CompressRequest,
    CompressResponse,
    ErrorResponse,
    FidelityReportDTO,
    HealthResponse,
    ValidateRequest,
    ValidateResponse,
)
from compaction_engine.schemas.extraction_schema import CompressedContextState
from compaction_engine.utils.exceptions import IntegrityValidationError
from compaction_engine.utils.logging_config import get_logger
from compaction_engine.validation.integrity_validator import ContextIntegrityValidator

logger = get_logger(__name__)
router = APIRouter()


def _get_container(request: Request) -> BridgeContainer:
    return request.app.state.container


ContainerDep = Annotated[BridgeContainer, Depends(_get_container)]


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_by_alias=True,
    tags=["ops"],
)
async def health(container: ContainerDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=container.settings.llm_provider.value,
        model=container.settings.llm_model_name,
    )


@router.post(
    "/v1/compress",
    response_model=CompressResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    tags=["compaction"],
)
async def compress(body: CompressRequest, container: ContainerDep) -> CompressResponse:
    turns = [t.to_domain() for t in body.turns]
    try:
        result = await asyncio.to_thread(
            container.router.compress, body.workflow_id, turns
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error=str(exc),
                error_type="ValidationError",
                workflow_id=body.workflow_id,
            ).model_dump(by_alias=True),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /v1/compress",
                         extra={"workflow_id": body.workflow_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="Internal server error — check server logs.",
                error_type=type(exc).__name__,
                workflow_id=body.workflow_id,
            ).model_dump(by_alias=True),
        ) from exc

    logger.info("POST /v1/compress", extra=result.to_audit_dict())
    return CompressResponse.from_result(result)


@router.post(
    "/v1/validate",
    response_model=ValidateResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    tags=["compaction"],
)
async def validate(body: ValidateRequest, container: ContainerDep) -> ValidateResponse:
    try:
        gold       = CompressedContextState.model_validate(body.gold_standard)
        compressed = CompressedContextState.model_validate(body.compressed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error=f"State payload failed schema validation: {exc}",
                error_type="SchemaValidationError",
            ).model_dump(by_alias=True),
        ) from exc

    validator = ContextIntegrityValidator(container.settings)
    passed = True
    try:
        report = await asyncio.to_thread(
            validator.validate, gold, compressed,
            match_threshold=body.match_threshold,
        )
    except IntegrityValidationError as exc:
        passed = False
        raw = exc.context.get("report")
        if raw is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error="Validator raised without a report object.",
                    error_type="IntegrityValidationError",
                ).model_dump(by_alias=True),
            ) from exc
        report_dict: dict = raw if isinstance(raw, dict) else raw.to_dict()
    else:
        report_dict = report.to_dict()

    return ValidateResponse(
        report=FidelityReportDTO(
            workflow_id            = report_dict["workflow_id"],
            fact_recall            = report_dict["fact_recall"],
            fact_precision         = report_dict["fact_precision"],
            hallucination_rate     = report_dict["hallucination_rate"],
            constraint_recall      = report_dict["constraint_recall"],
            open_question_recall   = report_dict["open_question_recall"],
            overall_fidelity_score = report_dict["overall_fidelity_score"],
            passed                 = passed,
            notes                  = report_dict["notes"],
        ),
        passed=passed,
    )
