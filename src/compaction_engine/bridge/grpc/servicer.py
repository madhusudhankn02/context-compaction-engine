"""
gRPC servicer for CompactionService.

This file references `compaction_pb2` and `compaction_pb2_grpc`, the Python
stubs generated from compaction.proto. They do not exist until you run the
compile step in scripts/compile_proto.sh.  To make the module importable
before compilation (e.g. for static analysis or REST-only deployments), the
import is guarded and the servicer raises a clear `ImportError` if the stubs
are missing rather than crashing at the module level.

See scripts/compile_proto.sh for the one-line compile command.
"""

from __future__ import annotations

import json
from typing import Any

from compaction_engine.bridge.container import BridgeContainer
from compaction_engine.schemas.extraction_schema import CompressedContextState, RawDialogueTurn
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Guarded proto stub import ─────────────────────────────────────────────────
try:
    from compaction_engine.bridge.grpc import compaction_pb2, compaction_pb2_grpc  # type: ignore[attr-defined]
    _STUBS_AVAILABLE = True
except ImportError:
    _STUBS_AVAILABLE = False
    compaction_pb2 = None          # type: ignore[assignment]
    compaction_pb2_grpc = None     # type: ignore[assignment]


def _require_stubs() -> None:
    if not _STUBS_AVAILABLE:
        raise ImportError(
            "gRPC stubs not found. Run: bash scripts/compile_proto.sh\n"
            "This generates compaction_pb2.py and compaction_pb2_grpc.py from "
            "src/compaction_engine/bridge/grpc/compaction.proto."
        )


def _proto_turn_to_domain(proto_turn: Any) -> RawDialogueTurn:
    kwargs: dict[str, Any] = {
        "turn_index": proto_turn.turn_index,
        "agent_name": proto_turn.agent_name,
        "role": proto_turn.role,
        "content": proto_turn.content,
    }
    if proto_turn.timestamp_iso:
        from datetime import datetime, timezone
        kwargs["timestamp"] = datetime.fromisoformat(proto_turn.timestamp_iso).replace(
            tzinfo=timezone.utc
        )
    return RawDialogueTurn(**kwargs)


def _state_to_proto(state: CompressedContextState, schema_version: str = "0.1.0") -> Any:
    _require_stubs()
    return compaction_pb2.StatePayload(
        json=state.model_dump_json(),
        schema_version=schema_version,
    )


def _proto_to_state(payload: Any) -> CompressedContextState:
    return CompressedContextState.model_validate_json(payload.json)


class CompactionServicer:
    """
    gRPC servicer. Shares the same BridgeContainer as the REST layer —
    same router instance, same pipeline, same deduplicator. The two transports
    are thin protocol adapters over a single business-logic stack.
    """

    def __init__(self, container: BridgeContainer) -> None:
        _require_stubs()
        self._container = container

    def Compress(self, request: Any, context: Any) -> Any:  # noqa: N802
        _require_stubs()
        turns = [_proto_turn_to_domain(t) for t in request.turns]
        result = self._container.router.compress(request.workflow_id, turns)

        logger.info("gRPC Compress called", extra=result.to_audit_dict())

        return compaction_pb2.CompressResponse(
            workflow_id=result.state.workflow_id,
            route=result.route.value,
            compression_ratio=result.compression_ratio,
            latency_ms=result.latency_ms,
            fact_count=len(result.state.facts),
            source_turn_count=result.state.source_turn_count,
            token_count_estimate=result.state.token_count_estimate,
            warnings=result.warnings,
            fallback_reason=result.fallback_reason or "",
            state=_state_to_proto(result.state),
        )

    def Validate(self, request: Any, context: Any) -> Any:  # noqa: N802
        _require_stubs()
        from compaction_engine.utils.exceptions import IntegrityValidationError
        from compaction_engine.validation.integrity_validator import ContextIntegrityValidator

        gold = _proto_to_state(request.gold_standard)
        compressed = _proto_to_state(request.compressed)
        threshold = request.match_threshold or 0.75

        validator = ContextIntegrityValidator(self._container.settings)
        passed = True
        try:
            report = validator.validate(gold, compressed, match_threshold=threshold)
        except IntegrityValidationError as exc:
            passed = False
            raw = exc.context.get("report", {})
            report_dict: dict[str, Any] = raw if isinstance(raw, dict) else raw.to_dict()
        else:
            report_dict = report.to_dict()

        proto_report = compaction_pb2.FidelityReport(
            workflow_id=report_dict.get("workflow_id", compressed.workflow_id),
            fact_recall=report_dict.get("fact_recall", 0.0),
            fact_precision=report_dict.get("fact_precision", 0.0),
            hallucination_rate=report_dict.get("hallucination_rate", 0.0),
            constraint_recall=report_dict.get("constraint_recall", 0.0),
            open_question_recall=report_dict.get("open_question_recall", 0.0),
            overall_fidelity_score=report_dict.get("overall_fidelity_score", 0.0),
            passed=passed,
            notes=report_dict.get("notes", []),
        )
        return compaction_pb2.ValidateResponse(report=proto_report, passed=passed)

    def Health(self, request: Any, context: Any) -> Any:  # noqa: N802
        _require_stubs()
        s = self._container.settings
        return compaction_pb2.HealthResponse(
            status="ok",
            provider=s.llm_provider.value,
            model=s.llm_model_name,
            version="0.1.0",
        )
