"""
FallbackRouter — the production-grade wrapper around ExtractionPipeline.

Phase 3 adds two optional injected dependencies:
  - ArbitrationGate: consulted BEFORE attempting compression; if it returns
    NOOP, the router skips the LLM call entirely and returns a passthrough
    with route=CompressionRoute.NOOP.
  - CostLedger: records every outcome (compressed, passthrough, noop) for
    cost accounting and model training.

When neither is supplied (the default), the router behaves exactly as in
Phase 2 — compresses everything, no cost gating.  Phase 2 tests therefore
need no changes.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from compaction_engine.bridge.result import CompressionResult, CompressionRoute
from compaction_engine.config import EngineSettings
from compaction_engine.extraction.pipeline import ExtractionPipeline
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
    RawDialogueTurn,
)
from compaction_engine.utils.exceptions import (
    CompactionEngineError,
    ExtractionFidelityError,
    SchemaValidationError,
)
from compaction_engine.utils.logging_config import get_logger

if TYPE_CHECKING:
    from compaction_engine.cost.arbitration import ArbitrationGate
    from compaction_engine.cost.ledger import CostLedger

logger = get_logger(__name__)

_MAX_BACKOFF_MULTIPLIER: float = 4.0
_BACKOFF_STEP: float = 0.05


def _build_passthrough_state(
    workflow_id: str,
    turns: list[RawDialogueTurn],
) -> CompressedContextState:
    facts: list[ExtractedFact] = []
    for turn in turns:
        facts.append(
            ExtractedFact(
                fact_id=f"passthrough-{turn.turn_index}-{uuid.uuid4().hex[:6]}",
                fact_type=FactType.EXTERNAL_FACT,
                statement=f"[{turn.role}/{turn.agent_name}]: {turn.content[:380]}",
                source_agent=turn.agent_name,
                source_turn_start=turn.turn_index,
                source_turn_end=turn.turn_index,
                confidence=ConfidenceLevel.HIGH,
            )
        )
    raw_text = "\n".join(t.content for t in turns)
    return CompressedContextState(
        workflow_id=workflow_id,
        compaction_version=1,
        source_turn_count=len(turns),
        facts=tuple(facts),
        token_count_estimate=max(1, len(raw_text) // 4),
    )


class FallbackRouter:
    def __init__(
        self,
        pipeline: ExtractionPipeline,
        settings: EngineSettings,
        arbitration_gate: "ArbitrationGate | None" = None,
        cost_ledger: "CostLedger | None" = None,
    ) -> None:
        self._pipeline = pipeline
        self._settings = settings
        self._gate = arbitration_gate
        self._ledger = cost_ledger
        self._threshold_registry: dict[str, float] = {}

    def get_backoff_multiplier(self, workflow_id: str) -> float:
        return self._threshold_registry.get(workflow_id, 1.0)

    def compress(
        self,
        workflow_id: str,
        turns: list[RawDialogueTurn],
    ) -> CompressionResult:
        if not turns:
            raise ValueError(f"compress() called with empty turn list for workflow {workflow_id!r}")

        backoff = self.get_backoff_multiplier(workflow_id)

        # ── Phase 3: Arbitration gate ────────────────────────────────────────
        if self._gate is not None:
            decision = self._gate.evaluate(workflow_id, turns, backoff)
            if not decision.should_compress:
                return self._handle_noop(workflow_id, turns, decision)

        # ── Attempt compression ──────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            run_result = self._pipeline.run(workflow_id, turns)
            latency_ms = (time.monotonic() - t0) * 1000

            result = CompressionResult(
                state=run_result.final_state,
                route=CompressionRoute.COMPRESSED,
                run_result=run_result,
                latency_ms=latency_ms,
                warnings=run_result.warnings,
            )

            logger.info(
                "Compression succeeded",
                extra={
                    "workflow_id": workflow_id,
                    "route": CompressionRoute.COMPRESSED.value,
                    "facts": len(run_result.final_state.facts),
                    "dedup_ratio": round(run_result.dedup_ratio, 3),
                    "latency_ms": round(latency_ms, 2),
                },
            )

            # ── Train predictor + record in ledger ───────────────────────────
            if self._gate is not None:
                # `decision` is in scope from the gate check above.
                self._gate.record_outcome(decision.features, result)  # type: ignore[possibly-undefined]
                self._record_in_ledger(result, decision, "compressed")  # type: ignore[possibly-undefined]

            self.reset_backoff(workflow_id)
            return result

        except (ExtractionFidelityError, SchemaValidationError) as exc:
            return self._handle_fallback(workflow_id, turns, exc, t0, is_expected=True)
        except CompactionEngineError as exc:
            return self._handle_fallback(workflow_id, turns, exc, t0, is_expected=False)
        except Exception as exc:  # noqa: BLE001
            return self._handle_fallback(workflow_id, turns, exc, t0, is_expected=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _handle_noop(
        self,
        workflow_id: str,
        turns: list[RawDialogueTurn],
        decision: "ArbitrationDecision",  # type: ignore[name-defined]
    ) -> CompressionResult:
        passthrough_state = _build_passthrough_state(workflow_id, turns)
        result = CompressionResult(
            state=passthrough_state,
            route=CompressionRoute.NOOP,
            fallback_reason=decision.reason,
            latency_ms=0.0,
            warnings=[f"ArbitrationGate NOOP: {decision.reason}"],
        )
        if self._ledger is not None:
            self._ledger.record_noop(decision.features, decision.reason)
        logger.info(
            "ArbitrationGate NOOP — skipped compression",
            extra={"workflow_id": workflow_id, "reason": decision.reason},
        )
        return result

    def _handle_fallback(
        self,
        workflow_id: str,
        turns: list[RawDialogueTurn],
        exc: Exception,
        t0: float,
        *,
        is_expected: bool,
    ) -> CompressionResult:
        latency_ms = (time.monotonic() - t0) * 1000
        reason = f"{type(exc).__name__}: {exc}"
        self._apply_backoff(workflow_id)

        log = logger.warning if is_expected else logger.error
        log(
            "Compression failed — routing to passthrough",
            extra={
                "workflow_id": workflow_id,
                "fallback_reason": reason,
                "backoff_multiplier": self._threshold_registry.get(workflow_id, 1.0),
                "latency_ms": round(latency_ms, 2),
            },
        )
        passthrough_state = _build_passthrough_state(workflow_id, turns)
        return CompressionResult(
            state=passthrough_state,
            route=CompressionRoute.PASSTHROUGH,
            fallback_reason=reason,
            latency_ms=latency_ms,
            warnings=[f"Fallback triggered: {reason}"],
        )

    def _apply_backoff(self, workflow_id: str) -> None:
        current = self._threshold_registry.get(workflow_id, 1.0)
        new_mult = min(current + _BACKOFF_STEP, _MAX_BACKOFF_MULTIPLIER)
        self._threshold_registry[workflow_id] = new_mult
        logger.info(
            "Backoff applied",
            extra={"workflow_id": workflow_id, "backoff_multiplier": new_mult},
        )

    def reset_backoff(self, workflow_id: str) -> None:
        self._threshold_registry.pop(workflow_id, None)

    def _record_in_ledger(
        self,
        result: CompressionResult,
        decision: "ArbitrationDecision",  # type: ignore[name-defined]
        _route_hint: str,
    ) -> None:
        if self._ledger is None:
            return
        pred = decision.prediction
        self._ledger.record_compression(
            result=result,
            features=decision.features,
            arbitration_reason=decision.reason,
            predictor_model=pred.model_used if pred else "none",
            expected_saving_tokens=pred.expected_saving_tokens if pred else 0,
        )
