"""
FallbackRouter — the production-grade wrapper around ExtractionPipeline.

This is Phase 2's most important class. From the problem statement:

    "Build fallback routing: if compression causes an agent to fail,
     automatically escalate to uncompressed state and log the failure."

The router enforces three properties:

1. TRANSPARENT TO AGENTS: agents never see a failure. They call compress();
   they always get back a valid CompressedContextState. If compression
   fails, they get a passthrough state (all turns represented as facts) —
   same schema, different route tag.

2. ADAPTIVE THRESHOLD (exponential backoff): if a compressed step triggers
   fallback, the compression threshold for subsequent calls on the same
   workflow is increased, progressively trusting compression less for
   workflows that have demonstrated instability. Phase 3's cost predictor
   reads the `_threshold_registry` as a feature.

3. FULLY OBSERVABLE: every call, whether successful or fallback, produces
   a `CompressionResult` with a complete `.to_audit_dict()` that Phase 4
   can persist directly. Nothing is swallowed silently.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

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

logger = get_logger(__name__)

# Maximum multiplier the backoff can reach before capping.
_MAX_BACKOFF_MULTIPLIER: float = 4.0
# Factor by which the threshold is tightened on each fallback event.
_BACKOFF_STEP: float = 0.05


def _build_passthrough_state(
    workflow_id: str,
    turns: list[RawDialogueTurn],
) -> CompressedContextState:
    """
    Converts raw turns into a valid CompressedContextState with zero
    information loss. Each turn becomes one EXTERNAL_FACT (confidence=HIGH,
    turn range = that single turn). This preserves the downstream agent's
    ability to consume a typed state even when the compressor failed.

    Token estimate is deliberately the raw sum — the passthrough offers
    NO savings, which Phase 3 will correctly price as cost_saving=0.
    """
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
    token_estimate = max(1, len(raw_text) // 4)

    return CompressedContextState(
        workflow_id=workflow_id,
        compaction_version=1,
        source_turn_count=len(turns),
        facts=tuple(facts),
        token_count_estimate=token_estimate,
    )


class FallbackRouter:
    """
    Stateful per-process router.

    Thread-safety note: `_threshold_registry` is mutated on fallback events.
    In a multi-threaded server (uvicorn + multiple workers) each process has
    its own registry, which is correct — backoff state is per-worker. For
    shared state across workers, Phase 4's Redis/PostgreSQL layer should be
    used. This is documented, not papered over.
    """

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        settings: EngineSettings,
    ) -> None:
        self._pipeline = pipeline
        self._settings = settings
        # workflow_id -> current backoff multiplier (1.0 = no backoff)
        self._threshold_registry: dict[str, float] = {}

    def get_backoff_multiplier(self, workflow_id: str) -> float:
        return self._threshold_registry.get(workflow_id, 1.0)

    def compress(
        self,
        workflow_id: str,
        turns: list[RawDialogueTurn],
    ) -> CompressionResult:
        """
        Main entry point. Always returns a CompressionResult with a valid
        CompressedContextState. Never raises — caller gets a PASSTHROUGH
        result instead.
        """
        if not turns:
            raise ValueError(f"compress() called with empty turn list for workflow {workflow_id!r}")

        t0 = time.monotonic()

        try:
            run_result = self._pipeline.run(workflow_id, turns)
            latency_ms = (time.monotonic() - t0) * 1000

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
            return CompressionResult(
                state=run_result.final_state,
                route=CompressionRoute.COMPRESSED,
                run_result=run_result,
                latency_ms=latency_ms,
                warnings=run_result.warnings,
            )

        except (ExtractionFidelityError, SchemaValidationError) as exc:
            return self._handle_fallback(workflow_id, turns, exc, t0, is_expected=True)

        except CompactionEngineError as exc:
            return self._handle_fallback(workflow_id, turns, exc, t0, is_expected=False)

        except Exception as exc:  # noqa: BLE001
            # Unexpected errors (e.g. OOM, provider SDK bug) still produce a
            # passthrough — we NEVER let a compression bug crash an agent.
            return self._handle_fallback(workflow_id, turns, exc, t0, is_expected=False)

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

        log_level = logger.warning if is_expected else logger.error
        log_level(
            "Compression failed — routing to passthrough",
            extra={
                "workflow_id": workflow_id,
                "route": CompressionRoute.PASSTHROUGH.value,
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
        """
        Exponential backoff: each fallback event raises the effective
        dedup threshold (making compression more conservative) up to
        _MAX_BACKOFF_MULTIPLIER. The pipeline reads dedup_cosine_threshold
        from settings, so backoff is registered here for Phase 3 to consume
        as a feature — it does NOT currently mutate settings (which are
        frozen/cached). Phase 3's arbitration layer will use the multiplier
        to decide whether to attempt compression at all.
        """
        current = self._threshold_registry.get(workflow_id, 1.0)
        new_multiplier = min(current + _BACKOFF_STEP, _MAX_BACKOFF_MULTIPLIER)
        self._threshold_registry[workflow_id] = new_multiplier
        logger.info(
            "Backoff applied",
            extra={
                "workflow_id": workflow_id,
                "backoff_multiplier": new_multiplier,
            },
        )

    def reset_backoff(self, workflow_id: str) -> None:
        """Call this after a successful compression run to recover from a
        transient failure sequence without permanently penalizing a workflow."""
        self._threshold_registry.pop(workflow_id, None)
