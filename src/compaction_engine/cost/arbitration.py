"""
Arbitration Gate — the cost-aware gatekeeper before each compression attempt.

Decision flow
─────────────
  1. Fast-reject: turn count < min_turns_to_compress → NOOP
  2. Fast-reject: backoff_multiplier >= max_backoff_to_compress → NOOP
  3. Build pre-compression CostFeatureVector
  4. CostPredictor.predict(features) → PredictionResult
  5. prediction.should_compress → COMPRESS else NOOP

The gate returns an ArbitrationDecision that the FallbackRouter uses:
  - COMPRESS: proceed with the extraction pipeline as normal
  - NOOP:     skip compression; the router builds a passthrough state directly

This decouples the "should we compress?" question (cost model) from the
"how do we compress?" question (extraction pipeline), which is the
architecture the problem statement calls for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from compaction_engine.config import EngineSettings
from compaction_engine.cost.features import CostFeatureVector, FeatureExtractor
from compaction_engine.cost.predictor import CostPredictor, PredictionResult
from compaction_engine.utils.logging_config import get_logger

if TYPE_CHECKING:
    from compaction_engine.schemas.extraction_schema import RawDialogueTurn

logger = get_logger(__name__)


class ArbitrationOutcome(str, Enum):
    COMPRESS = "compress"   # proceed with the extraction pipeline
    NOOP = "noop"           # skip compression; return passthrough directly


@dataclass(frozen=True)
class ArbitrationDecision:
    outcome: ArbitrationOutcome
    features: CostFeatureVector
    prediction: PredictionResult | None  # None on fast-reject (no predictor call)
    reason: str

    @property
    def should_compress(self) -> bool:
        return self.outcome == ArbitrationOutcome.COMPRESS


class ArbitrationGate:
    """
    Stateless w.r.t. workflow data; holds injected dependencies only.

    Usage pattern:
        decision = gate.evaluate(workflow_id, turns, backoff_multiplier)
        if decision.should_compress:
            result = router._attempt_compress(workflow_id, turns)
            gate.record_outcome(decision.features, result)  # trains the predictor
        else:
            result = build_passthrough(workflow_id, turns)
    """

    def __init__(
        self,
        predictor: CostPredictor,
        settings: EngineSettings,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self._predictor = predictor
        self._settings = settings
        self._extractor = extractor or FeatureExtractor()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        workflow_id: str,
        turns: "list[RawDialogueTurn]",
        backoff_multiplier: float = 1.0,
    ) -> ArbitrationDecision:
        """
        Evaluate whether compression should be attempted for this workflow run.
        Always returns a decision — never raises.
        """
        s = self._settings

        # ── Fast-reject 1: too few turns ────────────────────────────────────
        if len(turns) < s.cost_min_turns_to_compress:
            features = self._extractor.from_turns(workflow_id, turns, backoff_multiplier)
            return ArbitrationDecision(
                outcome=ArbitrationOutcome.NOOP,
                features=features,
                prediction=None,
                reason=(
                    f"turn count {len(turns)} < min {s.cost_min_turns_to_compress}"
                ),
            )

        # ── Fast-reject 2: backoff ceiling breached ──────────────────────────
        if backoff_multiplier >= s.cost_max_backoff_to_compress:
            features = self._extractor.from_turns(workflow_id, turns, backoff_multiplier)
            return ArbitrationDecision(
                outcome=ArbitrationOutcome.NOOP,
                features=features,
                prediction=None,
                reason=(
                    f"backoff_multiplier {backoff_multiplier:.2f} >= "
                    f"ceiling {s.cost_max_backoff_to_compress}"
                ),
            )

        # ── Full prediction path ─────────────────────────────────────────────
        features = self._extractor.from_turns(workflow_id, turns, backoff_multiplier)
        try:
            prediction = self._predictor.predict(features)
        except Exception as exc:  # noqa: BLE001
            # Predictor failure must never block the pipeline — default to COMPRESS.
            logger.warning(
                "CostPredictor.predict() failed; defaulting to COMPRESS.",
                extra={"workflow_id": workflow_id, "error": str(exc)},
            )
            from compaction_engine.cost.predictor import PredictionResult
            prediction = PredictionResult(
                expected_saving_tokens=0,
                expected_saving_usd=0.0,
                should_compress=True,
                confidence=0.0,
                model_used="fallback_on_error",
            )

        outcome = (
            ArbitrationOutcome.COMPRESS
            if prediction.should_compress
            else ArbitrationOutcome.NOOP
        )
        reason = (
            f"predictor={prediction.model_used} "
            f"expected_saving={prediction.expected_saving_tokens}tok "
            f"threshold={s.cost_saving_threshold_tokens}tok "
            f"confidence={prediction.confidence:.2f}"
        )

        logger.info(
            "Arbitration decision",
            extra={
                "workflow_id": workflow_id,
                "outcome": outcome.value,
                "reason": reason,
                "model_used": prediction.model_used,
                "expected_saving_tokens": prediction.expected_saving_tokens,
            },
        )
        return ArbitrationDecision(
            outcome=outcome,
            features=features,
            prediction=prediction,
            reason=reason,
        )

    def record_outcome(
        self,
        features: CostFeatureVector,
        result: "CompressionResult",  # type: ignore[name-defined]
    ) -> None:
        """
        Call after a successful compression run to train the predictor
        with the actual saving achieved. Safe to skip on passthrough runs
        (no ground truth to train on).
        """
        from compaction_engine.bridge.result import CompressionRoute

        if result.route != CompressionRoute.COMPRESSED:
            return  # passthrough → no ground truth

        enriched = self._extractor.enrich_from_result(features, result)
        self._predictor.update(enriched, enriched.actual_saving_tokens)
