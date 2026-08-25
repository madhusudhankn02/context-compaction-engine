"""
Cost Predictor — predicts expected token savings for a compression attempt.

Decision boundary
─────────────────
  compress  if  predicted_saving_tokens > settings.cost_saving_threshold_tokens

Model lifecycle
───────────────
  Cold start  (< min_samples):   rule-based heuristic fires.
  Warm        (>= min_samples):  GradientBoostingRegressor trained online.
  Every time update() adds a sample that crosses a multiple of RETRAIN_EVERY,
  the model is refitted on all accumulated samples.

Sklearn is an optional dependency (it ships with sentence-transformers, so it
will typically be present, but the heuristic fallback keeps the engine
functional if it is ever unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compaction_engine.config import EngineSettings
from compaction_engine.cost.features import CostFeatureVector
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)

RETRAIN_EVERY = 5   # refit the model after every N new samples


@dataclass(frozen=True)
class PredictionResult:
    """Output of CostPredictor.predict()."""
    expected_saving_tokens: int
    expected_saving_usd: float
    should_compress: bool
    confidence: float       # 0.0 = pure heuristic, 1.0 = model with ≥100 samples
    model_used: str         # "heuristic" | "gradient_boosting"


class CostPredictor:
    """
    Gradient-boosting cost predictor with heuristic cold-start fallback.

    Thread-safety: NOT thread-safe. Each worker process owns one instance
    (same as FallbackRouter._threshold_registry).  Cross-process model
    sharing is a Phase 4 concern (serialise via joblib, store in shared
    storage).
    """

    def __init__(self, settings: EngineSettings) -> None:
        self._settings = settings
        self._X: list[list[float]] = []   # feature rows (pre-compression)
        self._y: list[float] = []          # actual saving tokens (post-compression)
        self._model: Any = None
        self._model_fitted = False
        self._sklearn_available: bool | None = None   # lazy probe

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, features: CostFeatureVector) -> PredictionResult:
        """
        Predict expected token savings for a compression attempt.
        Uses the trained model when available; falls back to heuristic.
        """
        if self._model_fitted:
            return self._predict_model(features)
        return self._predict_heuristic(features)

    def update(self, features: CostFeatureVector, actual_saving_tokens: int) -> None:
        """
        Online learning: add one training sample. Refits when crossing
        min_samples or every RETRAIN_EVERY samples thereafter.
        """
        self._X.append(features.to_sklearn_row())
        self._y.append(float(actual_saving_tokens))

        n = len(self._X)
        min_s = self._settings.cost_predictor_min_samples
        if n >= min_s and (n == min_s or (n - min_s) % RETRAIN_EVERY == 0):
            self._fit()

    @property
    def sample_count(self) -> int:
        return len(self._X)

    @property
    def is_model_active(self) -> bool:
        return self._model_fitted

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit(self) -> None:
        if not self._sklearn_available:
            self._sklearn_available = self._probe_sklearn()
        if not self._sklearn_available:
            logger.warning(
                "scikit-learn unavailable — CostPredictor staying in heuristic mode.",
                extra={"sample_count": len(self._X)},
            )
            return

        try:
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.9,
                random_state=42,
            )
            model.fit(self._X, self._y)
            self._model = model
            self._model_fitted = True
            logger.info(
                "CostPredictor refitted",
                extra={"sample_count": len(self._X)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CostPredictor refit failed — staying in heuristic mode.",
                extra={"error": str(exc)},
            )

    @staticmethod
    def _probe_sklearn() -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def _predict_model(self, features: CostFeatureVector) -> PredictionResult:
        row = [features.to_sklearn_row()]
        raw_pred = float(self._model.predict(row)[0])
        saving = max(0, int(raw_pred))
        threshold = self._settings.cost_saving_threshold_tokens
        # Confidence grows with sample count, capping at 1.0 with 100 samples.
        confidence = min(1.0, len(self._X) / 100.0)
        usd = self._tokens_to_usd(saving)
        return PredictionResult(
            expected_saving_tokens=saving,
            expected_saving_usd=round(usd, 6),
            should_compress=saving > threshold,
            confidence=confidence,
            model_used="gradient_boosting",
        )

    def _predict_heuristic(self, features: CostFeatureVector) -> PredictionResult:
        """
        Conservative rule-based heuristic for cold-start period.

        Compress when:
          - Enough turns to make compression worthwhile
          - Dialogue is long enough to have meaningful redundancy
          - Workflow hasn't failed too many times (backoff not excessive)
        """
        s = self._settings
        should = (
            features.source_turn_count >= s.cost_min_turns_to_compress
            and features.raw_token_estimate > s.cost_saving_threshold_tokens * 2
            and features.backoff_multiplier < s.cost_max_backoff_to_compress
        )
        # Estimate: assume 55% compression ratio (conservative)
        estimated_saving = int(features.raw_token_estimate * 0.55) if should else 0
        usd = self._tokens_to_usd(estimated_saving)
        return PredictionResult(
            expected_saving_tokens=estimated_saving,
            expected_saving_usd=round(usd, 6),
            should_compress=should,
            confidence=0.0,
            model_used="heuristic",
        )

    def _tokens_to_usd(self, tokens: int) -> float:
        price = self._settings.cost_token_price_per_million_usd
        return tokens * price / 1_000_000
