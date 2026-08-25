"""
Cost Ledger — in-memory accumulator for workflow cost statistics.

Every completed compression attempt (compressed, passthrough, or noop) is
recorded here with its token counts and dollar estimates.  The ledger
provides the aggregate stats the Phase 4 observability dashboard will display
and the Phase 5 benchmark suite will use for cost-per-workflow comparisons.

Thread-safety: NOT thread-safe (same assumption as FallbackRouter).
Phase 4 will replace the in-memory store with PostgreSQL writes inside a
proper async session, at which point this class becomes a thin adapter.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from compaction_engine.bridge.result import CompressionResult, CompressionRoute
from compaction_engine.config import EngineSettings
from compaction_engine.cost.features import CostFeatureVector
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LedgerEntry:
    """One row in the cost ledger."""
    workflow_id: str
    recorded_at: datetime
    route: str                      # CompressionRoute.value
    source_turn_count: int
    raw_token_estimate: int
    compressed_token_estimate: int
    tokens_saved: int
    usd_saved: float
    latency_ms: float
    arbitration_reason: str
    predictor_model: str            # "heuristic" | "gradient_boosting" | "none"
    expected_saving_tokens: int
    actual_saving_tokens: int
    prediction_error_tokens: int    # |expected - actual| (training signal quality)


@dataclass
class LedgerStats:
    """Aggregate statistics across all recorded entries (or a filtered subset)."""
    total_runs: int = 0
    compressed_runs: int = 0
    passthrough_runs: int = 0
    noop_runs: int = 0              # skipped by ArbitrationGate before any LLM call
    total_tokens_raw: int = 0
    total_tokens_compressed: int = 0
    total_tokens_saved: int = 0
    total_usd_saved: float = 0.0
    total_latency_ms: float = 0.0
    mean_compression_ratio: float = 0.0
    mean_prediction_error_tokens: float = 0.0

    @property
    def compression_rate(self) -> float:
        """Fraction of runs that were successfully compressed."""
        return self.compressed_runs / self.total_runs if self.total_runs else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_runs if self.total_runs else 0.0

    def to_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "compressed_runs": self.compressed_runs,
            "passthrough_runs": self.passthrough_runs,
            "noop_runs": self.noop_runs,
            "total_tokens_saved": self.total_tokens_saved,
            "total_usd_saved": round(self.total_usd_saved, 6),
            "compression_rate": round(self.compression_rate, 4),
            "mean_compression_ratio": round(self.mean_compression_ratio, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "mean_prediction_error_tokens": round(self.mean_prediction_error_tokens, 1),
        }


class CostLedger:
    """
    Thread-safe (via a simple lock) in-memory cost ledger.

    Entries are immutable once written. The ledger never evicts entries —
    memory is bounded by the number of workflow runs in the process lifetime,
    which is fine for a single-worker deployment. Phase 4 adds a PostgreSQL
    backend and per-worker flush.
    """

    def __init__(self, settings: EngineSettings) -> None:
        self._settings = settings
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()

    # ── Write ──────────────────────────────────────────────────────────────

    def record_compression(
        self,
        result: CompressionResult,
        features: CostFeatureVector,
        arbitration_reason: str = "",
        predictor_model: str = "heuristic",
        expected_saving_tokens: int = 0,
    ) -> LedgerEntry:
        """
        Record the outcome of a FallbackRouter.compress() call.
        Returns the created entry so callers can log it immediately.
        """
        actual_saving = max(
            0, features.raw_token_estimate - result.state.token_count_estimate
        )
        usd_saved = (
            actual_saving
            * self._settings.cost_token_price_per_million_usd
            / 1_000_000
        )
        pred_error = abs(expected_saving_tokens - actual_saving)

        entry = LedgerEntry(
            workflow_id=result.state.workflow_id,
            recorded_at=datetime.now(timezone.utc),
            route=result.route.value,
            source_turn_count=features.source_turn_count,
            raw_token_estimate=features.raw_token_estimate,
            compressed_token_estimate=result.state.token_count_estimate,
            tokens_saved=actual_saving,
            usd_saved=round(usd_saved, 8),
            latency_ms=result.latency_ms,
            arbitration_reason=arbitration_reason,
            predictor_model=predictor_model,
            expected_saving_tokens=expected_saving_tokens,
            actual_saving_tokens=actual_saving,
            prediction_error_tokens=pred_error,
        )
        with self._lock:
            self._entries.append(entry)

        logger.info(
            "Ledger entry recorded",
            extra={
                "workflow_id": entry.workflow_id,
                "route": entry.route,
                "tokens_saved": entry.tokens_saved,
                "usd_saved": entry.usd_saved,
                "prediction_error_tokens": entry.prediction_error_tokens,
            },
        )
        return entry

    def record_noop(
        self,
        features: CostFeatureVector,
        arbitration_reason: str,
    ) -> LedgerEntry:
        """Record an ArbitrationGate NOOP decision (no LLM call made)."""
        entry = LedgerEntry(
            workflow_id=features.workflow_id,
            recorded_at=datetime.now(timezone.utc),
            route=CompressionRoute.NOOP.value,
            source_turn_count=features.source_turn_count,
            raw_token_estimate=features.raw_token_estimate,
            compressed_token_estimate=features.raw_token_estimate,  # no change
            tokens_saved=0,
            usd_saved=0.0,
            latency_ms=0.0,
            arbitration_reason=arbitration_reason,
            predictor_model="none",
            expected_saving_tokens=0,
            actual_saving_tokens=0,
            prediction_error_tokens=0,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    # ── Read ───────────────────────────────────────────────────────────────

    def entries(self, workflow_id: str | None = None) -> list[LedgerEntry]:
        with self._lock:
            if workflow_id is None:
                return list(self._entries)
            return [e for e in self._entries if e.workflow_id == workflow_id]

    def stats(self, workflow_id: str | None = None) -> LedgerStats:
        rows = self.entries(workflow_id)
        if not rows:
            return LedgerStats()

        s = LedgerStats(total_runs=len(rows))
        ratios: list[float] = []
        pred_errors: list[float] = []

        for e in rows:
            if e.route == CompressionRoute.COMPRESSED.value:
                s.compressed_runs += 1
            elif e.route == CompressionRoute.PASSTHROUGH.value:
                s.passthrough_runs += 1
            else:
                s.noop_runs += 1

            s.total_tokens_raw += e.raw_token_estimate
            s.total_tokens_compressed += e.compressed_token_estimate
            s.total_tokens_saved += e.tokens_saved
            s.total_usd_saved += e.usd_saved
            s.total_latency_ms += e.latency_ms

            if e.raw_token_estimate > 0:
                ratios.append(e.tokens_saved / e.raw_token_estimate)
            pred_errors.append(float(e.prediction_error_tokens))

        s.mean_compression_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        s.mean_prediction_error_tokens = (
            sum(pred_errors) / len(pred_errors) if pred_errors else 0.0
        )
        return s

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        with self._lock:
            return iter(list(self._entries))
