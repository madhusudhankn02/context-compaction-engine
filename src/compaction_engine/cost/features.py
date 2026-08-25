"""
Cost feature engineering.

Two classes:
  CostFeatureVector  — immutable snapshot of all features for one workflow run.
  FeatureExtractor   — builds a CostFeatureVector from raw turns (pre-compression)
                       and optionally enriches it from a CompressionResult
                       (post-compression, used for model training).

Feature stability guarantee
───────────────────────────
`to_sklearn_row()` returns features in a FIXED ORDER.  FEATURE_NAMES matches
that order exactly.  Never reorder or remove columns between versions — doing
so silently invalidates any persisted model.  Append new features at the end
with a safe default so old serialised models don't break.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compaction_engine.bridge.result import CompressionResult
    from compaction_engine.schemas.extraction_schema import RawDialogueTurn

# Ordered feature names — must match to_sklearn_row() exactly.
FEATURE_NAMES: tuple[str, ...] = (
    "source_turn_count",
    "raw_token_estimate",
    "avg_content_length",
    "backoff_multiplier",
    "workflow_id_bucket",   # hash(workflow_id) % 1000 — workflow identity signal
    # ── post-compression (0.0 when not yet compressed) ──
    "compressed_token_estimate",
    "dedup_ratio",
    "compression_ratio",
    "chunk_count",
    "constraint_fact_fraction",
)


@dataclass(frozen=True)
class CostFeatureVector:
    """Immutable feature snapshot for one workflow compression attempt."""

    workflow_id: str
    captured_at: datetime

    # ── Pre-compression features (available before running the pipeline) ──
    source_turn_count: int
    raw_token_estimate: int        # ~4 chars / token heuristic
    avg_content_length: float      # mean chars per turn
    backoff_multiplier: float      # from FallbackRouter._threshold_registry

    # ── Post-compression features (default 0 until enriched) ──
    compressed_token_estimate: int = 0
    dedup_ratio: float = 0.0
    compression_ratio: float = 0.0
    chunk_count: int = 0
    constraint_fact_fraction: float = 0.0  # fraction of facts that are CONSTRAINT/OPEN_QUESTION

    # ── Derived / cached ──
    workflow_id_bucket: int = field(init=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        bucket = int(hashlib.md5(self.workflow_id.encode()).hexdigest(), 16) % 1000
        object.__setattr__(self, "workflow_id_bucket", bucket)

    # ------------------------------------------------------------------
    # Sklearn interface
    # ------------------------------------------------------------------

    def to_sklearn_row(self) -> list[float]:
        """Stable, ordered float vector matching FEATURE_NAMES."""
        return [
            float(self.source_turn_count),
            float(self.raw_token_estimate),
            float(self.avg_content_length),
            float(self.backoff_multiplier),
            float(self.workflow_id_bucket),
            float(self.compressed_token_estimate),
            float(self.dedup_ratio),
            float(self.compression_ratio),
            float(self.chunk_count),
            float(self.constraint_fact_fraction),
        ]

    @property
    def actual_saving_tokens(self) -> int:
        """Tokens saved vs. raw baseline.  Only meaningful post-compression."""
        return max(0, self.raw_token_estimate - self.compressed_token_estimate)


class FeatureExtractor:
    """
    Builds CostFeatureVector objects from pipeline inputs/outputs.

    Stateless — safe to reuse across workflows.
    """

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def from_turns(
        self,
        workflow_id: str,
        turns: "list[RawDialogueTurn]",
        backoff_multiplier: float = 1.0,
    ) -> CostFeatureVector:
        """
        Build a PRE-compression feature vector from raw dialogue turns.
        Used by the ArbitrationGate to decide whether to even attempt compression.
        """
        if not turns:
            raise ValueError("Cannot extract features from an empty turn list.")

        total_chars = sum(len(t.content) for t in turns)
        avg_len = total_chars / len(turns)
        raw_tokens = self._estimate_tokens(" ".join(t.content for t in turns))

        return CostFeatureVector(
            workflow_id=workflow_id,
            captured_at=datetime.now(timezone.utc),
            source_turn_count=len(turns),
            raw_token_estimate=raw_tokens,
            avg_content_length=avg_len,
            backoff_multiplier=backoff_multiplier,
        )

    def enrich_from_result(
        self,
        base: CostFeatureVector,
        result: "CompressionResult",
    ) -> CostFeatureVector:
        """
        Enrich a pre-compression CostFeatureVector with post-compression data.
        Returns a new (frozen) instance — does not mutate `base`.
        Used after a compression run to produce the training sample.
        """
        run = result.run_result
        facts = result.state.facts

        constraint_count = sum(1 for f in facts if f.is_safety_critical)
        constraint_fraction = constraint_count / len(facts) if facts else 0.0

        compressed_est = result.state.token_count_estimate
        compression_ratio = (
            max(0.0, 1.0 - compressed_est / base.raw_token_estimate)
            if base.raw_token_estimate > 0
            else 0.0
        )

        return CostFeatureVector(
            workflow_id=base.workflow_id,
            captured_at=base.captured_at,
            source_turn_count=base.source_turn_count,
            raw_token_estimate=base.raw_token_estimate,
            avg_content_length=base.avg_content_length,
            backoff_multiplier=base.backoff_multiplier,
            compressed_token_estimate=compressed_est,
            dedup_ratio=run.dedup_ratio if run else 0.0,
            compression_ratio=compression_ratio,
            chunk_count=run.chunks_processed if run else 0,
            constraint_fact_fraction=constraint_fraction,
        )
