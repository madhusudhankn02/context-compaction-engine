"""
CompressionResult — typed envelope that every transport adapter returns.

Keeping this in its own file prevents circular imports: the FallbackRouter
imports it, the REST and gRPC layers import it, and neither needs to import
the other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from compaction_engine.extraction.pipeline import ExtractionRunResult
from compaction_engine.schemas.extraction_schema import CompressedContextState


class CompressionRoute(str, Enum):
    COMPRESSED = "compressed"        # pipeline ran successfully, context was reduced
    PASSTHROUGH = "passthrough"      # pipeline failed; original turns forwarded as-is
    NOOP = "noop"                    # dialogue was too short to bother compressing


@dataclass
class CompressionResult:
    """
    The universal return type from FallbackRouter.compress().

    Design invariant: `state` is ALWAYS a valid, schema-checked
    CompressedContextState, regardless of which route was taken. Downstream
    agents never need to branch on `route` to consume `state` safely —
    `route` is purely for observability and cost-accounting (Phase 3/4).
    """

    state: CompressedContextState
    route: CompressionRoute
    run_result: ExtractionRunResult | None = None  # None on PASSTHROUGH / NOOP
    fallback_reason: str | None = None             # populated on PASSTHROUGH
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """
        Fraction of tokens saved relative to a naïve passthrough.
        0.0 = no saving (passthrough), 1.0 = perfect (nothing left).
        Used by Phase 3's cost predictor as a feature.
        """
        if self.run_result is None:
            return 0.0
        raw_est = self.run_result.raw_turn_count * 100  # ~100 tokens/turn heuristic
        if raw_est == 0:
            return 0.0
        compressed_est = self.state.token_count_estimate
        return max(0.0, 1.0 - compressed_est / raw_est)

    def to_audit_dict(self) -> dict:
        """Flat dict suitable for direct insertion into the Phase 4 audit log."""
        return {
            "workflow_id": self.state.workflow_id,
            "route": self.route.value,
            "compression_ratio": round(self.compression_ratio, 4),
            "latency_ms": round(self.latency_ms, 2),
            "fallback_reason": self.fallback_reason,
            "fact_count": len(self.state.facts),
            "source_turn_count": self.state.source_turn_count,
            "token_count_estimate": self.state.token_count_estimate,
            "warnings": list(self.warnings),
        }
