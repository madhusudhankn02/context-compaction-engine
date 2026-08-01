from __future__ import annotations

import pytest

from compaction_engine.bridge.result import CompressionRoute
from compaction_engine.bridge.router import FallbackRouter, _build_passthrough_state
from compaction_engine.extraction.llm_providers import FakeStructuredExtractor
from compaction_engine.extraction.pipeline import ExtractionPipeline
from compaction_engine.extraction.reranker import FactDeduplicator
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
    RawDialogueTurn,
)
from compaction_engine.utils.exceptions import SchemaValidationError


def _make_turns(n: int = 3) -> list[RawDialogueTurn]:
    return [
        RawDialogueTurn(turn_index=i, agent_name="agent", role="agent", content=f"Turn {i}.")
        for i in range(n)
    ]


class TestFallbackRouterHappyPath:
    def test_returns_compressed_route_on_success(self, bridge_container):
        result = bridge_container.router.compress("wf-test-001", _make_turns(5))
        assert result.route == CompressionRoute.COMPRESSED
        assert result.fallback_reason is None
        assert result.run_result is not None
        assert len(result.state.facts) >= 1

    def test_compression_ratio_is_positive_on_success(self, bridge_container):
        result = bridge_container.router.compress("wf-test-001", _make_turns(5))
        assert result.compression_ratio >= 0.0

    def test_audit_dict_has_all_required_keys(self, bridge_container):
        result = bridge_container.router.compress("wf-test-001", _make_turns(5))
        audit = result.to_audit_dict()
        required = {
            "workflow_id", "route", "compression_ratio", "latency_ms",
            "fallback_reason", "fact_count", "source_turn_count",
            "token_count_estimate", "warnings",
        }
        assert required.issubset(audit.keys())

    def test_raises_on_empty_turns(self, bridge_container):
        with pytest.raises(ValueError, match="empty turn list"):
            bridge_container.router.compress("wf-empty", [])


class TestFallbackRouterFallbackPath:
    def _make_failing_router(self, bridge_settings) -> FallbackRouter:
        """Router whose extractor always raises SchemaValidationError."""
        class _AlwaysFailExtractor:
            def invoke(self, _msgs):
                raise SchemaValidationError("Intentional test failure.")

        dedup = FactDeduplicator(bridge_settings, embedder=None)
        pipeline = ExtractionPipeline(bridge_settings, _AlwaysFailExtractor(), deduplicator=dedup)
        return FallbackRouter(pipeline, bridge_settings)

    def test_fallback_produces_passthrough_route(self, bridge_settings):
        router = self._make_failing_router(bridge_settings)
        result = router.compress("wf-fail-001", _make_turns(3))
        assert result.route == CompressionRoute.PASSTHROUGH

    def test_fallback_state_is_schema_valid(self, bridge_settings):
        router = self._make_failing_router(bridge_settings)
        result = router.compress("wf-fail-001", _make_turns(3))
        # Should not raise — passthrough state must always be schema-valid.
        assert isinstance(result.state, CompressedContextState)

    def test_fallback_state_contains_all_turns(self, bridge_settings):
        n = 4
        router = self._make_failing_router(bridge_settings)
        result = router.compress("wf-fail-001", _make_turns(n))
        # Passthrough: one fact per turn.
        assert len(result.state.facts) == n

    def test_fallback_reason_is_populated(self, bridge_settings):
        router = self._make_failing_router(bridge_settings)
        result = router.compress("wf-fail-001", _make_turns(3))
        assert result.fallback_reason is not None
        assert "SchemaValidationError" in result.fallback_reason

    def test_repeated_fallbacks_increase_backoff_multiplier(self, bridge_settings):
        router = self._make_failing_router(bridge_settings)
        wf = "wf-backoff-test"
        assert router.get_backoff_multiplier(wf) == 1.0
        router.compress(wf, _make_turns(2))
        m1 = router.get_backoff_multiplier(wf)
        router.compress(wf, _make_turns(2))
        m2 = router.get_backoff_multiplier(wf)
        assert m2 > m1 > 1.0

    def test_reset_backoff_clears_multiplier(self, bridge_settings):
        router = self._make_failing_router(bridge_settings)
        wf = "wf-reset-test"
        router.compress(wf, _make_turns(2))
        assert router.get_backoff_multiplier(wf) > 1.0
        router.reset_backoff(wf)
        assert router.get_backoff_multiplier(wf) == 1.0


class TestPassthroughStateBuilder:
    def test_passthrough_state_has_one_fact_per_turn(self):
        turns = _make_turns(6)
        state = _build_passthrough_state("wf-x", turns)
        assert len(state.facts) == 6

    def test_passthrough_facts_cover_every_turn_index(self):
        turns = _make_turns(4)
        state = _build_passthrough_state("wf-x", turns)
        covered = {f.source_turn_start for f in state.facts}
        assert covered == {0, 1, 2, 3}

    def test_passthrough_state_is_schema_valid(self):
        turns = _make_turns(3)
        state = _build_passthrough_state("wf-x", turns)
        # Re-validate via model_validate to be sure schema passes.
        re_validated = CompressedContextState.model_validate(state.model_dump())
        assert re_validated.workflow_id == "wf-x"
