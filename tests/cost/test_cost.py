"""
Phase 3 cost layer tests.

All tests run without sklearn, API keys, or network access.
The CostPredictor heuristic path is the primary target; the gradient-boosting
path is exercised if sklearn is installed (which it will be, given
sentence-transformers is a dependency), but we don't assert on it —
we assert on the INTERFACE, not the model internals.
"""

from __future__ import annotations

import pytest

from compaction_engine.bridge.result import CompressionResult, CompressionRoute
from compaction_engine.config import EngineSettings, LLMProviderName
from compaction_engine.cost.arbitration import ArbitrationGate, ArbitrationOutcome
from compaction_engine.cost.features import FEATURE_NAMES, CostFeatureVector, FeatureExtractor
from compaction_engine.cost.ledger import CostLedger
from compaction_engine.cost.predictor import CostPredictor
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
    RawDialogueTurn,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def cost_settings() -> EngineSettings:
    return EngineSettings(
        llm_provider=LLMProviderName.FAKE,
        cost_saving_threshold_tokens=50,
        cost_min_turns_to_compress=3,
        cost_predictor_min_samples=5,
        cost_max_backoff_to_compress=3.0,
        cost_token_price_per_million_usd=0.80,
    )


def _make_turns(n: int, content: str = "Agent discussed the procurement budget approval process.") -> list[RawDialogueTurn]:
    return [
        RawDialogueTurn(turn_index=i, agent_name="agent", role="agent", content=content)
        for i in range(n)
    ]


def _make_state(workflow_id: str = "wf-cost-001", token_estimate: int = 30) -> CompressedContextState:
    return CompressedContextState(
        workflow_id=workflow_id,
        compaction_version=1,
        source_turn_count=5,
        token_count_estimate=token_estimate,
        facts=(
            ExtractedFact(
                fact_id="cf1",
                fact_type=FactType.CONSTRAINT,
                statement="Budget ceiling is five thousand dollars per quarter.",
                source_agent="finance_agent",
                source_turn_start=0,
                source_turn_end=4,
                confidence=ConfidenceLevel.HIGH,
            ),
        ),
    )


def _make_compressed_result(workflow_id: str = "wf-cost-001", token_estimate: int = 30) -> CompressionResult:
    return CompressionResult(
        state=_make_state(workflow_id, token_estimate),
        route=CompressionRoute.COMPRESSED,
        latency_ms=50.0,
    )


# ── Feature extraction ────────────────────────────────────────────────────────

class TestCostFeatureVector:
    def test_feature_names_length_matches_sklearn_row(self, cost_settings):
        extractor = FeatureExtractor()
        turns = _make_turns(5)
        fv = extractor.from_turns("wf-x", turns, backoff_multiplier=1.0)
        assert len(fv.to_sklearn_row()) == len(FEATURE_NAMES)

    def test_workflow_id_bucket_is_deterministic(self, cost_settings):
        extractor = FeatureExtractor()
        turns = _make_turns(4)
        fv1 = extractor.from_turns("wf-stable-id", turns)
        fv2 = extractor.from_turns("wf-stable-id", turns)
        assert fv1.workflow_id_bucket == fv2.workflow_id_bucket

    def test_different_workflow_ids_may_have_different_buckets(self):
        extractor = FeatureExtractor()
        turns = _make_turns(4)
        fv_a = extractor.from_turns("workflow-aaa", turns)
        fv_b = extractor.from_turns("workflow-zzz", turns)
        # buckets are in [0, 1000)
        assert 0 <= fv_a.workflow_id_bucket < 1000
        assert 0 <= fv_b.workflow_id_bucket < 1000

    def test_raw_token_estimate_scales_with_content(self):
        extractor = FeatureExtractor()
        short_turns = _make_turns(5, content="Hi.")
        long_turns  = _make_turns(5, content="A" * 400)
        fv_short = extractor.from_turns("wf-x", short_turns)
        fv_long  = extractor.from_turns("wf-x", long_turns)
        assert fv_long.raw_token_estimate > fv_short.raw_token_estimate

    def test_enrich_from_result_updates_post_compression_fields(self, cost_settings):
        extractor = FeatureExtractor()
        turns = _make_turns(5)
        pre = extractor.from_turns("wf-x", turns, backoff_multiplier=1.2)
        result = _make_compressed_result(token_estimate=20)
        post = extractor.enrich_from_result(pre, result)

        assert post.compressed_token_estimate == 20
        assert post.compression_ratio > 0.0
        assert post.constraint_fact_fraction == 1.0   # 1 constraint / 1 total fact

    def test_actual_saving_tokens_is_non_negative(self, cost_settings):
        extractor = FeatureExtractor()
        turns = _make_turns(5)
        pre = extractor.from_turns("wf-x", turns)
        result = _make_compressed_result(token_estimate=pre.raw_token_estimate + 100)
        post = extractor.enrich_from_result(pre, result)
        assert post.actual_saving_tokens == 0   # no negative savings


# ── CostPredictor ─────────────────────────────────────────────────────────────

class TestCostPredictorHeuristic:
    def test_predict_returns_prediction_result(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        extractor = FeatureExtractor()
        fv = extractor.from_turns("wf-x", _make_turns(5), backoff_multiplier=1.0)
        result = predictor.predict(fv)
        assert result.model_used == "heuristic"
        assert result.confidence == 0.0
        assert result.expected_saving_tokens >= 0

    def test_heuristic_compresses_long_dialogues(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        extractor = FeatureExtractor()
        # Long content → high raw_token_estimate → should compress
        turns = _make_turns(10, content="A" * 300)
        fv = extractor.from_turns("wf-x", turns, backoff_multiplier=1.0)
        result = predictor.predict(fv)
        assert result.should_compress is True

    def test_heuristic_skips_short_dialogues(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        extractor = FeatureExtractor()
        # Very short turns → raw_token_estimate ≤ threshold*2 → skip
        turns = _make_turns(4, content="Hi.")
        fv = extractor.from_turns("wf-x", turns, backoff_multiplier=1.0)
        result = predictor.predict(fv)
        # With content "Hi." (3 chars), raw_token_estimate = (3*4)//4 = 3 tokens
        # threshold*2 = 100 → 3 < 100 → should_compress=False
        assert result.should_compress is False

    def test_heuristic_respects_backoff_ceiling(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        extractor = FeatureExtractor()
        turns = _make_turns(10, content="A" * 300)
        fv = extractor.from_turns("wf-x", turns,
                                   backoff_multiplier=cost_settings.cost_max_backoff_to_compress)
        result = predictor.predict(fv)
        assert result.should_compress is False

    def test_predictor_switches_to_model_after_min_samples(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        extractor = FeatureExtractor()
        assert not predictor.is_model_active

        for i in range(cost_settings.cost_predictor_min_samples):
            turns = _make_turns(5, content="A" * 200)
            fv = extractor.from_turns(f"wf-{i}", turns)
            post = extractor.enrich_from_result(fv, _make_compressed_result(token_estimate=30))
            predictor.update(post, actual_saving_tokens=post.actual_saving_tokens)

        # sklearn is installed (via sentence-transformers dep), so model should be active
        assert predictor.sample_count == cost_settings.cost_predictor_min_samples
        # model may or may not be fitted depending on sklearn availability — just check it doesn't crash
        fv2 = extractor.from_turns("wf-predict", _make_turns(5))
        result = predictor.predict(fv2)
        assert result.expected_saving_tokens >= 0
        assert isinstance(result.should_compress, bool)


# ── ArbitrationGate ───────────────────────────────────────────────────────────

class TestArbitrationGate:
    def _make_gate(self, cost_settings) -> ArbitrationGate:
        return ArbitrationGate(
            predictor=CostPredictor(cost_settings),
            settings=cost_settings,
            extractor=FeatureExtractor(),
        )

    def test_fast_reject_too_few_turns(self, cost_settings):
        gate = self._make_gate(cost_settings)
        # min_turns_to_compress=3, send 2
        turns = _make_turns(2)
        decision = gate.evaluate("wf-x", turns, backoff_multiplier=1.0)
        assert decision.outcome == ArbitrationOutcome.NOOP
        assert decision.prediction is None
        assert "turn count" in decision.reason

    def test_fast_reject_excessive_backoff(self, cost_settings):
        gate = self._make_gate(cost_settings)
        turns = _make_turns(10)
        decision = gate.evaluate(
            "wf-x", turns,
            backoff_multiplier=cost_settings.cost_max_backoff_to_compress + 0.1,
        )
        assert decision.outcome == ArbitrationOutcome.NOOP
        assert "backoff" in decision.reason

    def test_compress_decision_for_substantial_dialogue(self, cost_settings):
        gate = self._make_gate(cost_settings)
        turns = _make_turns(10, content="A" * 300)
        decision = gate.evaluate("wf-x", turns, backoff_multiplier=1.0)
        # Heuristic: 10 turns × 300 chars → ~750 tokens → well above threshold*2
        assert decision.outcome == ArbitrationOutcome.COMPRESS
        assert decision.prediction is not None
        assert decision.features.source_turn_count == 10

    def test_decision_is_deterministic_for_same_input(self, cost_settings):
        gate = self._make_gate(cost_settings)
        turns = _make_turns(8, content="B" * 250)
        d1 = gate.evaluate("wf-stable", turns, backoff_multiplier=1.0)
        d2 = gate.evaluate("wf-stable", turns, backoff_multiplier=1.0)
        assert d1.outcome == d2.outcome

    def test_record_outcome_trains_predictor(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        gate = ArbitrationGate(
            predictor=predictor,
            settings=cost_settings,
            extractor=FeatureExtractor(),
        )
        assert predictor.sample_count == 0

        turns = _make_turns(8, content="C" * 200)
        decision = gate.evaluate("wf-train", turns, backoff_multiplier=1.0)
        result = _make_compressed_result(token_estimate=30)
        gate.record_outcome(decision.features, result)

        assert predictor.sample_count == 1

    def test_record_outcome_skips_passthrough(self, cost_settings):
        predictor = CostPredictor(cost_settings)
        gate = ArbitrationGate(
            predictor=predictor,
            settings=cost_settings,
            extractor=FeatureExtractor(),
        )
        turns = _make_turns(8, content="D" * 200)
        decision = gate.evaluate("wf-x", turns)
        passthrough_result = CompressionResult(
            state=_make_state(token_estimate=500),
            route=CompressionRoute.PASSTHROUGH,
            latency_ms=5.0,
        )
        gate.record_outcome(decision.features, passthrough_result)
        # Passthrough has no ground truth → predictor must NOT be updated
        assert predictor.sample_count == 0


# ── CostLedger ────────────────────────────────────────────────────────────────

class TestCostLedger:
    def test_record_compression_increments_length(self, cost_settings):
        ledger = CostLedger(cost_settings)
        extractor = FeatureExtractor()
        fv = extractor.from_turns("wf-l", _make_turns(5))
        result = _make_compressed_result(token_estimate=30)
        ledger.record_compression(result, fv)
        assert len(ledger) == 1

    def test_record_noop_increments_length(self, cost_settings):
        ledger = CostLedger(cost_settings)
        extractor = FeatureExtractor()
        fv = extractor.from_turns("wf-noop", _make_turns(2))
        ledger.record_noop(fv, "too few turns")
        assert len(ledger) == 1

    def test_stats_tokens_saved_is_correct(self, cost_settings):
        ledger = CostLedger(cost_settings)
        extractor = FeatureExtractor()
        turns = _make_turns(5, content="A" * 200)
        fv = extractor.from_turns("wf-l", turns)
        result = _make_compressed_result(token_estimate=20)
        ledger.record_compression(result, fv)

        stats = ledger.stats()
        assert stats.total_runs == 1
        assert stats.compressed_runs == 1
        assert stats.total_tokens_saved == max(0, fv.raw_token_estimate - 20)

    def test_stats_workflow_filter(self, cost_settings):
        ledger = CostLedger(cost_settings)
        extractor = FeatureExtractor()

        for wf in ["wf-a", "wf-a", "wf-b"]:
            fv = extractor.from_turns(wf, _make_turns(5))
            ledger.record_compression(_make_compressed_result(wf), fv)

        assert ledger.stats("wf-a").total_runs == 2
        assert ledger.stats("wf-b").total_runs == 1
        assert ledger.stats().total_runs == 3

    def test_stats_to_dict_has_required_keys(self, cost_settings):
        ledger = CostLedger(cost_settings)
        extractor = FeatureExtractor()
        fv = extractor.from_turns("wf-l", _make_turns(5))
        ledger.record_compression(_make_compressed_result(), fv)

        d = ledger.stats().to_dict()
        required = {
            "total_runs", "compressed_runs", "passthrough_runs", "noop_runs",
            "total_tokens_saved", "total_usd_saved", "compression_rate",
            "mean_compression_ratio", "mean_latency_ms",
            "mean_prediction_error_tokens",
        }
        assert required.issubset(d.keys())

    def test_usd_saved_calculation(self, cost_settings):
        ledger = CostLedger(cost_settings)
        extractor = FeatureExtractor()
        turns = _make_turns(5, content="A" * 400)
        fv = extractor.from_turns("wf-usd", turns)
        result = _make_compressed_result(token_estimate=10)

        entry = ledger.record_compression(result, fv)
        # price = 0.80 USD/M tokens; saving ≈ raw_estimate - 10
        expected_usd = entry.tokens_saved * 0.80 / 1_000_000
        assert abs(entry.usd_saved - expected_usd) < 1e-9

    def test_empty_ledger_stats_are_zero(self, cost_settings):
        ledger = CostLedger(cost_settings)
        s = ledger.stats()
        assert s.total_runs == 0
        assert s.total_tokens_saved == 0
        assert s.compression_rate == 0.0
