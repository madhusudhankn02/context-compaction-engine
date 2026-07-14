from __future__ import annotations

import pytest

from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
)
from compaction_engine.utils.exceptions import IntegrityValidationError
from compaction_engine.validation.integrity_validator import ContextIntegrityValidator


def _fact(fact_id, statement, fact_type=FactType.DECISION, confidence=ConfidenceLevel.HIGH):
    return ExtractedFact(
        fact_id=fact_id,
        fact_type=fact_type,
        statement=statement,
        source_agent="test_agent",
        source_turn_start=0,
        source_turn_end=0,
        confidence=confidence,
    )


class TestContextIntegrityValidatorHappyPath:
    def test_near_identical_compressed_state_passes_all_gates(self, settings, gold_standard_state):
        # Re-derive a "compressed" state from the same active facts as gold —
        # simulating a compaction result that lost nothing of substance.
        compressed = CompressedContextState(
            workflow_id=gold_standard_state.workflow_id,
            compaction_version=1,
            source_turn_count=10,
            token_count_estimate=80,
            facts=tuple(gold_standard_state.active_facts()),
        )
        validator = ContextIntegrityValidator(settings, embedder=None)

        report = validator.validate(gold_standard_state, compressed, match_threshold=0.99)

        assert report.fact_recall == 1.0
        assert report.hallucination_rate == 0.0
        assert report.constraint_recall == 1.0


class TestContextIntegrityValidatorHardGates:
    def test_raises_when_constraint_is_dropped(self, settings):
        gold = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=10,
            facts=(
                _fact("c1", "Hard cap of fifty dollars on shipping.", fact_type=FactType.CONSTRAINT),
                _fact("d1", "Decision to ship via standard shipping."),
            ),
        )
        # Compressed state drops the constraint entirely.
        compressed = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=5,
            facts=(_fact("d1", "Decision to ship via standard shipping."),),
        )
        validator = ContextIntegrityValidator(settings, embedder=None)

        with pytest.raises(IntegrityValidationError, match="constraint_recall"):
            validator.validate(gold, compressed, match_threshold=0.5)

    def test_raises_when_open_question_is_dropped(self, settings):
        gold = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=10,
            facts=(
                _fact("q1", "Should shipping be standard or expedited.", fact_type=FactType.OPEN_QUESTION),
                _fact("d1", "Decision to issue a refund."),
            ),
        )
        compressed = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=5,
            facts=(_fact("d1", "Decision to issue a refund."),),
        )
        validator = ContextIntegrityValidator(settings, embedder=None)

        with pytest.raises(IntegrityValidationError, match="open_question_recall"):
            validator.validate(gold, compressed, match_threshold=0.5)

    def test_raises_on_excessive_hallucination(self, settings):
        gold = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=10,
            facts=(_fact("d1", "Decision to issue a refund."),),
        )
        # Compressed state keeps the real fact AND fabricates several more
        # that share no lexical overlap with anything in gold.
        compressed = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=30,
            facts=(
                _fact("d1", "Decision to issue a refund."),
                _fact("h1", "Zebra migration patterns in winter."),
                _fact("h2", "Quarterly tax filing extension granted."),
            ),
        )
        validator = ContextIntegrityValidator(settings, embedder=None)

        with pytest.raises(IntegrityValidationError, match="hallucination_rate"):
            validator.validate(gold, compressed, match_threshold=0.5)

    def test_compare_does_not_raise_even_on_failing_state(self, settings):
        """`compare()` is the non-raising sibling used by the Phase 5 benchmark
        suite — it must always return a report, never throw, regardless of
        how bad the fidelity is."""
        gold = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=10,
            facts=(_fact("c1", "Hard cap of fifty dollars.", fact_type=FactType.CONSTRAINT),),
        )
        # Totally unrelated fact — simulates a worst-case failed compression
        # without tripping the schema's "zero facts" guard.
        compressed = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=3,
            token_count_estimate=5,
            facts=(_fact("h1", "Completely unrelated fabricated claim."),),
        )
        validator = ContextIntegrityValidator(settings, embedder=None)
        report = validator.compare(gold, compressed)

        assert report.fact_recall == 0.0
        assert report.constraint_recall == 0.0
        assert report.to_dict()["overall_fidelity_score"] == 0.0
