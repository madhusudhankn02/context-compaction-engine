from __future__ import annotations

import pytest
from pydantic import ValidationError

from compaction_engine.extraction.llm_providers import FakeStructuredExtractor
from compaction_engine.extraction.pipeline import ExtractionPipeline, chunk_turns
from compaction_engine.extraction.reranker import FactDeduplicator
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
    RawDialogueTurn,
)
from compaction_engine.utils.exceptions import ExtractionFidelityError


def _fact(
    fact_id: str,
    statement: str,
    *,
    fact_type: FactType = FactType.DECISION,
    turn_start: int = 0,
    turn_end: int = 0,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> ExtractedFact:
    return ExtractedFact(
        fact_id=fact_id,
        fact_type=fact_type,
        statement=statement,
        source_agent="test_agent",
        source_turn_start=turn_start,
        source_turn_end=turn_end,
        confidence=confidence,
    )


class TestChunking:
    def test_splits_turns_into_fixed_size_windows(self, customer_support_turns):
        chunks = chunk_turns(customer_support_turns, max_turns_per_chunk=4)
        # 10 turns / 4 per chunk -> chunks of size 4, 4, 2
        assert [len(c.turns) for c in chunks] == [4, 4, 2]

    def test_rejects_non_monotonic_turn_indices(self, customer_support_turns):
        shuffled = [customer_support_turns[1], customer_support_turns[0], *customer_support_turns[2:]]
        with pytest.raises(ValueError, match="strictly increasing"):
            chunk_turns(shuffled, max_turns_per_chunk=4)

    def test_empty_input_returns_no_chunks(self):
        assert chunk_turns([], max_turns_per_chunk=4) == []


class TestExtractionPipelineHappyPath:
    def test_run_produces_valid_merged_state(self, settings, customer_support_turns):
        # Root cause of the original failure: the settings fixture sets
        # max_turns_per_chunk=5, so 10 turns always produce 2 chunks (0-4,
        # 5-9).  Supplying a single seeded state whose facts cite turns 0-3
        # causes the grounding check to *correctly* reject those facts when
        # replayed against chunk 2 (valid range 5-9).
        #
        # Fix: supply one grounding-valid response per chunk, each citing only
        # the turn indices that exist in its own chunk.
        chunk0_state = CompressedContextState(
            workflow_id="wf-cs-escalation-0001",
            compaction_version=1,
            source_turn_count=10,
            token_count_estimate=42,
            facts=(
                _fact(
                    "f1", "Customer has a $50 shipping fee cap.",
                    fact_type=FactType.CONSTRAINT,
                    turn_start=0, turn_end=0,   # turn 0 is in chunk 0 (turns 0-4)
                ),
                _fact("f2", "Refund of $129.99 was decided.", turn_start=3, turn_end=3),
            ),
        )
        chunk1_state = CompressedContextState(
            workflow_id="wf-cs-escalation-0001",
            compaction_version=1,
            source_turn_count=10,
            token_count_estimate=20,
            facts=(
                _fact(
                    "f3", "Final resolution switched to standard-shipping replacement.",
                    turn_start=5, turn_end=9,   # turns 5-9 are all in chunk 1
                ),
            ),
        )
        extractor = FakeStructuredExtractor(responses=[chunk0_state, chunk1_state])
        pipeline = ExtractionPipeline(settings, extractor)

        result = pipeline.run("wf-cs-escalation-0001", customer_support_turns)

        assert result.final_state.workflow_id == "wf-cs-escalation-0001"
        assert result.raw_turn_count == 10
        assert result.chunks_processed == 2
        assert result.facts_after_dedup >= 1
        # The extractor must have been called exactly once per chunk.
        assert extractor.call_count == result.chunks_processed

    def test_multi_chunk_run_calls_extractor_once_per_chunk(self, settings, customer_support_turns):
        # max_turns_per_chunk=5 (from the `settings` fixture) over 10 turns -> 2 chunks.
        #
        # CRITICAL: pass an explicit FactDeduplicator(embedder=None) so the pipeline
        # never tries to load a live sentence-transformers model. Without this, the
        # deduplicator loads all-MiniLM-L6-v2 and computes *real* cosine similarity
        # over the test's seeded fact statements. The old placeholder statements
        # ("First chunk fact." / "Second chunk fact.") had cosine similarity ≥ 0.92,
        # so the real model correctly deduplicated them to one — failing the assertion.
        # Tests that exercise pipeline orchestration must be deterministic regardless
        # of what optional ML dependencies are installed.
        controlled_dedup = FactDeduplicator(settings, embedder=None)

        per_chunk_state_1 = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=10,
            token_count_estimate=10,
            facts=(
                _fact(
                    "f1",
                    # Semantically distinct from f2: distinct subject, predicate, and domain.
                    # Jaccard with f2's statement ≈ 0.05 (nearly disjoint token sets).
                    "Customer confirmed a $50 hard cap on replacement shipping fees.",
                    fact_type=FactType.CONSTRAINT,
                    turn_start=0, turn_end=4,
                ),
            ),
        )
        per_chunk_state_2 = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=10,
            token_count_estimate=10,
            facts=(
                _fact(
                    "f2",
                    "Logistics agent confirmed replacement unit is in stock at warehouse W-12.",
                    fact_type=FactType.INTERMEDIATE_RESULT,
                    turn_start=5, turn_end=9,
                ),
            ),
        )
        extractor = FakeStructuredExtractor(responses=[per_chunk_state_1, per_chunk_state_2])
        pipeline = ExtractionPipeline(settings, extractor, deduplicator=controlled_dedup)

        result = pipeline.run("wf-cs-escalation-0001", customer_support_turns)

        assert result.chunks_processed == 2
        assert extractor.call_count == 2
        fact_ids = {f.fact_id for f in result.final_state.facts}
        assert {"f1", "f2"}.issubset(fact_ids)


class TestExtractionPipelineGroundingEnforcement:
    def test_rejects_fact_citing_out_of_range_turn(self, settings, customer_support_turns):
        # max_turns_per_chunk=5 -> first chunk covers turns 0-4. A fact citing
        # turn 7 in that chunk is, by construction, an ungrounded citation.
        bad_state = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=10,
            token_count_estimate=10,
            facts=(_fact("f1", "Hallucinated citation.", turn_start=7, turn_end=7),),
        )
        extractor = FakeStructuredExtractor(responses=[bad_state])
        pipeline = ExtractionPipeline(settings, extractor)

        with pytest.raises(ExtractionFidelityError, match="outside the chunk's valid range"):
            pipeline.run("wf-cs-escalation-0001", customer_support_turns)

    def test_empty_turns_raises_value_error(self, settings):
        extractor = FakeStructuredExtractor(
            responses=[
                CompressedContextState(
                    workflow_id="wf-x", compaction_version=1, source_turn_count=0,
                    token_count_estimate=0, facts=(),
                )
            ]
        )
        pipeline = ExtractionPipeline(settings, extractor)
        with pytest.raises(ValueError, match="empty turn list"):
            pipeline.run("wf-empty", [])


class TestSchemaEnforcement:
    def test_rejects_narrative_style_statement(self):
        with pytest.raises(ValidationError):
            _fact("f1", "First the customer complained. Then the agent apologized. Then a refund was issued.")

    def test_rejects_zero_facts_for_nonempty_dialogue(self):
        with pytest.raises(ValidationError, match="collapsed a non-empty dialogue"):
            CompressedContextState(
                workflow_id="wf-x", compaction_version=1, source_turn_count=5,
                token_count_estimate=0, facts=(),
            )

    def test_rejects_duplicate_fact_ids(self):
        with pytest.raises(ValidationError, match="duplicate fact_id"):
            CompressedContextState(
                workflow_id="wf-x", compaction_version=1, source_turn_count=2,
                token_count_estimate=10,
                facts=(_fact("dup", "First."), _fact("dup", "Second.")),
            )

    def test_rejects_invalid_turn_range(self):
        with pytest.raises(ValidationError, match="cannot precede"):
            _fact("f1", "Bad range.", turn_start=5, turn_end=2)

    def test_active_facts_excludes_superseded(self):
        state = CompressedContextState(
            workflow_id="wf-x", compaction_version=1, source_turn_count=2,
            token_count_estimate=10,
            facts=(
                _fact("f1", "Original decision."),
                ExtractedFact(
                    fact_id="f2",
                    fact_type=FactType.DECISION,
                    statement="Revised decision.",
                    source_agent="test_agent",
                    source_turn_start=1,
                    source_turn_end=1,
                    confidence=ConfidenceLevel.HIGH,
                    supersedes=("f1",),
                ),
            ),
        )
        active_ids = {f.fact_id for f in state.active_facts()}
        assert active_ids == {"f2"}


class TestFactDeduplicator:
    def test_drops_lexically_near_duplicate_facts(self, settings):
        # Root cause of the original failure: the old pair —
        #   "Customer wants a refund issued today."   (6 tokens)
        #   "Customer wants a refund issued today now."  (7 tokens)
        # Jaccard = 6/7 ≈ 0.857  →  below the settings.dedup_cosine_threshold=0.92
        # so the deduplicator *correctly* kept both.
        #
        # Fix: use a pair whose Jaccard score is >= 0.92 so the test actually
        # exercises the dedup-and-keep-higher-confidence path.
        #
        # Chosen strings:
        #   s1: "Full refund issued to customer payment method."
        #       tokens: {full, refund, issued, to, customer, payment, method} = 7
        #   s2: "Full refund issued to the customer payment method."
        #       tokens: {full, refund, issued, to, the, customer, payment, method} = 8
        #   intersection = 7,  union = 8,  Jaccard = 7/8 = 0.875  -- still not enough.
        #
        # Simplest high-Jaccard pair: identical except one extra stop-word,
        # where the stop-word brings union up by 1 over a large shared base:
        #   s1 (12 tokens): {a, budget, cap, customer, fee, fifty, hard, limit, on,
        #                     replacement, shipping, the}
        #   s2 (13 tokens): same + {dollar}
        #   Jaccard = 12/13 ≈ 0.923  ✓
        #
        # To keep test intent clear we use a settings override with a threshold
        # that matches realistic lexical-fallback usage (0.80), so this test
        # does not depend on the precise Jaccard of the chosen strings being
        # above some tight global threshold.
        lexical_settings = settings.model_copy(update={"dedup_cosine_threshold": 0.80})
        dedup = FactDeduplicator(lexical_settings, embedder=None)  # forces lexical fallback

        # Jaccard = 10/11 ≈ 0.909  ✓ (well above 0.80 threshold)
        # s1 tokens: {a, agent, amount, decided, full, issue, of, refund, the, to}       = 10
        # s2 tokens: {a, agent, amount, decided, full, issue, now, of, refund, the, to}  = 11
        # intersection = 10,  union = 11,  Jaccard ≈ 0.909
        facts = [
            _fact(
                "f1",
                "Agent decided to issue a full refund of the amount.",
                confidence=ConfidenceLevel.MEDIUM,
            ),
            _fact(
                "f2",
                "Agent decided to issue a full refund of the amount now.",
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        result = dedup.deduplicate(facts)
        assert len(result) == 1
        assert result[0].fact_id == "f2"  # higher confidence wins the tie-break

    def test_never_drops_constraints_or_open_questions(self, settings):
        dedup = FactDeduplicator(settings, embedder=None)
        facts = [
            _fact("f1", "Hard budget cap is fifty dollars.", fact_type=FactType.CONSTRAINT),
            _fact("f2", "Hard budget cap is fifty dollars exactly.", fact_type=FactType.CONSTRAINT),
        ]
        result = dedup.deduplicate(facts)
        # Both retained — safety-critical types are exempt from similarity-based dedup.
        assert {f.fact_id for f in result} == {"f1", "f2"}
