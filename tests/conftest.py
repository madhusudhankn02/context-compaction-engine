from __future__ import annotations

import json
from pathlib import Path

import pytest

from compaction_engine.config import EngineSettings, LLMProviderName
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
    RawDialogueTurn,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_dialogues"


@pytest.fixture
def settings() -> EngineSettings:
    # Explicit FAKE provider + relaxed gates: unit tests exercise pipeline
    # *orchestration* logic, not provider correctness or production thresholds.
    return EngineSettings(
        llm_provider=LLMProviderName.FAKE,
        max_turns_per_chunk=5,
        max_facts_per_chunk=10,
        dedup_cosine_threshold=0.92,
        min_acceptable_fact_recall=0.9,
        max_acceptable_hallucination_rate=0.02,
        min_acceptable_constraint_recall=0.99,
    )


@pytest.fixture
def customer_support_turns() -> list[RawDialogueTurn]:
    raw = json.loads((DATA_DIR / "customer_support_escalation.json").read_text())
    return [RawDialogueTurn(**t) for t in raw["turns"]]


@pytest.fixture
def gold_standard_state() -> CompressedContextState:
    """
    Hand-annotated 'ground truth' extraction for
    customer_support_escalation.json — used as the reference trajectory in
    integrity-validator tests. In Phase 5 this kind of object is what the
    50+-workflow benchmark suite stores per workflow.
    """
    return CompressedContextState(
        workflow_id="wf-cs-escalation-0001",
        compaction_version=1,
        source_turn_count=10,
        token_count_estimate=120,
        facts=(
            ExtractedFact(
                fact_id="f-constraint-budget",
                fact_type=FactType.CONSTRAINT,
                statement="Customer has a hard $50 cap on replacement shipping fees.",
                source_agent="user",
                source_turn_start=0,
                source_turn_end=0,
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedFact(
                fact_id="f-decision-refund-1",
                fact_type=FactType.DECISION,
                statement="Refunds agent decided to issue a full $129.99 refund.",
                source_agent="refunds_agent",
                source_turn_start=3,
                source_turn_end=3,
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedFact(
                fact_id="f-decision-replacement",
                fact_type=FactType.DECISION,
                statement="Resolution switched from refund to replacement shipment.",
                source_agent="refunds_agent",
                source_turn_start=5,
                source_turn_end=5,
                confidence=ConfidenceLevel.HIGH,
                supersedes=("f-decision-refund-1",),
            ),
            ExtractedFact(
                fact_id="f-result-shipping-cost",
                fact_type=FactType.INTERMEDIATE_RESULT,
                statement="Expedited shipping would cost $65, exceeding the $50 cap.",
                source_agent="logistics_agent",
                source_turn_start=6,
                source_turn_end=6,
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedFact(
                fact_id="f-open-question-shipping",
                fact_type=FactType.OPEN_QUESTION,
                statement="Should shipping be standard (free) or expedited with a fee waiver?",
                source_agent="logistics_agent",
                source_turn_start=7,
                source_turn_end=7,
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedFact(
                fact_id="f-decision-final-shipping",
                fact_type=FactType.DECISION,
                statement="Final decision: ship replacement via free standard shipping.",
                source_agent="logistics_agent",
                source_turn_start=9,
                source_turn_end=9,
                confidence=ConfidenceLevel.HIGH,
                supersedes=("f-open-question-shipping",),
            ),
        ),
    )
