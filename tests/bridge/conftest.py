from __future__ import annotations

import pytest

from compaction_engine.bridge.container import BridgeContainer
from compaction_engine.bridge.rest.app import create_app
from compaction_engine.config import EngineSettings, LLMProviderName
from compaction_engine.extraction.llm_providers import FakeStructuredExtractor
from compaction_engine.extraction.pipeline import ExtractionPipeline
from compaction_engine.extraction.reranker import FactDeduplicator
from compaction_engine.bridge.router import FallbackRouter
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
)


def _make_seeded_state(
    workflow_id: str = "wf-test-001",
    turn_start: int = 0,
    turn_end: int = 4,
) -> CompressedContextState:
    return CompressedContextState(
        workflow_id=workflow_id,
        compaction_version=1,
        source_turn_count=5,
        token_count_estimate=30,
        facts=(
            ExtractedFact(
                fact_id="bridge-f1",
                fact_type=FactType.CONSTRAINT,
                statement="Budget ceiling is five thousand dollars per quarter.",
                source_agent="finance_agent",
                source_turn_start=turn_start,
                source_turn_end=turn_end,
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedFact(
                fact_id="bridge-f2",
                fact_type=FactType.DECISION,
                statement="Procurement team approved vendor shortlist for Q3.",
                source_agent="procurement_agent",
                source_turn_start=turn_start,
                source_turn_end=turn_end,
                confidence=ConfidenceLevel.HIGH,
            ),
        ),
    )


@pytest.fixture
def bridge_settings() -> EngineSettings:
    return EngineSettings(
        llm_provider=LLMProviderName.FAKE,
        max_turns_per_chunk=5,
        max_facts_per_chunk=10,
        dedup_cosine_threshold=0.92,
        min_acceptable_fact_recall=0.90,
        max_acceptable_hallucination_rate=0.02,
        min_acceptable_constraint_recall=0.99,
    )


@pytest.fixture
def seeded_extractor() -> FakeStructuredExtractor:
    """Returns a valid seeded state for the first chunk (turns 0-4)."""
    return FakeStructuredExtractor(responses=[_make_seeded_state()])


@pytest.fixture
def bridge_container(bridge_settings, seeded_extractor) -> BridgeContainer:
    dedup = FactDeduplicator(bridge_settings, embedder=None)
    pipeline = ExtractionPipeline(bridge_settings, seeded_extractor, deduplicator=dedup)
    router = FallbackRouter(pipeline, bridge_settings)
    return BridgeContainer(
        settings=bridge_settings,
        extractor=seeded_extractor,
        deduplicator=dedup,
        pipeline=pipeline,
        router=router,
    )


@pytest.fixture
def test_client(bridge_container):
    """
    FastAPI TestClient wired to the fake container — no network, no API keys.

    Must yield inside `with TestClient(app)`. Starlette only fires the
    lifespan startup event (which sets app.state.container) when TestClient
    is entered as a context manager. A bare return TestClient(app) skips
    lifespan entirely, causing AttributeError on every request.
    """
    from fastapi.testclient import TestClient
    app = create_app(container=bridge_container)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def raw_turns_payload() -> list[dict]:
    """Five ordered turns suitable for the seeded extractor (chunk 0 = turns 0-4)."""
    return [
        {"turnIndex": i, "agentName": "agent", "role": "agent",
         "content": f"Turn {i} content about the procurement budget approval."}
        for i in range(5)
    ]
