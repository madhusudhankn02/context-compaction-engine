"""
Dependency injection container for the bridge layer.

Why this exists: FastAPI's dependency system and the gRPC server both need
access to the same pipeline stack (settings → extractor → pipeline →
deduplicator → router). Without a container, each would either instantiate
their own copy (wasteful — embedding model loaded twice) or reach for a
module-level global (untestable). The container is constructed once at
process startup via the FastAPI lifespan and injected wherever needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from compaction_engine.bridge.router import FallbackRouter
from compaction_engine.config import EngineSettings, LLMProviderName
from compaction_engine.extraction.llm_providers import (
    FakeStructuredExtractor,
    StructuredExtractor,
    build_structured_extractor,
)
from compaction_engine.extraction.pipeline import ExtractionPipeline
from compaction_engine.extraction.reranker import FactDeduplicator
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ConfidenceLevel,
    ExtractedFact,
    FactType,
)
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BridgeContainer:
    settings: EngineSettings
    extractor: StructuredExtractor
    deduplicator: FactDeduplicator
    pipeline: ExtractionPipeline
    router: FallbackRouter


def _build_test_extractor() -> FakeStructuredExtractor:
    """Deterministic stub returned when llm_provider=fake (CI / offline dev)."""
    stub_state = CompressedContextState(
        workflow_id="__stub__",
        compaction_version=1,
        source_turn_count=1,
        token_count_estimate=10,
        facts=(
            ExtractedFact(
                fact_id="stub-001",
                fact_type=FactType.EXTERNAL_FACT,
                statement="Stub extractor active — configure a real provider for production.",
                source_agent="stub",
                source_turn_start=0,
                source_turn_end=0,
                confidence=ConfidenceLevel.LOW,
            ),
        ),
    )
    return FakeStructuredExtractor(responses=[stub_state])


def build_container(settings: EngineSettings) -> BridgeContainer:
    """
    Constructs the full dependency stack from settings.
    Call once at process startup; reuse the returned container everywhere.
    """
    logger.info(
        "Building bridge container",
        extra={"provider": settings.llm_provider.value, "model": settings.llm_model_name},
    )

    if settings.llm_provider == LLMProviderName.FAKE:
        extractor: StructuredExtractor = _build_test_extractor()
    else:
        extractor = build_structured_extractor(settings)

    deduplicator = FactDeduplicator(settings)
    pipeline = ExtractionPipeline(settings, extractor, deduplicator=deduplicator)
    router = FallbackRouter(pipeline, settings)

    logger.info("Bridge container ready")
    return BridgeContainer(
        settings=settings,
        extractor=extractor,
        deduplicator=deduplicator,
        pipeline=pipeline,
        router=router,
    )
