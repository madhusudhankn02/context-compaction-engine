"""
Dependency injection container for the bridge layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from compaction_engine.bridge.router import FallbackRouter
from compaction_engine.config import EngineSettings, LLMProviderName
from compaction_engine.cost.arbitration import ArbitrationGate
from compaction_engine.cost.features import FeatureExtractor
from compaction_engine.cost.ledger import CostLedger
from compaction_engine.cost.predictor import CostPredictor
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
    # Phase 3 cost layer (always present in full builds)
    cost_predictor: CostPredictor
    cost_ledger: CostLedger
    arbitration_gate: ArbitrationGate


def _build_test_extractor() -> FakeStructuredExtractor:
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
    Constructs the full dependency stack — Phase 1 + 2 + 3 — from settings.
    """
    logger.info(
        "Building bridge container",
        extra={"provider": settings.llm_provider.value, "model": settings.llm_model_name},
    )

    # Phase 1/2: extraction stack
    if settings.llm_provider == LLMProviderName.FAKE:
        extractor: StructuredExtractor = _build_test_extractor()
    else:
        extractor = build_structured_extractor(settings)

    deduplicator = FactDeduplicator(settings)
    pipeline = ExtractionPipeline(settings, extractor, deduplicator=deduplicator)

    # Phase 3: cost stack
    cost_predictor = CostPredictor(settings)
    cost_ledger = CostLedger(settings)
    arbitration_gate = ArbitrationGate(
        predictor=cost_predictor,
        settings=settings,
        extractor=FeatureExtractor(),
    )

    router = FallbackRouter(
        pipeline=pipeline,
        settings=settings,
        arbitration_gate=arbitration_gate,
        cost_ledger=cost_ledger,
    )

    logger.info("Bridge container ready (Phase 1+2+3)")
    return BridgeContainer(
        settings=settings,
        extractor=extractor,
        deduplicator=deduplicator,
        pipeline=pipeline,
        router=router,
        cost_predictor=cost_predictor,
        cost_ledger=cost_ledger,
        arbitration_gate=arbitration_gate,
    )
