"""
Semantic deduplication ("retriever-reranker") for extracted facts.

After per-chunk extraction, the same fact is frequently re-stated across
chunks (agents repeat constraints, restate decisions for confirmation,
etc.). This module removes near-duplicates by cosine similarity over
sentence embeddings, keeping the most recent / highest-confidence
occurrence and recording the rest as superseded-by-duplication.

A dependency-free fallback (`lexical_similarity`) is provided so this
module degrades gracefully — not silently no-ops — in environments without
`sentence-transformers` installed (e.g. constrained edge deployments).
"""

from __future__ import annotations

from collections.abc import Sequence

from compaction_engine.config import EngineSettings
from compaction_engine.schemas.extraction_schema import ConfidenceLevel, ExtractedFact
from compaction_engine.utils.logging_config import get_logger
from compaction_engine.utils.similarity import cosine_similarity, lexical_similarity

logger = get_logger(__name__)

_CONFIDENCE_RANK: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.HIGH: 2,
}


class FactDeduplicator:
    """
    Deduplicates a list of `ExtractedFact` using embeddings when available,
    falling back to lexical (Jaccard) similarity otherwise. Never raises on
    missing dependencies — degrades, logs loudly, and continues, because a
    failed dedup pass must not block the compaction pipeline (fail-open on
    a non-safety-critical optimization, fail-closed on schema validity).
    """

    def __init__(self, settings: EngineSettings, embedder: object | None = None) -> None:
        self._settings = settings
        self._embedder = embedder if embedder is not None else self._try_load_default_embedder()

    def _try_load_default_embedder(self) -> object | None:
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self._settings.embedding_model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; FactDeduplicator falling "
                "back to lexical similarity. Install sentence-transformers for "
                "higher-quality dedup.",
            )
            return None

    def _similarity(self, a: ExtractedFact, b: ExtractedFact) -> float:
        if self._embedder is not None:
            try:
                vectors = self._embedder.encode([a.statement, b.statement])
                return cosine_similarity(vectors[0], vectors[1])
            except Exception:  # noqa: BLE001 - embedder failure must not crash the pipeline
                logger.exception("Embedder failed mid-dedup; falling back to lexical similarity.")
        return lexical_similarity(a.statement, b.statement)

    def deduplicate(self, facts: Sequence[ExtractedFact]) -> tuple[ExtractedFact, ...]:
        """
        Returns a de-duplicated tuple of facts. Among a near-duplicate
        cluster, the fact with the highest confidence (ties broken by most
        recent `source_turn_end`) is kept; the rest are dropped, NOT marked
        superseded, because deduplication is not the same claim as a state
        change — it is the same claim re-stated. Constraints and
        open_questions are exempt from dedup-by-similarity entirely: losing
        one due to a false-positive similarity match is an unacceptable
        risk relative to the storage savings.
        """
        threshold = self._settings.dedup_cosine_threshold
        protected = [f for f in facts if f.is_safety_critical]
        candidates = [f for f in facts if not f.is_safety_critical]

        kept: list[ExtractedFact] = []
        for fact in candidates:
            duplicate_of: ExtractedFact | None = None
            for existing in kept:
                if existing.fact_type != fact.fact_type:
                    continue
                if self._similarity(existing, fact) >= threshold:
                    duplicate_of = existing
                    break

            if duplicate_of is None:
                kept.append(fact)
                continue

            should_replace = (
                _CONFIDENCE_RANK[fact.confidence] > _CONFIDENCE_RANK[duplicate_of.confidence]
            ) or (
                _CONFIDENCE_RANK[fact.confidence] == _CONFIDENCE_RANK[duplicate_of.confidence]
                and fact.source_turn_end > duplicate_of.source_turn_end
            )
            if should_replace:
                kept.remove(duplicate_of)
                kept.append(fact)
            logger.info(
                "Deduplicated near-duplicate fact",
                extra={
                    "kept_fact_id": (duplicate_of if not should_replace else fact).fact_id,
                    "dropped_fact_id": (fact if not should_replace else duplicate_of).fact_id,
                    "fact_type": fact.fact_type.value,
                },
            )

        return tuple(protected) + tuple(kept)
