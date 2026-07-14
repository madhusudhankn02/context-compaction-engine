"""
Context Integrity Validator.

This is Phase 1's third deliverable: an automated framework that compares
a compressed trajectory against a gold-standard/uncompressed reference and
decides, deterministically, whether the compression was acceptable.

Matching strategy
------------------
Facts are matched gold<->compressed within the same `fact_type` bucket by
semantic similarity (delegates to the same similarity backend as
`FactDeduplicator`, so embedding vs. lexical fallback behavior is
consistent across the codebase). Matching is solved as a greedy
highest-similarity-first assignment, which is sufficient at the
per-workflow fact-count scale this system operates at (tens, not millions,
of facts) and keeps the validator dependency-free of an LP solver.

Hard gates vs. soft score
-------------------------
`validate()` raises `IntegrityValidationError` when a *hard* gate from
`EngineSettings` is violated (recall floor, hallucination ceiling,
constraint-recall floor). It does NOT raise on a merely low
`overall_fidelity_score` — that's a diagnostic for the benchmark suite
(Phase 5), not a pipeline-blocking failure.
"""

from __future__ import annotations

from compaction_engine.config import EngineSettings
from compaction_engine.schemas.extraction_schema import CompressedContextState, ExtractedFact
from compaction_engine.utils.exceptions import IntegrityValidationError
from compaction_engine.utils.logging_config import get_logger
from compaction_engine.utils.similarity import cosine_similarity, lexical_similarity
from compaction_engine.validation.metrics import FactMatch, FidelityReport

logger = get_logger(__name__)


class ContextIntegrityValidator:
    def __init__(self, settings: EngineSettings, embedder: object | None = None) -> None:
        self._settings = settings
        self._embedder = embedder

    def _similarity(self, a: ExtractedFact, b: ExtractedFact) -> float:
        if self._embedder is not None:
            try:
                vectors = self._embedder.encode([a.statement, b.statement])
                return cosine_similarity(vectors[0], vectors[1])
            except Exception:  # noqa: BLE001
                logger.exception("Embedder failed during validation; using lexical fallback.")
        return lexical_similarity(a.statement, b.statement)

    def _match_bucket(
        self, gold: list[ExtractedFact], compressed: list[ExtractedFact], threshold: float
    ) -> tuple[list[FactMatch], list[ExtractedFact], list[ExtractedFact]]:
        """Greedy highest-similarity-first matching within one fact_type bucket.
        Returns (matches, unmatched_gold, unmatched_compressed)."""
        pairs: list[tuple[float, ExtractedFact, ExtractedFact]] = []
        for g in gold:
            for c in compressed:
                pairs.append((self._similarity(g, c), g, c))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_gold_ids: set[str] = set()
        matched_compressed_ids: set[str] = set()
        matches: list[FactMatch] = []

        for similarity, g, c in pairs:
            if g.fact_id in matched_gold_ids or c.fact_id in matched_compressed_ids:
                continue
            if similarity < threshold:
                continue
            matched_gold_ids.add(g.fact_id)
            matched_compressed_ids.add(c.fact_id)
            matches.append(
                FactMatch(
                    gold_fact_id=g.fact_id,
                    compressed_fact_id=c.fact_id,
                    similarity=similarity,
                    is_match=True,
                    fact_type=g.fact_type.value,
                )
            )

        unmatched_gold = [g for g in gold if g.fact_id not in matched_gold_ids]
        unmatched_compressed = [c for c in compressed if c.fact_id not in matched_compressed_ids]
        return matches, unmatched_gold, unmatched_compressed

    def compare(
        self,
        gold_standard: CompressedContextState,
        compressed: CompressedContextState,
        *,
        match_threshold: float = 0.75,
    ) -> FidelityReport:
        """Pure comparison — no raising. Use `validate()` if you want hard-gate
        enforcement; use `compare()` directly for benchmark-suite scoring
        (Phase 5) where you want the report regardless of pass/fail."""
        gold_facts = list(gold_standard.active_facts())
        compressed_facts = list(compressed.active_facts())

        all_matches: list[FactMatch] = []
        all_unmatched_gold: list[ExtractedFact] = []
        all_unmatched_compressed: list[ExtractedFact] = []

        fact_types = {f.fact_type for f in gold_facts} | {f.fact_type for f in compressed_facts}
        for fact_type in fact_types:
            gold_bucket = [f for f in gold_facts if f.fact_type == fact_type]
            compressed_bucket = [f for f in compressed_facts if f.fact_type == fact_type]
            matches, unmatched_gold, unmatched_compressed = self._match_bucket(
                gold_bucket, compressed_bucket, match_threshold
            )
            all_matches.extend(matches)
            all_unmatched_gold.extend(unmatched_gold)
            all_unmatched_compressed.extend(unmatched_compressed)

        true_positives = len(all_matches)
        false_negatives = len(all_unmatched_gold)
        false_positives = len(all_unmatched_compressed)

        constraint_recall = self._bucket_recall(all_matches, all_unmatched_gold, "constraint")
        open_question_recall = self._bucket_recall(
            all_matches, all_unmatched_gold, "open_question"
        )

        notes: list[str] = []
        if all_unmatched_gold:
            dropped_types = sorted({f.fact_type.value for f in all_unmatched_gold})
            notes.append(f"Dropped gold facts of types: {dropped_types}")
        if all_unmatched_compressed:
            extra_types = sorted({f.fact_type.value for f in all_unmatched_compressed})
            notes.append(f"Unmatched/possibly-hallucinated compressed facts of types: {extra_types}")

        return FidelityReport(
            workflow_id=compressed.workflow_id,
            gold_fact_count=len(gold_facts),
            compressed_fact_count=len(compressed_facts),
            true_positive_count=true_positives,
            false_negative_count=false_negatives,
            false_positive_count=false_positives,
            constraint_recall=constraint_recall,
            open_question_recall=open_question_recall,
            matches=all_matches,
            notes=notes,
        )

    @staticmethod
    def _bucket_recall(
        matches: list[FactMatch], unmatched_gold: list[ExtractedFact], fact_type: str
    ) -> float:
        tp = sum(1 for m in matches if m.fact_type == fact_type)
        fn = sum(1 for f in unmatched_gold if f.fact_type.value == fact_type)
        denom = tp + fn
        return tp / denom if denom else 1.0

    def validate(
        self,
        gold_standard: CompressedContextState,
        compressed: CompressedContextState,
        *,
        match_threshold: float = 0.75,
    ) -> FidelityReport:
        """Same as `compare()`, but raises `IntegrityValidationError` if any
        hard gate from `EngineSettings` is violated. This is the entry point
        the orchestrator (Phase 2's bridge) should call before allowing a
        compressed state to replace the uncompressed one on the wire."""
        report = self.compare(gold_standard, compressed, match_threshold=match_threshold)
        s = self._settings
        violations: list[str] = []

        if report.fact_recall < s.min_acceptable_fact_recall:
            violations.append(
                f"fact_recall {report.fact_recall:.4f} < floor {s.min_acceptable_fact_recall}"
            )
        if report.hallucination_rate > s.max_acceptable_hallucination_rate:
            violations.append(
                f"hallucination_rate {report.hallucination_rate:.4f} > ceiling "
                f"{s.max_acceptable_hallucination_rate}"
            )
        if report.constraint_recall < s.min_acceptable_constraint_recall:
            violations.append(
                f"constraint_recall {report.constraint_recall:.4f} < floor "
                f"{s.min_acceptable_constraint_recall}"
            )
        if report.open_question_recall < 1.0:
            # Open questions are exempt from being a *tunable* gate — losing
            # even one is always a hard failure, by design.
            violations.append(
                f"open_question_recall {report.open_question_recall:.4f} < required 1.0"
            )

        if violations:
            raise IntegrityValidationError(
                f"Compressed state for workflow {compressed.workflow_id!r} failed "
                f"integrity validation: {'; '.join(violations)}",
                context={"workflow_id": compressed.workflow_id, "report": report.to_dict()},
            )

        return report
