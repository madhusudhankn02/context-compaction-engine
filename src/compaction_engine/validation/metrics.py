"""
Quantitative fidelity metrics for comparing a compressed trajectory against
a gold-standard (uncompressed, or human-annotated) reference trajectory.

These dataclasses are intentionally serialization-friendly (plain types
only) so Phase 4 can dump them straight into PostgreSQL / OTel spans
without a translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FactMatch:
    """One matched (or unmatched) pair between gold and compressed fact sets."""

    gold_fact_id: str | None
    compressed_fact_id: str | None
    similarity: float
    is_match: bool
    fact_type: str


@dataclass
class FidelityReport:
    """
    Full output of the Context Integrity Validator for one workflow run.

    `passed` reflects ONLY the hard acceptance gates (recall, hallucination
    rate, constraint recall) defined in `EngineSettings`. A report can have
    a low `overall_fidelity_score` and still `passed == True` if it clears
    every hard gate — the score is a soft diagnostic signal, the gates are
    the actual pass/fail contract.
    """

    workflow_id: str
    gold_fact_count: int
    compressed_fact_count: int

    true_positive_count: int        # gold facts correctly represented
    false_negative_count: int       # gold facts missing from compressed (lost info)
    false_positive_count: int       # compressed facts with no grounding in gold (hallucinations)

    constraint_recall: float        # recall restricted to FactType.CONSTRAINT
    open_question_recall: float     # recall restricted to FactType.OPEN_QUESTION

    matches: list[FactMatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def fact_recall(self) -> float:
        denom = self.true_positive_count + self.false_negative_count
        return self.true_positive_count / denom if denom else 1.0

    @property
    def fact_precision(self) -> float:
        denom = self.true_positive_count + self.false_positive_count
        return self.true_positive_count / denom if denom else 1.0

    @property
    def hallucination_rate(self) -> float:
        denom = self.true_positive_count + self.false_positive_count
        return self.false_positive_count / denom if denom else 0.0

    @property
    def overall_fidelity_score(self) -> float:
        """Harmonic-mean-style composite: punishes either low recall or
        high hallucination rate, doesn't let one offset the other."""
        precision, recall = self.fact_precision, self.fact_recall
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "gold_fact_count": self.gold_fact_count,
            "compressed_fact_count": self.compressed_fact_count,
            "true_positive_count": self.true_positive_count,
            "false_negative_count": self.false_negative_count,
            "false_positive_count": self.false_positive_count,
            "fact_recall": round(self.fact_recall, 4),
            "fact_precision": round(self.fact_precision, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "constraint_recall": round(self.constraint_recall, 4),
            "open_question_recall": round(self.open_question_recall, 4),
            "overall_fidelity_score": round(self.overall_fidelity_score, 4),
            "notes": list(self.notes),
        }
