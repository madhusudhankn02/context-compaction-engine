"""
Semantic Context Summarization pipeline.

This is the orchestrator that the rest of Phase 1 hangs off of:

    RawDialogueTurn[]
        -> chunk into RawDialogueChunk[]            (sliding window, ordered)
        -> per chunk: StructuredExtractor.invoke()    -> CompressedContextState
        -> merge facts across chunks
        -> FactDeduplicator.deduplicate()
        -> final, schema-valid CompressedContextState

Every step that can fail raises a typed exception from
`compaction_engine.utils.exceptions` — nothing here ever returns `None`
on failure or swallows an error silently. That property is what lets
Phase 2's fallback routing make a deterministic decision (escalate to
uncompressed state) when this pipeline fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compaction_engine.config import EngineSettings
from compaction_engine.extraction.llm_providers import StructuredExtractor
from compaction_engine.extraction.prompts import build_extraction_messages
from compaction_engine.extraction.reranker import FactDeduplicator
from compaction_engine.schemas.extraction_schema import (
    CompressedContextState,
    ExtractedFact,
    RawDialogueChunk,
    RawDialogueTurn,
)
from compaction_engine.utils.exceptions import ExtractionFidelityError, SchemaValidationError
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free heuristic (~4 chars/token). Swap for a real
    tokenizer (tiktoken / anthropic.count_tokens) once a provider is wired in."""
    return max(1, len(text) // 4)


def chunk_turns(
    turns: list[RawDialogueTurn], *, max_turns_per_chunk: int
) -> list[RawDialogueChunk]:
    """Splits ordered turns into fixed-size, non-overlapping chunks.

    Turns must already be sorted by `turn_index`; this function enforces
    that invariant rather than silently re-sorting (re-sorting could mask
    an upstream ordering bug)."""
    if not turns:
        return []

    for prev, curr in zip(turns, turns[1:], strict=False):
        if curr.turn_index <= prev.turn_index:
            raise ValueError(
                f"turns must be strictly increasing by turn_index; got "
                f"{prev.turn_index} followed by {curr.turn_index}"
            )

    workflow_id = "__pending__"  # overwritten by ExtractionPipeline._chunk via model_copy
    chunks: list[RawDialogueChunk] = []
    for chunk_index, start in enumerate(range(0, len(turns), max_turns_per_chunk)):
        window = turns[start : start + max_turns_per_chunk]
        chunks.append(
            RawDialogueChunk(workflow_id=workflow_id, chunk_index=chunk_index, turns=tuple(window))
        )
    return chunks


@dataclass
class ExtractionRunResult:
    """Full audit trail of one extraction run — this is what Phase 4's
    observability layer will persist verbatim."""

    final_state: CompressedContextState
    chunks_processed: int
    raw_turn_count: int
    facts_before_dedup: int
    facts_after_dedup: int
    provider_call_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def dedup_ratio(self) -> float:
        if self.facts_before_dedup == 0:
            return 0.0
        return 1.0 - (self.facts_after_dedup / self.facts_before_dedup)


class ExtractionPipeline:
    """
    Stateless w.r.t. workflow data (safe to reuse across workflows); holds
    only injected dependencies (settings, extractor, deduplicator), per the
    Component-Store mandate — every dependency is swappable at construction
    time, nothing is reached for via global state.
    """

    def __init__(
        self,
        settings: EngineSettings,
        extractor: StructuredExtractor,
        deduplicator: FactDeduplicator | None = None,
    ) -> None:
        self._settings = settings
        self._extractor = extractor
        self._deduplicator = deduplicator or FactDeduplicator(settings)

    def run(self, workflow_id: str, turns: list[RawDialogueTurn]) -> ExtractionRunResult:
        if not turns:
            raise ValueError("Cannot run extraction on an empty turn list.")

        ordered = sorted(turns, key=lambda t: t.turn_index)
        raw_chunks = self._chunk(workflow_id, ordered)

        all_facts: list[ExtractedFact] = []
        warnings: list[str] = []
        cumulative_turns = 0

        for chunk in raw_chunks:
            cumulative_turns += len(chunk.turns)
            carry_forward = self._summarize_carry_forward(all_facts)
            messages = build_extraction_messages(
                chunk,
                max_facts=self._settings.max_facts_per_chunk,
                cumulative_turn_count=cumulative_turns,
                carry_forward_summary=carry_forward,
            )

            try:
                chunk_state = self._extractor.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - normalized into a typed engine error
                raise SchemaValidationError(
                    f"Structured extraction failed on chunk {chunk.chunk_index} "
                    f"for workflow {workflow_id!r}: {exc}",
                    context={"chunk_index": chunk.chunk_index, "workflow_id": workflow_id},
                ) from exc

            self._validate_chunk_grounding(chunk, chunk_state, warnings)
            all_facts.extend(chunk_state.facts)

        facts_before_dedup = len(all_facts)
        deduped_facts = self._deduplicator.deduplicate(all_facts)

        full_text = "\n".join(t.content for t in ordered)
        final_state = CompressedContextState(
            workflow_id=workflow_id,
            compaction_version=1,
            source_turn_count=len(ordered),
            facts=deduped_facts,
            unresolved_questions=tuple(
                f.statement for f in deduped_facts if f.fact_type.value == "open_question"
            ),
            token_count_estimate=_estimate_tokens(
                "\n".join(f.statement for f in deduped_facts)
            ),
        )

        if final_state.token_count_estimate >= _estimate_tokens(full_text):
            warnings.append(
                "Compressed token estimate is not smaller than the raw dialogue "
                "estimate — compaction provided no savings for this workflow."
            )

        return ExtractionRunResult(
            final_state=final_state,
            chunks_processed=len(raw_chunks),
            raw_turn_count=len(ordered),
            facts_before_dedup=facts_before_dedup,
            facts_after_dedup=len(deduped_facts),
            provider_call_count=len(raw_chunks),
            warnings=warnings,
        )

    def _chunk(self, workflow_id: str, turns: list[RawDialogueTurn]) -> list[RawDialogueChunk]:
        chunks = chunk_turns(turns, max_turns_per_chunk=self._settings.max_turns_per_chunk)
        # `chunk_turns` doesn't know the real workflow_id (kept decoupled/testable
        # in isolation); stamp it here.
        return [c.model_copy(update={"workflow_id": workflow_id}) for c in chunks]

    @staticmethod
    def _summarize_carry_forward(facts_so_far: list[ExtractedFact]) -> str:
        if not facts_so_far:
            return "(none — first chunk)"
        lines = [f"- [{f.fact_id}] ({f.fact_type.value}) {f.statement}" for f in facts_so_far[-15:]]
        return "\n".join(lines)

    @staticmethod
    def _validate_chunk_grounding(
        chunk: RawDialogueChunk,
        chunk_state: CompressedContextState,
        warnings: list[str],
    ) -> None:
        """
        Hard grounding check: every fact's cited turn range must fall
        inside the turns we actually sent for this chunk. A fact citing an
        out-of-range turn index is, by construction, ungrounded — and is
        rejected outright rather than merely logged, because it indicates
        the model fabricated a citation.
        """
        valid_start, valid_end = chunk.turn_index_range
        for fact in chunk_state.facts:
            if fact.source_turn_start < valid_start or fact.source_turn_end > valid_end:
                raise ExtractionFidelityError(
                    f"Fact {fact.fact_id!r} cites turn range "
                    f"({fact.source_turn_start}, {fact.source_turn_end}) outside "
                    f"the chunk's valid range ({valid_start}, {valid_end}) — "
                    f"rejecting as an ungrounded / hallucinated citation.",
                    context={"chunk_index": chunk.chunk_index, "fact_id": fact.fact_id},
                )
        if chunk_state.facts and chunk_state.token_count_estimate == 0:
            warnings.append(f"Chunk {chunk.chunk_index} returned zero token estimate.")
