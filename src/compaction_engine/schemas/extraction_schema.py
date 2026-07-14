"""
Strict structured-extraction schema.

This is the single most important file in Phase 1. Every downstream
guarantee in this system (no hallucinated facts, no silently-dropped
constraints, deterministic diffing for the integrity validator) depends on
extraction output being forced through *this* schema rather than freeform
LLM text.

Design principles
------------------
1. `extra="forbid"` everywhere: the model cannot smuggle in unstructured
   fields that bypass validation.
2. Facts are atomic claims, not paragraphs. We enforce this with a
   validator, not a prompt instruction alone — prompts are guidance,
   schemas are guarantees.
3. Every fact is traceable to a `source_turn_index` range. This is what
   makes the Context Integrity Validator's hallucination check possible:
   a fact with no valid grounding range is a hallucination by definition.
4. `supersedes` lets later facts explicitly invalidate earlier ones (e.g.
   "customer wants a refund" -> later -> "customer wants store credit
   instead"), which is how we avoid stale/contradictory state surviving
   compaction.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FactType(str, Enum):
    DECISION = "decision"                      # an action that was decided/taken
    CONSTRAINT = "constraint"                   # a hard rule/limit that MUST hold
    INTERMEDIATE_RESULT = "intermediate_result"  # output of a tool/agent step
    USER_PREFERENCE = "user_preference"          # stated preference, soft constraint
    EXTERNAL_FACT = "external_fact"              # retrieved/looked-up fact
    OPEN_QUESTION = "open_question"               # unresolved item, must not be culled


class ConfidenceLevel(str, Enum):
    HIGH = "high"      # explicitly, unambiguously stated by an agent or user
    MEDIUM = "medium"  # reasonably inferred, well-supported by surrounding turns
    LOW = "low"        # weakly supported — candidate for re-retrieval, not deletion


# Fact types that the system treats as non-negotiable: if the integrity
# validator finds one of these dropped during compaction, that is a hard
# failure, not a tunable fidelity-loss metric.
SAFETY_CRITICAL_FACT_TYPES: frozenset[FactType] = frozenset(
    {FactType.CONSTRAINT, FactType.OPEN_QUESTION}
)


class ExtractedFact(BaseModel):
    """A single atomic, sourced, typed claim extracted from raw dialogue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    fact_type: FactType
    statement: str = Field(
        ...,
        min_length=3,
        max_length=400,
        description="A single atomic claim. Not a summary paragraph.",
    )
    source_agent: str = Field(..., min_length=1, max_length=120)
    source_turn_start: int = Field(..., ge=0)
    source_turn_end: int = Field(..., ge=0)
    confidence: ConfidenceLevel
    supersedes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="fact_ids this fact explicitly overrides/invalidates.",
    )

    @field_validator("statement")
    @classmethod
    def reject_narrative_style(cls, v: str) -> str:
        v = v.strip()
        # Crude but effective heuristic guard: atomic facts read like a
        # single clause, not a multi-sentence narrative. This is a backstop
        # against the model "cheating" the schema by stuffing a paragraph
        # into one field — the prompt forbids it, the schema enforces it.
        if v.count(". ") > 1 or v.count("\n") > 0:
            raise ValueError(
                "statement must be a single atomic claim, not a multi-sentence "
                "narrative. Split into multiple ExtractedFact entries instead."
            )
        return v

    @model_validator(mode="after")
    def turn_range_must_be_valid(self) -> "ExtractedFact":
        if self.source_turn_end < self.source_turn_start:
            raise ValueError(
                f"source_turn_end ({self.source_turn_end}) cannot precede "
                f"source_turn_start ({self.source_turn_start})"
            )
        return self

    @property
    def is_safety_critical(self) -> bool:
        return self.fact_type in SAFETY_CRITICAL_FACT_TYPES


class CompressedContextState(BaseModel):
    """
    The compacted state tensor handed to the next agent in the pipeline.
    This — and only this — is what crosses the Multi-Protocol Agent Bridge
    as "compressed state." It is JSON-schema-validated on every hop.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(..., min_length=1)
    compaction_version: int = Field(..., ge=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_turn_count: int = Field(..., ge=0)
    facts: tuple[ExtractedFact, ...]
    unresolved_questions: tuple[str, ...] = Field(default_factory=tuple)
    token_count_estimate: int = Field(..., ge=0)

    @model_validator(mode="after")
    def non_trivial_dialogue_must_yield_facts(self) -> "CompressedContextState":
        if self.source_turn_count > 0 and len(self.facts) == 0:
            raise ValueError(
                "compaction collapsed a non-empty dialogue into zero facts — "
                "this is almost certainly an extraction failure, not a "
                "legitimate compression. Refusing to produce empty state."
            )
        return self

    @model_validator(mode="after")
    def fact_ids_must_be_unique(self) -> "CompressedContextState":
        ids = [f.fact_id for f in self.facts]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate fact_id(s) detected: {dupes}")
        return self

    def active_facts(self) -> tuple[ExtractedFact, ...]:
        """Facts not superseded by a later fact. This is the 'live' state view."""
        superseded_ids: set[str] = set()
        for f in self.facts:
            superseded_ids.update(f.supersedes)
        return tuple(f for f in self.facts if f.fact_id not in superseded_ids)

    def safety_critical_facts(self) -> tuple[ExtractedFact, ...]:
        return tuple(f for f in self.active_facts() if f.is_safety_critical)


class RawDialogueTurn(BaseModel):
    """One turn of raw, uncompressed agent/user dialogue — the extraction input."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(..., ge=0)
    agent_name: str = Field(..., min_length=1, max_length=120)
    role: str = Field(..., min_length=1, max_length=40)  # e.g. "user", "agent", "tool"
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RawDialogueChunk(BaseModel):
    """A contiguous window of turns handed to the extractor in one LLM call."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    chunk_index: int = Field(..., ge=0)
    turns: tuple[RawDialogueTurn, ...] = Field(..., min_length=1)

    @property
    def turn_index_range(self) -> tuple[int, int]:
        indices = [t.turn_index for t in self.turns]
        return (min(indices), max(indices))
