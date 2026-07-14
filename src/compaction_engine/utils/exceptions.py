"""
Exception hierarchy for the Compaction Engine.

Rationale: catching bare `Exception` anywhere in this codebase is treated as
a code-review blocker. Every failure mode that can occur in production must
be representable as a typed exception so the Multi-Protocol Agent Bridge
(Phase 2) can map it deterministically onto fallback routing / retry logic.
"""

from __future__ import annotations

from typing import Any


class CompactionEngineError(Exception):
    """Base class for all engine-raised errors. Never raise this directly."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"


class ProviderError(CompactionEngineError):
    """Raised when the underlying LLM/embedding provider fails or times out."""


class ProviderRateLimitError(ProviderError):
    """Raised specifically on 429 / rate-limit responses, to drive backoff logic."""


class SchemaValidationError(CompactionEngineError):
    """
    Raised when raw LLM output cannot be coerced into the strict Pydantic
    schema (`CompressedContextState`), e.g. malformed JSON, missing required
    fields, or freeform narrative text where atomic facts were required.
    """


class ExtractionFidelityError(CompactionEngineError):
    """
    Raised when extraction technically succeeds (valid schema) but violates
    a hard fidelity invariant, e.g. zero facts extracted from a non-trivial
    dialogue, or a constraint fact silently dropped.
    """


class IntegrityValidationError(CompactionEngineError):
    """
    Raised by the Context Integrity Validator when a compressed trajectory
    fails an acceptance gate (recall, hallucination rate, constraint
    preservation) against the uncompressed reference trajectory.
    """


class ConfigurationError(CompactionEngineError):
    """Raised on invalid or missing runtime configuration."""
