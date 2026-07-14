"""
Shared text-similarity primitives.

Pulled out into its own module because both `FactDeduplicator` (extraction
side) and `ContextIntegrityValidator` (validation side) need identical
similarity semantics — duplicating this logic in two places would let them
silently drift apart, which would make validator pass/fail decisions
inconsistent with what the pipeline actually deduplicated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard similarity over token sets — the dependency-free fallback path."""
    tokens_a, tokens_b = tokenize(a), tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=True))
    norm_a = sum(x * x for x in vec_a) ** 0.5
    norm_b = sum(y * y for y in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
