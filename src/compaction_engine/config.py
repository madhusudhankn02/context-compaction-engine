"""
Centralized, environment-driven configuration for the Compaction Engine.

Design rationale
-----------------
Per the GitOps / Component-Store mandate, NOTHING here is hardcoded into
business logic. Every tunable (model name, thresholds, chunk sizes) is a
config field with a sane default that can be overridden via environment
variables or a `.env` file, so the same code image runs differently in
dev/staging/prod purely through config injection.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL_OLLAMA = "local_ollama"
    FAKE = "fake"  # deterministic stub provider, used in tests / CI


class EngineSettings(BaseSettings):
    """
    Single source of truth for runtime configuration.

    All fields are overridable via env vars prefixed `COMPACTION_`, e.g.
    `COMPACTION_LLM_PROVIDER=anthropic`.
    """

    model_config = SettingsConfigDict(
        env_prefix="COMPACTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider (extraction model) ---
    llm_provider: LLMProviderName = LLMProviderName.ANTHROPIC
    llm_model_name: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_request_timeout_s: float = Field(default=30.0, gt=0.0)

    # --- Chunking / extraction behavior ---
    max_turns_per_chunk: int = Field(default=8, ge=1, le=64)
    max_facts_per_chunk: int = Field(default=20, ge=1, le=200)

    # --- Deduplication / reranker ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dedup_cosine_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    # --- Integrity validation thresholds (Phase 1 acceptance gates) ---
    min_acceptable_fact_recall: float = Field(default=0.90, ge=0.0, le=1.0)
    max_acceptable_hallucination_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    min_acceptable_constraint_recall: float = Field(default=0.99, ge=0.0, le=1.0)

    # --- Observability ---
    log_level: str = "INFO"
    enable_json_logging: bool = True

    @field_validator("max_facts_per_chunk")
    @classmethod
    def facts_must_not_exceed_turns_budget(cls, v: int, info) -> int:
        max_turns = info.data.get("max_turns_per_chunk", 8)
        if v > max_turns * 10:
            raise ValueError(
                f"max_facts_per_chunk={v} is unrealistic for "
                f"max_turns_per_chunk={max_turns}; check for misconfiguration."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> EngineSettings:
    """Cached settings accessor. Import this, never instantiate EngineSettings directly."""
    return EngineSettings()
