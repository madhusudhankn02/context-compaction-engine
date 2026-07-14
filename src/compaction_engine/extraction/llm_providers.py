"""
LLM provider factory.

Why this exists: the problem statement requires the engine to work with
"a smaller SLM or retriever-reranker," and to be provider-agnostic at the
Multi-Protocol Agent Bridge layer. Hardcoding `ChatAnthropic(...)` inside
the pipeline would violate the Component-Store / config-driven mandate.
Instead, every provider is built behind `BaseChatModel.with_structured_output`
so the pipeline code never branches on provider identity.

Note on the FAKE provider: it exists specifically so that the extraction
pipeline and integrity validator can be exercised in CI / offline
environments with zero network access and zero API keys, returning
deterministic, schema-valid output. This is not a mock of correctness —
it is a mock of the network boundary.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from compaction_engine.config import EngineSettings, LLMProviderName
from compaction_engine.schemas.extraction_schema import CompressedContextState
from compaction_engine.utils.exceptions import ConfigurationError, ProviderError
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class StructuredExtractor(Protocol):
    """
    The only contract the extraction pipeline depends on. Any LangChain
    `BaseChatModel.with_structured_output(CompressedContextState)` instance
    satisfies this protocol, as does `FakeStructuredExtractor` below.
    """

    def invoke(self, prompt_messages: list[dict[str, str]]) -> CompressedContextState: ...


class FakeStructuredExtractor:
    """
    Deterministic, dependency-free stand-in for a real LLM call.

    Returns a pre-seeded `CompressedContextState` (or a sequence of them via
    `responses`, consumed in order) regardless of input. Used in tests and
    in any environment without LLM API access, so that pipeline
    orchestration logic can be validated independently of provider
    correctness.
    """

    def __init__(self, responses: list[CompressedContextState]) -> None:
        if not responses:
            raise ConfigurationError("FakeStructuredExtractor requires at least one response.")
        self._responses = list(responses)
        self._call_count = 0

    def invoke(self, prompt_messages: list[dict[str, str]]) -> CompressedContextState:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return self._call_count


def build_structured_extractor(settings: EngineSettings) -> StructuredExtractor:
    """
    Factory: returns a provider-specific chat model bound to
    `CompressedContextState` via structured output, selected purely by
    `settings.llm_provider`. The extraction pipeline never imports a
    provider SDK directly.
    """
    provider = settings.llm_provider

    if provider == LLMProviderName.FAKE:
        raise ConfigurationError(
            "FakeStructuredExtractor must be injected explicitly by the caller "
            "(it needs seeded responses) — it cannot be constructed by the "
            "factory alone. Pass extractor=FakeStructuredExtractor([...]) "
            "directly to the pipeline instead of relying on this factory."
        )

    if provider == LLMProviderName.ANTHROPIC:
        return _build_anthropic(settings)

    if provider == LLMProviderName.OPENAI:
        return _build_openai(settings)

    if provider == LLMProviderName.LOCAL_OLLAMA:
        return _build_local_ollama(settings)

    raise ConfigurationError(f"Unsupported llm_provider: {provider!r}")


def _build_anthropic(settings: EngineSettings) -> StructuredExtractor:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - exercised only w/o deps installed
        raise ProviderError(
            "langchain-anthropic is not installed. Run: "
            "pip install langchain-anthropic"
        ) from exc

    try:
        model = ChatAnthropic(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_request_timeout_s,
        )
        return model.with_structured_output(CompressedContextState)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed engine error
        raise ProviderError(
            f"Failed to initialize Anthropic provider: {exc}",
            context={"model_name": settings.llm_model_name},
        ) from exc


def _build_openai(settings: EngineSettings) -> StructuredExtractor:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ProviderError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    try:
        model = ChatOpenAI(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_request_timeout_s,
        )
        return model.with_structured_output(CompressedContextState)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            f"Failed to initialize OpenAI provider: {exc}",
            context={"model_name": settings.llm_model_name},
        ) from exc


def _build_local_ollama(settings: EngineSettings) -> StructuredExtractor:
    """
    Local SLM path (e.g. Llama 3.2 3B, Qwen2.5 3B, Phi-3-mini) via Ollama.
    This is the cost-optimal path the problem statement calls out
    explicitly ("a smaller SLM") — no per-token API cost, runs on a single
    GPU or even CPU for small models.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover
        raise ProviderError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    try:
        model = ChatOllama(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )
        return model.with_structured_output(CompressedContextState)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            f"Failed to initialize local Ollama provider: {exc}",
            context={"model_name": settings.llm_model_name},
        ) from exc
