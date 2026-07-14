# Context-Aware State Compaction Engine — Phase 1

**Semantic Context Summarization Module**: structured extraction +
deduplication + Context Integrity Validator. This is the foundation every
later phase (Multi-Protocol Bridge, Cost Arbitration, Observability,
Benchmark Suite) builds on — it is the only place in the system where raw
agent dialogue is converted into the typed `CompressedContextState` that
flows everywhere else.

## Why the architecture looks like this

The central design bet is: **schemas are guarantees, prompts are hints.**
An LLM can be instructed not to hallucinate or not to write narrative
summaries, but instructions are not enforcement. Every fact that survives
into a `CompressedContextState` has been forced through:

1. A strict Pydantic schema (`extra="forbid"`, atomic-statement validator,
   valid turn-range validator, non-empty-output validator).
2. A grounding check (`ExtractionPipeline._validate_chunk_grounding`) that
   rejects any fact whose cited turn range falls outside the chunk it
   claims to come from — a hallucinated citation is a hard pipeline
   failure, not a logged warning.
3. (For compaction acceptance) The `ContextIntegrityValidator`, which
   diffs the compressed trajectory against a gold-standard/uncompressed
   reference and enforces hard gates on recall, hallucination rate, and
   — non-negotiably — 100% retention of `CONSTRAINT` and `OPEN_QUESTION`
   facts.

## Directory layout

```
context-compaction-engine/
├── pyproject.toml          # packaging + ruff/mypy config
├── requirements.txt
├── .env.example
├── Makefile
├── src/compaction_engine/
│   ├── config.py            # EngineSettings (pydantic-settings, env-driven)
│   ├── schemas/
│   │   └── extraction_schema.py   # ExtractedFact, CompressedContextState, RawDialogueTurn/Chunk
│   ├── extraction/
│   │   ├── llm_providers.py # provider factory (Anthropic/OpenAI/Ollama/Fake)
│   │   ├── prompts.py       # system + extraction prompt templates
│   │   ├── reranker.py      # FactDeduplicator (embedding or lexical fallback)
│   │   └── pipeline.py      # ExtractionPipeline: chunk -> extract -> dedup -> merge
│   ├── validation/
│   │   ├── metrics.py       # FidelityReport, FactMatch dataclasses
│   │   └── integrity_validator.py  # ContextIntegrityValidator
│   └── utils/
│       ├── exceptions.py    # typed exception hierarchy
│       ├── logging_config.py # JSON structured logging
│       └── similarity.py    # shared cosine/lexical similarity (used by both
│                             # reranker.py and integrity_validator.py so they
│                             # never silently drift apart)
├── tests/
│   ├── conftest.py          # fixtures: settings, sample dialogue, gold-standard state
│   ├── test_extraction_pipeline.py
│   └── test_integrity_validator.py
└── data/sample_dialogues/
    └── customer_support_escalation.json   # first of the 50+ benchmark workflows (Phase 5)
```

## Data flow (Phase 1 only)

```
RawDialogueTurn[]  (ordered, from agent transcripts)
        │
        ▼
  chunk_turns()  ──────────────► RawDialogueChunk[]  (sliding window, max_turns_per_chunk)
        │
        ▼  (per chunk, in order — carry-forward summary of prior facts included in prompt)
  StructuredExtractor.invoke()  ──────────────► CompressedContextState (per-chunk)
        │
        ▼
  _validate_chunk_grounding()   ──► raises ExtractionFidelityError on out-of-range citations
        │
        ▼  (facts merged across all chunks)
  FactDeduplicator.deduplicate() ─► drops near-duplicates; CONSTRAINT/OPEN_QUESTION exempt
        │
        ▼
  Final CompressedContextState  (schema-validated, this is what crosses the wire in Phase 2)
        │
        ▼ (offline / CI, against a gold-standard reference)
  ContextIntegrityValidator.validate() ─► FidelityReport, or raises IntegrityValidationError
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic,embeddings]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY, adjust thresholds
```

## Running

```python
from compaction_engine.config import get_settings
from compaction_engine.extraction.llm_providers import build_structured_extractor
from compaction_engine.extraction.pipeline import ExtractionPipeline
from compaction_engine.schemas.extraction_schema import RawDialogueTurn

settings = get_settings()  # reads .env
extractor = build_structured_extractor(settings)  # real Anthropic/OpenAI/Ollama call
pipeline = ExtractionPipeline(settings, extractor)

turns = [RawDialogueTurn(turn_index=0, agent_name="user", role="user", content="...")]
result = pipeline.run(workflow_id="wf-001", turns=turns)

print(result.final_state.model_dump_json(indent=2))
print(f"dedup ratio: {result.dedup_ratio:.2%}")
```

## A note on this repo's test strategy

Every test in `tests/` runs against `FakeStructuredExtractor` — a seeded,
deterministic stand-in for a real LLM call (see `llm_providers.py`). This
means:

- Tests validate **pipeline orchestration correctness** (chunking math,
  grounding enforcement, dedup wiring, schema rejection paths) completely
  independent of any specific model's extraction *quality*.
- Tests run with zero API keys, zero network access, in any CI runner.
- Model/extraction *quality* is a separate concern, owned by the Phase 5
  benchmark suite, which will run the same pipeline against a real
  provider over the 50+-workflow dataset and score it with
  `ContextIntegrityValidator.compare()`.

## What's intentionally NOT in Phase 1

- LangGraph wiring (graph nodes/edges) — Phase 1 ships the extraction
  *logic* as a plain, composable class (`ExtractionPipeline`) precisely so
  it can be dropped into a LangGraph node, a Celery task, or an MCP tool
  handler in Phase 2 without modification. Wrapping it in a graph now would
  couple a Phase-1 concern to a Phase-2 decision.
- Real embedding-model wiring by default — `FactDeduplicator` and
  `ContextIntegrityValidator` both accept an `embedder` and fall back to
  lexical (Jaccard) similarity if none is supplied or
  `sentence-transformers` isn't installed, so Phase 1 has zero hard
  dependency on a model download.
- Cost numbers — Phase 3 owns cost prediction; Phase 1 only produces the
  `token_count_estimate` field cost prediction will consume as a feature.
