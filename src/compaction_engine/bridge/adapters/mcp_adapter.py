"""
MCP (Model Context Protocol) adapter.

Wraps the FallbackRouter as a first-class MCP Tool so any MCP-compliant
orchestrator (Claude, LangGraph with MCP support, etc.) can call
`compress_context` as a tool without knowing anything about the underlying
pipeline. The orchestrator sees the same interface it uses for web_search
or code_execution — compaction is just another tool.

Architecture note: MCP tools communicate via JSON-Schema-typed input/output.
We reuse `CompressedContextState.model_json_schema()` as the output schema so
the orchestrator gets machine-readable documentation of every field it will
receive — this is what makes the context payload inspectable for
observability (Phase 4) without extra parsing.

This module has no dependency on the MCP SDK itself — it only produces the
tool definition dict and a handler callable, which the SDK (or a FastAPI
endpoint) can register. This keeps it testable without an MCP runtime.
"""

from __future__ import annotations

import json
from typing import Any

from compaction_engine.bridge.container import BridgeContainer
from compaction_engine.schemas.extraction_schema import CompressedContextState, RawDialogueTurn
from compaction_engine.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Tool definition (MCP JSON-Schema format) ──────────────────────────────────

MCP_COMPRESS_TOOL: dict[str, Any] = {
    "name": "compress_context",
    "description": (
        "Compresses a raw multi-agent dialogue history into a structured, "
        "schema-validated CompressedContextState. Eliminates redundant turns, "
        "deduplicates near-identical facts, and preserves all constraints and "
        "open questions with 100% recall. Returns the compressed state plus "
        "routing metadata. Falls back transparently to a passthrough state if "
        "compression fails — the caller always receives a valid state."
    ),
    "input_schema": {
        "type": "object",
        "required": ["workflow_id", "turns"],
        "additionalProperties": False,
        "properties": {
            "workflow_id": {
                "type": "string",
                "minLength": 1,
                "description": "Unique identifier for this workflow run.",
            },
            "turns": {
                "type": "array",
                "minItems": 1,
                "description": "Ordered list of raw dialogue turns to compress.",
                "items": {
                    "type": "object",
                    "required": ["turn_index", "agent_name", "role", "content"],
                    "additionalProperties": False,
                    "properties": {
                        "turn_index": {"type": "integer", "minimum": 0},
                        "agent_name": {"type": "string", "minLength": 1},
                        "role": {
                            "type": "string",
                            "enum": ["user", "agent", "tool", "system"],
                        },
                        "content": {"type": "string", "minLength": 1},
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO 8601 timestamp; optional.",
                        },
                    },
                },
            },
        },
    },
}

MCP_VALIDATE_TOOL: dict[str, Any] = {
    "name": "validate_compression",
    "description": (
        "Validates a compressed state against a gold-standard reference "
        "trajectory. Returns fact recall, hallucination rate, constraint recall, "
        "and an overall fidelity score. Raises if any hard acceptance gate is "
        "violated (constraint recall < 0.99, hallucination rate > 0.02, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "required": ["gold_standard", "compressed"],
        "additionalProperties": False,
        "properties": {
            "gold_standard": {
                "type": "object",
                "description": "JSON-serialized CompressedContextState as the reference.",
            },
            "compressed": {
                "type": "object",
                "description": "JSON-serialized CompressedContextState to evaluate.",
            },
            "match_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.75,
                "description": "Similarity threshold for fact matching (0–1).",
            },
        },
    },
}

ALL_MCP_TOOLS: list[dict[str, Any]] = [MCP_COMPRESS_TOOL, MCP_VALIDATE_TOOL]


# ── MCP tool handler ──────────────────────────────────────────────────────────

class MCPAdapter:
    """
    Stateless handler class. Register one instance per process and call
    `dispatch(tool_name, tool_input)` from your MCP server's tool-call hook.

    Example (pseudo-code for any MCP SDK):

        adapter = MCPAdapter(container)

        @mcp_server.tool_call_handler
        def handle(tool_name: str, tool_input: dict) -> dict:
            return adapter.dispatch(tool_name, tool_input)
    """

    def __init__(self, container: BridgeContainer) -> None:
        self._container = container
        self._dispatch_map = {
            "compress_context": self._compress,
            "validate_compression": self._validate,
        }

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = self._dispatch_map.get(tool_name)
        if handler is None:
            return {
                "error": f"Unknown tool: {tool_name!r}",
                "available_tools": list(self._dispatch_map),
            }
        try:
            return handler(tool_input)
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool dispatch failed", extra={"tool": tool_name})
            return {"error": str(exc), "error_type": type(exc).__name__, "tool": tool_name}

    def _compress(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        workflow_id: str = tool_input["workflow_id"]
        raw_turns = tool_input["turns"]

        turns: list[RawDialogueTurn] = []
        for t in raw_turns:
            kwargs: dict[str, Any] = {
                "turn_index": t["turn_index"],
                "agent_name": t["agent_name"],
                "role": t["role"],
                "content": t["content"],
            }
            if "timestamp" in t and t["timestamp"]:
                from datetime import datetime, timezone
                kwargs["timestamp"] = datetime.fromisoformat(t["timestamp"]).replace(
                    tzinfo=timezone.utc
                )
            turns.append(RawDialogueTurn(**kwargs))

        result = self._container.router.compress(workflow_id, turns)

        logger.info("MCP compress_context called", extra=result.to_audit_dict())

        return {
            "workflow_id": result.state.workflow_id,
            "route": result.route.value,
            "compression_ratio": round(result.compression_ratio, 4),
            "latency_ms": round(result.latency_ms, 2),
            "fact_count": len(result.state.facts),
            "warnings": result.warnings,
            "fallback_reason": result.fallback_reason,
            # Full typed state for the orchestrator to pass to the next agent
            "state": json.loads(result.state.model_dump_json()),
        }

    def _validate(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        from compaction_engine.utils.exceptions import IntegrityValidationError
        from compaction_engine.validation.integrity_validator import ContextIntegrityValidator

        gold = CompressedContextState.model_validate(tool_input["gold_standard"])
        compressed = CompressedContextState.model_validate(tool_input["compressed"])
        threshold = float(tool_input.get("match_threshold", 0.75))

        validator = ContextIntegrityValidator(self._container.settings)
        try:
            report = validator.validate(gold, compressed, match_threshold=threshold)
            report_dict = report.to_dict()
            passed = True
        except IntegrityValidationError as exc:
            passed = False
            raw = exc.context.get("report", {})
            report_dict = raw if isinstance(raw, dict) else raw.to_dict()

        return {"passed": passed, "report": report_dict}
