from __future__ import annotations

import json

import pytest

from compaction_engine.bridge.adapters.mcp_adapter import ALL_MCP_TOOLS, MCPAdapter


class TestMCPToolDefinitions:
    def test_all_tools_have_required_keys(self):
        for tool in ALL_MCP_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_compress_tool_input_schema_is_valid_json_schema(self):
        from compaction_engine.bridge.adapters.mcp_adapter import MCP_COMPRESS_TOOL
        schema = MCP_COMPRESS_TOOL["input_schema"]
        required = set(schema.get("required", []))
        assert "workflow_id" in required
        assert "turns" in required
        assert schema.get("additionalProperties") is False


class TestMCPAdapterDispatch:
    def _turns_payload(self, n: int = 5) -> list[dict]:
        return [
            {"turn_index": i, "agent_name": "agent", "role": "agent",
             "content": f"Turn {i} discusses the contract renewal and vendor performance metrics."}
            for i in range(n)
        ]

    def test_compress_happy_path_returns_state(self, bridge_container):
        adapter = MCPAdapter(bridge_container)
        result = adapter.dispatch("compress_context", {
            "workflow_id": "wf-mcp-001",
            "turns": self._turns_payload(5),
        })
        assert "state" in result
        assert "error" not in result
        assert result["workflow_id"] == "wf-mcp-001"

    def test_compress_result_state_is_schema_valid(self, bridge_container):
        from compaction_engine.schemas.extraction_schema import CompressedContextState
        adapter = MCPAdapter(bridge_container)
        result = adapter.dispatch("compress_context", {
            "workflow_id": "wf-mcp-002",
            "turns": self._turns_payload(5),
        })
        state = CompressedContextState.model_validate(result["state"])
        assert state.workflow_id == "wf-mcp-002"

    def test_compress_route_field_is_present(self, bridge_container):
        adapter = MCPAdapter(bridge_container)
        result = adapter.dispatch("compress_context", {
            "workflow_id": "wf-mcp-003",
            "turns": self._turns_payload(5),
        })
        assert result["route"] in ("compressed", "passthrough", "noop")

    def test_unknown_tool_returns_error_dict(self, bridge_container):
        adapter = MCPAdapter(bridge_container)
        result = adapter.dispatch("nonexistent_tool", {})
        assert "error" in result
        assert "available_tools" in result

    def test_validate_identical_states_returns_passed_true(self, bridge_container):
        from compaction_engine.schemas.extraction_schema import (
            CompressedContextState, ConfidenceLevel, ExtractedFact, FactType,
        )
        state = CompressedContextState(
            workflow_id="wf-mcp-val",
            compaction_version=1,
            source_turn_count=2,
            token_count_estimate=15,
            facts=(
                ExtractedFact(
                    fact_id="mf1",
                    fact_type=FactType.DECISION,
                    statement="Approved a three-year maintenance contract with vendor.",
                    source_agent="legal_agent",
                    source_turn_start=0,
                    source_turn_end=1,
                    confidence=ConfidenceLevel.HIGH,
                ),
            ),
        )
        state_dict = json.loads(state.model_dump_json())
        adapter = MCPAdapter(bridge_container)
        result = adapter.dispatch("validate_compression", {
            "gold_standard": state_dict,
            "compressed": state_dict,
            "match_threshold": 0.99,
        })
        assert result["passed"] is True

    def test_dispatch_never_raises_on_bad_input(self, bridge_container):
        """MCP tool errors must be returned as dicts, never propagated as
        exceptions — an MCP server must always be able to send a response."""
        adapter = MCPAdapter(bridge_container)
        result = adapter.dispatch("compress_context", {"workflow_id": "x"})  # missing turns
        assert "error" in result
