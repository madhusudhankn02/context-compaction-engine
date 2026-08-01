from __future__ import annotations

import json

import pytest


class TestHealthEndpoint:
    def test_returns_200_with_ok_status(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_returns_provider_and_model_fields(self, test_client):
        data = test_client.get("/health").json()
        assert "provider" in data
        assert "model" in data


class TestCompressEndpoint:
    def test_happy_path_returns_200(self, test_client, raw_turns_payload):
        resp = test_client.post(
            "/v1/compress",
            json={"workflowId": "wf-rest-001", "turns": raw_turns_payload},
        )
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, test_client, raw_turns_payload):
        data = test_client.post(
            "/v1/compress",
            json={"workflowId": "wf-rest-001", "turns": raw_turns_payload},
        ).json()
        required = {
            "workflowId", "route", "compressionRatio", "latencyMs",
            "factCount", "sourceTurnCount", "tokenCountEstimate",
            "warnings", "state",
        }
        assert required.issubset(data.keys())

    def test_state_field_is_valid_compressed_context_state(self, test_client, raw_turns_payload):
        from compaction_engine.schemas.extraction_schema import CompressedContextState
        data = test_client.post(
            "/v1/compress",
            json={"workflowId": "wf-rest-001", "turns": raw_turns_payload},
        ).json()
        # Must not raise
        state = CompressedContextState.model_validate(data["state"])
        assert state.workflow_id == "wf-rest-001"

    def test_empty_turns_returns_422(self, test_client):
        resp = test_client.post(
            "/v1/compress",
            json={"workflowId": "wf-empty", "turns": []},
        )
        assert resp.status_code == 422

    def test_missing_workflow_id_returns_422(self, test_client, raw_turns_payload):
        resp = test_client.post(
            "/v1/compress",
            json={"turns": raw_turns_payload},  # no workflowId
        )
        assert resp.status_code == 422

    def test_fallback_route_is_still_200(self, bridge_settings):
        """When the pipeline fails, the REST layer must return 200 (not 5xx)
        with route='passthrough'. Compression failure is NOT an HTTP error —
        the caller always gets a valid state."""
        from compaction_engine.bridge.container import BridgeContainer
        from compaction_engine.bridge.rest.app import create_app
        from compaction_engine.bridge.router import FallbackRouter
        from compaction_engine.extraction.pipeline import ExtractionPipeline
        from compaction_engine.extraction.reranker import FactDeduplicator
        from compaction_engine.utils.exceptions import SchemaValidationError
        from fastapi.testclient import TestClient

        class _AlwaysFailExtractor:
            def invoke(self, _msgs):
                raise SchemaValidationError("Forced failure for test.")

        dedup = FactDeduplicator(bridge_settings, embedder=None)
        pipeline = ExtractionPipeline(bridge_settings, _AlwaysFailExtractor(), deduplicator=dedup)
        router = FallbackRouter(pipeline, bridge_settings)
        container = BridgeContainer(
            settings=bridge_settings, extractor=_AlwaysFailExtractor(),
            deduplicator=dedup, pipeline=pipeline, router=router,
        )
        with TestClient(create_app(container=container)) as client:
            resp = client.post(
                "/v1/compress",
                json={
                    "workflowId": "wf-fail",
                    "turns": [{"turnIndex": 0, "agentName": "a", "role": "agent", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["route"] == "passthrough"


class TestValidateEndpoint:
    def _make_state_dict(self, workflow_id: str = "wf-val-001") -> dict:
        from compaction_engine.schemas.extraction_schema import (
            CompressedContextState, ConfidenceLevel, ExtractedFact, FactType,
        )
        state = CompressedContextState(
            workflow_id=workflow_id,
            compaction_version=1,
            source_turn_count=2,
            token_count_estimate=20,
            facts=(
                ExtractedFact(
                    fact_id="v1",
                    fact_type=FactType.DECISION,
                    statement="Contract was signed with vendor Acme Corp.",
                    source_agent="legal_agent",
                    source_turn_start=0,
                    source_turn_end=1,
                    confidence=ConfidenceLevel.HIGH,
                ),
            ),
        )
        return json.loads(state.model_dump_json())

    def test_identical_states_pass_all_gates(self, test_client):
        s = self._make_state_dict()
        resp = test_client.post(
            "/v1/validate",
            json={"goldStandard": s, "compressed": s, "matchThreshold": 0.99},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
        assert data["report"]["factRecall"] == 1.0

    def test_malformed_state_returns_422(self, test_client):
        resp = test_client.post(
            "/v1/validate",
            json={"goldStandard": {"not": "a state"}, "compressed": {"also": "invalid"}},
        )
        assert resp.status_code == 422

    def test_dropped_fact_sets_passed_false(self, test_client):
        from compaction_engine.schemas.extraction_schema import (
            CompressedContextState, ConfidenceLevel, ExtractedFact, FactType,
        )
        gold_state = CompressedContextState(
            workflow_id="wf-val-drop",
            compaction_version=1,
            source_turn_count=2,
            token_count_estimate=30,
            facts=(
                ExtractedFact(
                    fact_id="c1",
                    fact_type=FactType.CONSTRAINT,
                    statement="SLA requires ninety-nine point nine percent uptime.",
                    source_agent="ops_agent",
                    source_turn_start=0,
                    source_turn_end=1,
                    confidence=ConfidenceLevel.HIGH,
                ),
            ),
        )
        # Compressed drops the constraint — should fail constraint_recall gate.
        compressed_state = CompressedContextState(
            workflow_id="wf-val-drop",
            compaction_version=1,
            source_turn_count=2,
            token_count_estimate=10,
            facts=(
                ExtractedFact(
                    fact_id="d1",
                    fact_type=FactType.DECISION,
                    statement="Deployment scheduled for next Tuesday evening.",
                    source_agent="ops_agent",
                    source_turn_start=0,
                    source_turn_end=1,
                    confidence=ConfidenceLevel.HIGH,
                ),
            ),
        )
        resp = test_client.post(
            "/v1/validate",
            json={
                "goldStandard": json.loads(gold_state.model_dump_json()),
                "compressed": json.loads(compressed_state.model_dump_json()),
                "matchThreshold": 0.5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["passed"] is False
