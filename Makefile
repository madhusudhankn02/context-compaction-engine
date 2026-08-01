.PHONY: install install-dev install-phase2 test lint typecheck check clean \
        serve-rest serve-grpc compile-proto docker-up docker-down

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,anthropic,embeddings]"

install-phase2:
	pip install -e ".[dev,anthropic,embeddings,rest,grpc]"

test:
	pytest -v --cov=compaction_engine --cov-report=term-missing

lint:
	ruff check src tests

typecheck:
	mypy src

check: lint typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

# ── Phase 2 targets ────────────────────────────────────────────────────────

serve-rest:
	uvicorn compaction_engine.bridge.rest.app:app --reload --port 8000

serve-grpc:
	python -m compaction_engine.bridge.grpc.server

compile-proto:
	bash scripts/compile_proto.sh

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down
