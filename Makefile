.PHONY: install install-dev test lint typecheck check clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,anthropic,embeddings]"

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
