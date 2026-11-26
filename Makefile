# Thin developer convenience wrapper. Every target maps to a command that CI
# also runs, so "make check" locally means the same thing as a green pipeline.
.DEFAULT_GOAL := help

PYTHON ?= python3

.PHONY: help install up down logs ps test test-unit test-integration eval lint fmt typecheck check migrate

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies into the current environment
	$(PYTHON) -m pip install -e ".[dev]"

up: ## Build and start the full stack (API + PostgreSQL)
	docker compose up --build -d

down: ## Stop the stack and remove containers (keeps the database volume)
	docker compose down

logs: ## Follow API logs
	docker compose logs -f api

ps: ## Show stack status
	docker compose ps

test: ## Run the full suite with coverage (integration tests skip without RAG_TEST_DATABASE_URL)
	$(PYTHON) -m pytest --cov --cov-report=term-missing

test-unit: ## Run hermetic unit tests only (no coverage gate)
	$(PYTHON) -m pytest -m "not integration"

test-integration: ## Run integration tests (requires RAG_TEST_DATABASE_URL)
	$(PYTHON) -m pytest -m integration

eval: ## Run the retrieval evaluation and compare with the saved baseline
	$(PYTHON) -m rag_engine.cli evaluate \
		--dataset eval/datasets/enterprise_baseline.json \
		$(if $(wildcard eval/baselines/dense_baseline.json),--baseline eval/baselines/dense_baseline.json,) \
		$(EVAL_ARGS)

lint: ## Lint and check formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt: ## Auto-fix lint issues and format code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck: ## Static type checking
	$(PYTHON) -m mypy

migrate: ## Apply database migrations (uses RAG_DATABASE_URL)
	$(PYTHON) -m alembic upgrade head

check: lint typecheck test ## Everything CI runs (except the Docker build)

demo: ## Run the interactive scripted demo
	./scripts/demo.sh
