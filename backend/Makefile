.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-unit test-integration check migrate revision run worker

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the project with dev dependencies
	pip install -e ".[dev]"

lint: ## Lint with ruff
	ruff check .

format: ## Format and fix imports
	ruff format .
	ruff check --fix .

typecheck: ## Type-check with mypy
	mypy .

test: ## Run the whole test suite
	pytest

test-unit: ## Run fast tests only
	pytest -m unit

test-integration: ## Run tests that need Postgres and Redis
	pytest -m integration

check: lint typecheck test ## Everything CI runs

migrate: ## Apply migrations
	alembic upgrade head

revision: ## Autogenerate a migration: make revision m="add chunks"
	alembic revision --autogenerate -m "$(m)"

run: ## Run the API with reload
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Run the background worker
	arq app.workers.settings.WorkerSettings
