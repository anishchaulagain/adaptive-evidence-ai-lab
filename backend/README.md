# Backend — Adaptive Evidence AI Lab

Boilerplate only. Every module below is wired but unimplemented: route handlers,
services and pipeline stages raise `NotImplementedError` by design.

## Layout

```text
backend/
├── app/                  FastAPI delivery layer
│   ├── main.py           create_app(): middleware, routers, lifespan
│   ├── api/              deps.py, errors.py, v1/router.py, v1/routes/*
│   ├── core/             config, logging, errors, context, security
│   ├── db/               engine/session, declarative base, column types
│   ├── middleware/       request ID, structured access log
│   ├── models/           SQLAlchemy ORM — the relational schema
│   ├── schemas/          Pydantic — the API contract
│   ├── services/         orchestration; no FastAPI imports
│   ├── streaming/        SSE events and encoding
│   └── workers/          arq worker settings, queue, tasks/
│
├── core/                 provider-independent AI pipeline (protocols only)
│   ├── ingestion/ parsing/ chunking/ embeddings/
│   ├── retrieval/ reranking/ evidence/ routing/
│   └── reasoning/ verification/ storage/ tracing/
│
├── models/               provider adapters, embedding models, registry
├── evaluation/           datasets, metrics, judges, failure analysis
├── benchmark/            AE-Bench harness and condition suites
├── experiments/          reproducible research experiments
├── visualization/        backend-computed payloads for the UI
│
├── alembic/              migrations (env.py reads DATABASE_URL from settings)
└── tests/                unit / integration / evaluation / benchmark
```

### Two directories named `core`

`app/core/` holds framework concerns (settings, logging, error taxonomy).
`core/` holds the AI pipeline and imports neither FastAPI nor SQLAlchemy, so it
stays usable from workers, scripts and experiment runners. Imports read
unambiguously: `app.core.config` vs `core.retrieval.base`.

## Layout vs. the spec

The spec (§49) proposes `apps/api` with `core/`, `models/`, `evaluation/` and
friends at the repository root. Those packages keep their names here but live
under `backend/`, giving one `pyproject.toml`, one virtualenv, one test suite
and one Docker image, and matching the existing `backend/` + `frontend/` split.

## Commands

```bash
make install      # pip install -e ".[dev]"
make run          # uvicorn app.main:app --reload
make worker       # arq app.workers.settings.WorkerSettings
make check        # ruff + mypy + pytest
make revision m="add chunks"
make migrate
```

Or bring up the whole stack from the repository root:

```bash
cp .env.example .env
docker compose up --build
```

## Conventions

- Configuration is read only through `app.core.config.get_settings()` — never
  `os.environ`.
- Every failure carries an `ErrorCode` (`app/core/errors.py`) so it is
  attributable in traces.
- Logs are structured and redact credentials; secrets never reach the browser.
- Long-running work is enqueued to `app/workers/`, never run in a request.
- Tenancy columns (`user_id`, `organization_id`, `project_id`) exist on models
  from the start via `TenantMixin`.

## Not done yet

No dependencies are installed, no database exists, no migration has been
generated, and the app has not been started. ORM models declare only
`__tablename__` and the shared mixins — domain columns land with the features
that need them.
