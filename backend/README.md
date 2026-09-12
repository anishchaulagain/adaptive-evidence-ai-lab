# Backend — Adaptive Evidence AI Lab

Phase 1 (foundation) is implemented and verified. Later phases are scaffolded:
their route handlers and pipeline stages raise `NotImplementedError` by design.
See [../docs/INITIAL_ARCHITECTURE.md](../docs/INITIAL_ARCHITECTURE.md).

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

Requires `make` (not present on Windows by default — see Local setup below
for the portable equivalents).

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

## Status

Working today:

| | |
|---|---|
| `GET /api/v1/health` | liveness, dependency-free |
| `GET /api/v1/ready` | pings Postgres + Redis, 503 when degraded |
| `POST/GET /api/v1/projects` | full slice through service, ORM and migration |
| `POST /api/v1/documents/upload` | validate, store, queue (202) |
| `GET /api/v1/documents[/{id}][/chunks]` | ingestion state and chunk provenance |
| `POST /api/v1/search` | semantic or keyword retrieval, scored, with provenance |
| `alembic upgrade head` | extensions, identity, projects, documents, chunks + HNSW |
| `arq` worker | ingests PDF, TXT and Markdown, then embeds |
| `ruff` + `mypy --strict` + `pytest` | 159 tests green (156 without a provider key) |

Ingestion (Phase 2): upload -> validate -> store -> parse -> chunk. Chunks
carry page and character offsets, so slicing the source by a chunk's offsets
returns its text — the contract citations depend on.

Dense retrieval (Phase 3): chunks are embedded with **mistral-embed** (1024-d,
fixed) into a pgvector column with an HNSW cosine index.

Keyword retrieval (Phase 4): a generated `tsvector` column with a GIN index,
queried via `websearch_to_tsquery` and ranked by `ts_rank_cd`. Results report
score, rank and matched terms. It is named `keyword`, not `bm25`, because
`ts_rank_cd` is cover-density ranking with no IDF — see
`app/retrieval/postgres_fts.py` for the measured differences. Keyword search
needs no provider key and no embeddings.

`POST /search` takes `strategy: semantic | keyword`. Hybrid fusion is Phase 5.

Embedding needs `MISTRAL_API_KEY`. Without it the platform still ingests,
parses and chunks; documents simply are not dense-retrievable until a key is
set and the embedding job re-runs. Object storage is local-filesystem only.

Every other route still raises `NotImplementedError`. ORM modules for later
phases exist but are not registered in `Base.metadata`, so migrations never
create half-designed tables — see `app/models/__init__.py`.

Authentication is off (`AUTH_MODE=disabled`) and requests run as a seeded
development principal. The app refuses to boot in production in that mode.

## Local setup

```bash
docker compose up -d postgres redis     # from the repository root
cp .env.example .env
cd backend
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m uvicorn app.main:app --reload --no-access-log
```

Run the quality gates:

```bash
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy .
.venv/Scripts/python -m pytest
```

The `Makefile` wraps these, but `make` is not installed by default on Windows;
the commands above are the portable form.

Postgres binds host port 5433 by default (5432 is commonly taken); override
with `POSTGRES_PORT`.
