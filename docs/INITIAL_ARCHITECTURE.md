# Initial Architecture

Written per spec §90 (step 2). Records what exists, how it diverges from
`README.md` (the project spec), and what Phase 1 actually delivered.

## 1. Repository inspection

| Area | State before Phase 1 |
|---|---|
| `frontend/` | Next.js app initialised; `app/layout.tsx`, `app/page.tsx` only |
| `backend/` | Structural boilerplate: packages, protocols, route stubs raising `NotImplementedError` |
| Infrastructure | `docker-compose.yml`, `docker/postgres/init.sql`, `.env.example` |
| Database | None provisioned, no migrations |

No reusable application code existed beyond the boilerplate; nothing was
discarded.

## 2. Divergence from the spec

| Spec | Implementation | Reason |
|---|---|---|
| `apps/api` + root `core/`, `models/`, … (§49) | All Python under `backend/` keeping those package names | One `pyproject.toml`, venv, test suite and image; matches the existing `backend/` + `frontend/` split |
| Background workers unspecified (§46) | `arq` | Async-native and Redis-backed, so it shares the event loop and connection pool with FastAPI; Celery would add a second concurrency model |
| Host Postgres port 5432 | 5433 by default, `POSTGRES_PORT` overridable | 5432 was already bound on the development machine |

Everything else follows the spec: endpoint surface (§44), error taxonomy (§47),
observability fields (§48), tenancy columns (§43), pgvector (§6).

## 3. Layering

```text
app/api      HTTP only: routing, dependency resolution, error translation
     ↓
app/services orchestration and transactions; no FastAPI imports
     ↓
core/*       AI pipeline protocols; no FastAPI, no SQLAlchemy
     ↓
app/models   SQLAlchemy ORM + Alembic migrations
```

`core/` is deliberately framework-free so pipelines can run from workers,
scripts and experiment runners, not just from a request.

## 4. What Phase 1 delivered

- **Configuration** — typed `Settings` for every §81 variable;
  `validate_runtime_settings()` refuses to boot production with a template
  secret, `DEBUG=true`, console logging or `AUTH_MODE=disabled`.
- **Observability** — structlog with credential redaction; every line carries
  `request_id` (echoed as `X-Request-ID`, honoured if the caller supplies one).
- **Errors** — the §47 `ErrorCode` taxonomy behind one response envelope
  carrying `code`, `message`, `details`, `request_id`, `trace_id`.
- **Persistence** — async SQLAlchemy 2.0; initial migration creates the
  `vector` and `pg_trgm` extensions plus `organizations`, `users`, `projects`.
- **Health** — `/health` is dependency-free (liveness); `/ready` pings Postgres
  and Redis and returns 503 when degraded.
- **Queue** — one arq pool, created in the app lifespan and shared with the
  readiness probe; the worker registers 7 task entry points.
- **Projects** — the first vertical slice, proving route → dependency →
  service → ORM → migration.
- **Tests** — 29 tests. Integration tests run against real Postgres and Redis,
  and build the schema by running the real migrations, so a broken migration
  fails the suite.

## 5. Authentication

Phase 1 runs `AUTH_MODE=disabled` with a fixed development principal seeded at
startup (§43: do not over-engineer multi-tenancy initially). Tenancy columns
are populated from day one and every query already filters on the principal,
so enabling real auth changes only `get_current_principal`.

`get_project_scope` raises `NotFoundError`, never `ForbiddenError`, for a
project in another organization, so the API does not disclose which project IDs
exist.

## 6. Missing infrastructure (next phases)

| Need | Phase |
|---|---|
| Object storage implementation (local/S3) | 2 |
| Parser, chunker, ingestion worker | 2 |
| Embedding provider and pgvector index | 3 |
| BM25 / Postgres FTS index | 4 |
| Model provider adapters and registry | 7 |
| Trace persistence and SSE streaming | 8 |
| Real authentication (`AUTH_MODE=jwt`) | not yet scheduled |

ORM modules for these exist under `backend/app/models/` but are deliberately
not registered in `Base.metadata`; a model joins the metadata in the phase that
gives it real columns, so migrations never create half-designed tables.
