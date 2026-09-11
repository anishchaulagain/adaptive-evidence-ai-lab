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
| PDF library unspecified | `pypdf` | BSD licensed and pure Python; PyMuPDF is AGPL and unsuitable for this project |
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

## 6. What Phase 2 delivered

The ingestion pipeline (§67): upload -> validate -> store -> parse -> chunk.

- **Layering correction** — the error taxonomy moved to `core/errors.py`, since
  a failing pipeline stage must be attributable whether it ran in a request, a
  worker or a script. `app/core/errors.py` now adds only the HTTP status, via
  an explicit code-to-status map. Storage backend selection moved to
  `app/core/storage.py`: it reads `Settings`, so it belongs in the composition
  root. `core/` imports nothing from `app/`.
- **Parsers** — `pypdf` for PDF (one block per page), plus text and Markdown,
  behind a registry keyed on MIME type. DOCX and PPTX register without any
  other module changing.
- **Chunking** — deterministic recursive character splitting with overlap,
  cutting on the widest natural boundary that fits. Re-ingesting a document
  reproduces identical boundaries, so citations stay stable.
- **Provenance** — every chunk carries page and character offsets. Slicing the
  source by those offsets returns the chunk text; this is asserted directly.
- **Validation** — media type resolved from the extension in preference to the
  client's claim, size enforced while streaming, content hashed for
  per-project deduplication (409 names the existing document).
- **Idempotency** — a re-run clears prior chunks first, so arq's retries
  converge instead of tripping the `(document_id, ordinal)` constraint.
- **Logging** — log streams pinned to UTF-8. On a legacy Windows codepage, a
  document containing non-Latin text previously raised inside the logging
  handler, losing the line and emitting a traceback.

Token counts are an explicit estimate (`HeuristicTokenCounter`) because no
embedding model is configured yet; Phase 3 registers the real tokenizer behind
the same protocol.

## 7. Missing infrastructure (next phases)

| Need | Phase |
|---|---|
| S3 / Azure object storage backends | when deployed |
| DOCX and PPTX parsers, OCR for scanned pages | 2 (follow-up) |
| Embedding provider and pgvector index | 3 |
| BM25 / Postgres FTS index | 4 |
| Model provider adapters and registry | 7 |
| Trace persistence and SSE streaming | 8 |
| Real authentication (`AUTH_MODE=jwt`) | not yet scheduled |

External `DataSource` connectors remain deferred: documents arrive only by
direct upload, so a document carries no `source_id` yet.

ORM modules for the remaining entities exist under `backend/app/models/` but
are deliberately not registered in `Base.metadata`; a model joins the metadata in the phase that
gives it real columns, so migrations never create half-designed tables.
