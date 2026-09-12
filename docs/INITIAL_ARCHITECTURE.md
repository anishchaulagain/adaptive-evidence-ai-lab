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
| Model provider unspecified (§7) | **Mistral** (`mistral-embed`) | Chosen by the project owner; the only provider key configured. Adapters for others slot in behind the same protocol |
| `EMBEDDING_DIMENSIONS=1536` in the template | 1024 | 1536 is OpenAI's width; `mistral-embed` emits fixed 1024-d vectors that cannot be reduced |
| Object storage abstraction (§6) | Local filesystem only | Not deployed yet; S3 and Azure raise `NotImplementedError` behind the same protocol |
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

## 7. What Phase 3 delivered

Dense retrieval (§68): embedding provider -> vector storage -> similarity
search -> top-K, with scores exposed.

- **Provider** — `models/providers/mistral.py` talks to `POST /v1/embeddings`
  over httpx rather than the vendor SDK, so error mapping, timeouts and retry
  policy are explicit and testable. HTTP statuses map onto the domain codes
  (429 -> `MODEL_RATE_LIMIT`, 5xx -> `MODEL_UNAVAILABLE`, 401 ->
  `PROVIDER_NOT_CONFIGURED`); only transient failures are retried, because
  repeating a rejected key just burns the rate limit.
- **Response validation** — vectors are reordered by the response's `index`
  rather than assumed positional, and both the count and the width are checked.
  A silent mis-ordering would attach the wrong vector to every chunk.
- **Secrets** — the API key is applied per request, not baked into a client the
  adapter may not own, and provider response bodies never reach `details`,
  which is serialised into API responses.
- **Storage** — `chunks.embedding` is `vector(1024)` with an HNSW index using
  `vector_cosine_ops`. Mistral vectors are unit-norm, so cosine and inner
  product rank identically; cosine is used so the index stays correct if a
  future model emits un-normalised vectors.
- **Resumability** — only chunks missing a vector, or carrying a different
  `embedding_model`, are sent to the provider. Re-running a job costs nothing,
  and a model change becomes a re-run rather than a rebuild.
- **Degradation** — without a key the platform still ingests, parses and
  chunks; only the embedding stage is unavailable, and `/search` fails with
  `PROVIDER_NOT_CONFIGURED` rather than a bare 500.

`POST /search` is deliberately separate from `/query`: it returns evidence and
scores with no generated answer, so retrieval can be evaluated without an LLM
in the loop. `/query` (Phase 7) composes generation on top of it.

Guards added after finding real defects: a blank `MISTRAL_API_KEY=` line now
reads as unset rather than as an empty key that fails with a 401 much later;
`EMBEDDING_DIMENSIONS` is checked against the column width at boot; and a
worker job for a document deleted while queued is terminal rather than retried
five times.

The worker process also now establishes its own queue pool. Ingestion chains an
embedding job, and the worker runs outside the API's lifespan — so without it,
ingestion completed and then failed at the hand-off, after doing all the
parsing work. Tests that drive a job through the API fixtures inherit that
lifespan and cannot catch such a gap, so `tests/integration/test_worker_runtime.py`
exercises the worker's own startup directly.

## 8. What Phase 4 delivered

Keyword retrieval (§69): score, rank and matched terms over a generated
`tsvector` column with a GIN index.

- **Naming** — the retriever is `keyword` / `PostgresFtsRetriever`, never
  `bm25`. §6 permits Postgres FTS as the initial backend, but `ts_rank_cd` is
  cover-density ranking, not Okapi BM25. Mislabelling it would corrupt the very
  strategy comparisons this platform exists to run. `KEYWORD_BACKEND=bm25`
  remains reserved for a real implementation behind the same protocol.
- **Generated column** — `chunks.tsv` is `GENERATED ALWAYS AS (...) STORED`, so
  it cannot drift from `text` the way a trigger- or application-maintained
  column can.
- **Query parsing** — `websearch_to_tsquery`, which tolerates anything a user
  types. `to_tsquery` raises on malformed input, which would let a search box
  return a 500.
- **Matched terms** — ordered by position in the query, not alphabetically as
  `tsvector_to_array` returns them, so they read like the query that produced
  them. `query_terms` exposes what the query stemmed to, which is what turns a
  surprising ranking into an explicable one.

### Measured properties that constrain Phase 5

| Property | Consequence |
|---|---|
| `websearch_to_tsquery` is conjunctive | Every lexeme must be present, so the keyword arm is high-precision, low-recall, and frequently returns nothing for a natural-language question. Fusion must tolerate an empty keyword set. |
| `ts_rank_cd` has no IDF | A rare identifier and a common word weigh the same. This is why OR semantics were *not* adopted: without IDF, widening to OR would rank common-word matches alongside exact ones. It is also the concrete gap a true BM25 backend would close. |
| Cover density falls as a query spans more terms | Scores rank chunks within one query only. They are not comparable across queries or strategies and must not be averaged or thresholded globally. |

Both properties are covered by tests, so they are known behaviour rather than
Phase 5 surprises.

A live comparison on a corpus of near-identical passages differing only in an
identifier (`ERR-5521` vs `ERR-9310`, `4.2.1` vs `3.9.7`) had both strategies
at 4/4 top-1. The assumption that dense retrieval fails on exact identifiers
did not hold for `mistral-embed` at this corpus size — which is a finding for
the Phase 14 experiments to measure properly, not to assume.

The Postgres `english` configuration splits `get_user_by_id` into `get`,
`user`, `id` and drops `by` as a stopword, so snake_case identifiers are
retrievable but not matchable as exact units. Tested and documented rather than
papered over.

## 9. What Phase 5 delivered

Hybrid retrieval (§70): dense + keyword -> RRF -> unified ranking, behind the
uniform `Retriever` interface the spec asks for.

- **Uniform interface** — `PgVectorRetriever` gained a `retrieve()` that embeds
  internally, so `HybridRetriever` composes two `Retriever`s rather than
  reimplementing either. A true BM25 backend or Qdrant drops in untouched.
  `retrieve_with_vector` remains for callers that already hold the vector and
  should not pay to compute it twice.
- **Why RRF** — the arms return cosine similarity and normalised `ts_rank_cd`,
  which share no scale and, for the keyword arm, are not comparable across
  queries. RRF uses only each arm's ordering, so it cannot be skewed by one
  arm's scores happening to be larger. `weighted_score_fusion` stays explicitly
  unimplemented until Phase 9 produces the score distributions needed to
  calibrate it honestly.
- **Fusion provenance** (§17) — every fused hit carries `retrieval_source`,
  `fusion_score`, and the pre-fusion rank and score of each arm. This is what
  answers "did semantic miss this and keyword recover it?".
- **Fetch depth** — each arm retrieves `3 x top_k` (minimum 20) before fusion,
  because a chunk the dense arm ranked 12th that keyword ranked 1st is exactly
  the case hybrid exists to catch. Truncation happens after fusion.
- **Sequential arms** — the arms share one `AsyncSession`, which does not
  support concurrent operations. Overlapping a ~10ms keyword query with a
  ~400ms embedding call would not repay running a second session and its
  connection accounting.
- **Determinism** — ties break on chunk ID, never input order, and fusion
  recomputes rank from list position rather than trusting a hit's self-reported
  `rank`, so a mislabelling retriever cannot corrupt the result.

### Measured: does fusion beat dense alone?

On a 12-chunk corpus with real `mistral-embed`, across three exact-identifier
probes and four natural-language questions:

| strategy | top-1 accuracy |
|---|---|
| keyword | 3/7 — returned nothing for every natural-language question |
| semantic | 7/7 |
| hybrid | 7/7 |

Hybrid matched the best arm on every probe and never degraded dense, but it did
not beat it: dense is already at ceiling at this corpus size, so fusion has
nothing to add. That is a null result, not a success — the benefit of hybrid
retrieval remains unmeasured until the Phase 9 harness runs it over a corpus
large enough, and noisy enough, for dense retrieval to actually fail. Phases 13
and 14 exist to settle it.

## 10. Phase 6 (reranking): not implemented

`GET /v1/models` on this account lists 46 models — embeddings, chat and
moderation — and no reranker. The alternatives were a local cross-encoder,
which means a torch dependency for a project that currently installs in
seconds, or a second provider key that does not exist. Neither is justified to
satisfy a phase heading.

§71 asks for "a reranker interface... keep it replaceable", and
`core.reranking.base.Reranker` provides exactly that seam.
`RetrievedChunk.rerank_score` is already on the result type. An implementation
drops in without touching retrieval, fusion or generation.

## 11. What Phase 7 delivered

Grounded generation (§72): evidence set -> prompt -> LLM -> structured answer
-> citation mapping.

- **Claim-level provenance** (§22) — the answer decomposes into claims, each
  citing the passages supporting it. A single citation appended to a whole
  answer cannot be checked, because nothing says which part it supports.
- **Citations by index, not ID** — passages are numbered positionally in the
  prompt. A UUID wastes tokens and invites transcription errors; an index can
  be validated against what was actually retrieved, so an invented citation is
  *detectable* rather than plausible-looking. Out-of-range citations are
  dropped and counted in `invented_citations`.
- **Unsupported claims are surfaced, not hidden** — a claim citing nothing is
  kept and flagged, because that is precisely what the verification layer
  (§21) must be able to find.
- **Abstention is a correct outcome** — with no evidence the model is never
  called at all, and the endpoint abstains. Paying a provider to be told there
  is nothing to answer from is waste.
- **Determinism** — temperature pinned to 0, so an answer does not vary
  between identical runs and downstream evaluation stays reproducible.
- **Shared transport** — `MistralTransport` now carries HTTP, retry and error
  mapping for both embeddings and chat, with the caller's stage code passed
  through so a failure stays attributable to the stage that caused it rather
  than collapsing to a generic internal error.

### Generation is unverified against the live provider

The key has embedding quota but **zero** chat quota: `/chat/completions`
returns 429 with `x-ratelimit-limit-req-minute: 0`, which is a plan limit, not
a transient rate limit, so retrying cannot help. The live generation checks
skip themselves with that diagnosis rather than failing.

What this means honestly: the generation path is complete and tested against a
deterministic fake model — 26 unit tests covering citation validation,
abstention and malformed output, plus 18 integration tests through the real
retrieval stack — but no real model has yet produced an answer through it. The
JSON contract, prompt quality and abstention behaviour of the actual model
remain unconfirmed. Enabling chat on the plan and running
`pytest tests/integration/test_mistral_live.py` is the check that closes that
gap.

The `/query` endpoint was exercised live end to end: real hybrid retrieval ran
(`mistral-embed`, ~500ms), found no evidence in an empty project, and abstained
without calling the chat model — confirming the no-evidence path, which is the
one place the missing quota does not bite.

## 12. What Phase 8 delivered

The trace system (§73): every query persists a trace with a span per stage.

- **Failures are traced** — a query that raises still persists what ran before
  it did, then re-raises. Verified against a real provider failure: retrieval
  recorded `ok` with `candidate_count=1`, generation recorded `error` with
  `MODEL_RATE_LIMIT`. §47 requires errors to appear in traces, and an untraced
  failure is the one hardest to diagnose.
- **Retry cost is visible** — that failed generation span measured 4293ms
  against a ~440ms retrieval, which is the three retry attempts and their
  backoff showing up as a duration rather than as a mystery.
- **Spans nest** — a stage opened inside another records its parent, so the
  tree in §23 reconstructs.
- **Ordered by start sequence, not timestamp** — a monotonic counter assigned
  when a span opens. A wall-clock timestamp is not enough: Windows' clock is
  coarse enough (~15ms) that two stages share one, and a stable sort then
  silently falls back to *completion* order, placing a nested stage before its
  parent. A test caught this.
- **`offset_ms` is precomputed** so a client draws §24's timeline directly
  instead of deriving it and getting the zero point wrong.
- **Cost is null, not zero** — token pricing is unset by default because rates
  differ by plan and change. A guessed figure reported as a measured cost would
  be worse than no figure (spec rule 9). Set
  `GENERATION_INPUT_COST_PER_MTOK` / `GENERATION_OUTPUT_COST_PER_MTOK` to
  populate it.
- **Trace ID in the logging context** — bound for the duration of the query, so
  every log line carries it and logs cross-reference the stored trace (§48).

### A schema decision

The spec lists both `queries` and `traces` tables. The trace holds its query
text directly instead: a query with no trace is not something this platform
creates, so splitting them would mean a join on every read and two rows to keep
consistent, for no gain. A `queries` table can be added later if query history
needs to outlive trace retention.

### Stages that exist but never fire

`reranking` and `verification` are in `TraceStage` and report `null` latency
rather than `0`. Nothing runs them yet — Phase 6 is skipped and verification
(§21) is unscheduled — and a zero would read as "ran instantly" rather than
"did not run".

## 13. What Phase 9 delivered

The evaluation lab (§§26-28, 74) — and with it, answers to the questions
Phases 4 and 5 left open.

- **Metrics are pure functions** over ranked IDs and a gold set, with no
  database or provider, and 37 unit tests checking them against hand-computed
  values rather than against their own behaviour. Every later conclusion rests
  on these being right.
- **Undefined is not zero.** An item with no gold chunks has no recall.
  Averaging a fabricated zero would understate the system under test, so
  metrics return `null` and every aggregate reports `n`, the number of items
  that defined it — a recall of 0.9 over 2 of 50 items means something very
  different from 0.9 over 50.
- **Per-condition reporting**, because a system that does well on CLEAN says
  nothing about CONFLICTING or ADVERSARIAL.
- **Failures are categorised by cause** — `retrieval_miss`, `over_abstention`,
  `invented_citation`, `unsupported_claim` — ordered most-fundamental first, so
  a bad score says which stage failed (§36).
- **Frozen config** on each run, so a run's meaning cannot change because a
  project default was edited afterwards.
- **No provider needed** for retrieval metrics; a keyword run measures
  retrieval quality with no API key at all.

### A reproducibility fix this phase forced

Chunk IDs were random, so re-ingesting a document minted new ones and silently
invalidated every dataset's gold set. They are now derived deterministically
from `(document_id, ordinal)` via UUIDv5. Chunking was already deterministic;
now the IDs are too, and a dataset survives re-ingestion.

### Measured: keyword vs semantic vs hybrid

Real `mistral-embed`, 12-chunk corpus, 10 items, executed through the worker
queue. Six natural-language questions (CLEAN) and four exact-identifier probes
(ADVERSARIAL).

| metric | keyword | semantic | hybrid |
|---|---|---|---|
| recall@1 | 0.400 | 0.800 | 0.800 |
| recall@5 | 0.400 | 1.000 | 1.000 |
| MRR | 0.400 | 0.900 | 0.900 |
| nDCG@5 | 0.400 | 0.926 | 0.926 |

recall@5 by condition — CLEAN: keyword 0.000 (n=6), semantic 1.000, hybrid
1.000. ADVERSARIAL: all three 1.000. Failures: keyword 6 x `retrieval_miss`;
semantic and hybrid none.

Three findings, all contradicting assumptions the spec builds on:

1. **Hybrid is identical to semantic on every metric.** RRF fusion adds
   nothing measurable here, because the keyword arm contributes no candidate
   that dense retrieval had not already ranked highly.
2. **The keyword arm is strictly dominated.** It scores 0.000 on
   natural-language questions — conjunctive matching returns nothing — and
   merely ties on identifiers.
3. **§15B's premise does not hold.** Lexical search is supposed to rescue dense
   retrieval on exact identifiers, but `mistral-embed` scored 1.000 on the
   adversarial set unaided.

Scope honestly: 12 chunks and 10 items is small, and dense retrieval is at
ceiling, which is exactly the regime where fusion has no room to help. The
finding is that hybrid is *not yet justified*, not that it never could be. The
corpus size and noise level at which dense starts failing is what Phases 13-14
exist to find — and the harness to find it now exists.

## 14. What Phase 10 delivered

Visualization payloads (§§24, 37, 75) for the four views the spec prioritises.
Backend only: the builders live in `visualization/`, free of the ORM and of
FastAPI, so they are unit-testable and reusable by the experiment and benchmark
reports. The Next.js frontend is untouched.

- **Trace timeline** — bars with `offset_ms` and a resolved nesting `depth`.
  Both are computed server-side because deriving them is precisely where a
  viewer gets the zero point or the indentation wrong. Tolerates a missing
  parent and cannot loop on a cyclic parent link.
- **Evidence graph** — the §18 chain, rebuilt from the stored trace alone.
  Chunks that were retrieved but never cited still appear, marked
  `cited: false`: what the model was given and chose not to use is the
  interesting part, and a graph showing only citations would hide it.
- **Retrieval comparison** — organised by chunk rather than by strategy, since
  the questions worth asking ("did semantic miss this, did keyword recover
  it?") are about disagreement. `unique_to` counts what each arm contributed
  alone — an arm scoring zero there could be removed without changing results.
- **Evaluation heatmap** — strategy x condition for one metric. A combination
  never run is `null`, never `0`; §37 is explicit that benchmark numbers must
  not be fabricated.

### A schema addition this required

Traces recorded timings but not the answer, so the evidence graph could not be
rebuilt from one. `traces` now stores `answer`, `claims`,
`retrieved_chunk_ids` and `cited_chunk_ids` — which is what §18's "every answer
should be traceable to evidence" actually requires. The migration adds the
NOT NULL columns with server defaults (a populated table would otherwise reject
them) and drops the defaults immediately, so the schema still matches the model
and `alembic check` stays clean.

### Rendered from real data

The heatmap built from the Phase 9 runs reproduces that phase's finding
directly: `keyword` scores 0.000 on clean and 1.000 on adversarial, while
`semantic` and `hybrid` score 1.000 on both.

The retrieval comparison makes the same point in a second way. On a live query,
`unique_to` came back `{semantic: 0, keyword: 0, hybrid: 0}` — no strategy
retrieved a chunk the others missed. That is the Phase 5 null result visible as
a view rather than as a claim.

The timeline was rendered from a genuinely failed trace (retrieval 383.7ms ok,
generation 3793.4ms error, `MODEL_RATE_LIMIT`), confirming the views handle the
failure path, which is the one most worth being able to inspect.

### Not built

The Embedding Map (§37.3) needs dimensionality reduction, which means
scikit-learn or umap-learn. §75 does not list it among the four priorities, and
it is exploratory rather than diagnostic, so it was skipped rather than pulling
in a large dependency for it.

## 15. Missing infrastructure (next phases)

| Need | Phase |
|---|---|
| S3 / Azure object storage backends | when deployed |
| DOCX and PPTX parsers, OCR for scanned pages | 2 (follow-up) |
| True BM25 with IDF (`KEYWORD_BACKEND=bm25`) | when ranking quality is measured |
| Reranker implementation | when a rerank model or cross-encoder is available |
| Live generation verification | when chat quota is enabled on the key |
| Verification layer (§21) | unscheduled; `TraceStage.VERIFICATION` reserved |
| LLM-judge metrics (faithfulness, answer correctness) | blocked on chat quota |
| SSE streaming (§45) | carries the trace spans Phase 8 now records |
| Embedding map (§37.3) | needs scikit-learn; not a §75 priority |
| Frontend views | payloads exist; the Next.js app is still boilerplate |
| BM25 / Postgres FTS index | 4 |
| Model provider adapters and registry | 7 |
| Trace persistence and SSE streaming | 8 |
| Real authentication (`AUTH_MODE=jwt`) | not yet scheduled |

External `DataSource` connectors remain deferred: documents arrive only by
direct upload, so a document carries no `source_id` yet.

ORM modules for the remaining entities exist under `backend/app/models/` but
are deliberately not registered in `Base.metadata`; a model joins the metadata in the phase that
gives it real columns, so migrations never create half-designed tables.
