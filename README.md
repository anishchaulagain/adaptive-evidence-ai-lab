# Adaptive Evidence AI Lab

> **A visual laboratory for understanding, evaluating, and optimizing AI systems.**

**Status:** Initial Development Specification
**Project Type:** AI Systems / Research Platform / Multimodal RAG / Adaptive Inference
**Primary Goal:** Build one deployable, research-grade AI platform that combines data ingestion, hybrid retrieval, adaptive model routing, evidence verification, evaluation, observability, experimentation, and visual analytics.

---

# 1. Executive Summary

Adaptive Evidence AI Lab is an experimental AI systems platform designed to answer a central research question:

> **Can an AI system dynamically select the right evidence, retrieval strategy, model, and inference budget for each task while maximizing reliability and minimizing cost and latency?**

This is **not** a generic RAG chatbot.

The platform must allow a user to:

1. Connect heterogeneous data sources.
2. Inspect and visualize the ingested data.
3. Configure AI models and runtimes.
4. Ask questions against their data.
5. Compare semantic, keyword, and hybrid retrieval.
6. Visualize retrieved evidence.
7. Observe the complete AI execution trace.
8. Understand why specific evidence/models were selected.
9. Evaluate retrieval and answer quality.
10. Run controlled experiments.
11. Compare models and retrieval strategies.
12. Measure accuracy, faithfulness, citation quality, latency, token usage, and cost.
13. Dynamically adapt retrieval and inference based on query difficulty and evidence uncertainty.
14. Create reproducible research experiments and benchmarks.

The system should eventually support:

```text
Data
  ↓
Ingestion
  ↓
Document Understanding
  ↓
Indexing
  ↓
Query Analysis
  ↓
Adaptive Hybrid Retrieval
  ↓
Evidence Fusion
  ↓
Reranking
  ↓
Evidence Quality Assessment
  ↓
Model Routing
  ↓
Inference
  ↓
Verification
  ↓
Answer + Citations
  ↓
Evaluation
  ↓
Trace + Visualization
  ↓
Experiment / Benchmark
```

---

# 2. Product Philosophy

The project must be developed as a **research platform**, not simply as a chatbot.

The user should be able to answer:

* What data did the AI use?
* Where did the evidence come from?
* Why was this evidence selected?
* Which retrieval method found it?
* Did semantic retrieval miss something?
* Did keyword retrieval recover it?
* Was reranking useful?
* Which model generated the answer?
* Why was that model selected?
* How much context was used?
* How many tokens were consumed?
* How much did the query cost?
* How long did it take?
* Was the answer grounded?
* Were citations correct?
* Did the system encounter conflicting evidence?
* Would additional retrieval have helped?
* Would a stronger model have helped?
* Was additional reasoning worth its cost?

The platform should make these questions **visually inspectable**.

---

# 3. Core Research Thesis

The central research direction is:

> **Evidence quality × retrieval strategy × inference budget jointly determine the reliability of AI systems.**

The project should investigate whether an adaptive system can outperform fixed pipelines.

Instead of:

```text
Query
→ Fixed Retrieval
→ Fixed Model
→ Fixed Reasoning
→ Answer
```

build:

```text
Query
    ↓
Query Analysis
    ↓
Evidence Assessment
    ↓
Adaptive Retrieval
    ↓
Adaptive Model Selection
    ↓
Adaptive Inference Budget
    ↓
Verification
    ↓
Answer
```

The system must not blindly use more computation.

It should learn or implement policies for determining when additional:

* retrieval,
* reranking,
* model capability,
* context,
* verification,
* or reasoning

is actually useful.

---

# 4. Project Scope

## 4.1 Primary Modules

The application must contain these major areas:

```text
1. Dashboard
2. Data Lab
3. Model Lab
4. AI Playground
5. Trace Explorer
6. Evaluation Lab
7. Experiment Lab
8. Benchmark Lab
9. Visual Analytics
10. Settings
```

---

# 5. High-Level Architecture

```text
                         ┌──────────────────────────┐
                         │       WEB CLIENT         │
                         │ Next.js / React / TS     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       API GATEWAY        │
                         │ FastAPI / REST / SSE     │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
              ▼                       ▼                        ▼
      ┌──────────────┐       ┌──────────────┐        ┌──────────────┐
      │ Data Service │       │ AI Orchestr. │        │ Eval Service │
      └──────┬───────┘       └──────┬───────┘        └──────┬───────┘
             │                      │                       │
             ▼                      ▼                       ▼
      ┌──────────────┐       ┌──────────────┐        ┌──────────────┐
      │ Ingestion    │       │ Query        │        │ Metrics      │
      │ Parser       │       │ Analyzer     │        │ Judges       │
      │ OCR          │       │ Retrieval    │        │ Benchmarks   │
      │ Chunking     │       │ Reranking    │        │ Human Eval   │
      └──────┬───────┘       │ Routing      │        └──────────────┘
             │               │ Reasoning    │
             ▼               │ Verification│
      ┌──────────────┐       └──────┬───────┘
      │ Index Layer  │              │
      │ Vector DB    │              ▼
      │ BM25         │       ┌──────────────┐
      │ Metadata     │       │ Trace Store  │
      └──────────────┘       └──────────────┘
```

---

# 6. Recommended Technology Stack

## Frontend

Use:

* Next.js
* React
* TypeScript
* Tailwind CSS
* shadcn/ui
* React Query / TanStack Query
* React Flow
* D3.js
* Recharts or Plotly
* Monaco Editor where useful

The frontend should feel like a serious AI research/developer tool.

Avoid a generic chatbot UI.

---

## Backend

Use:

* Python
* FastAPI
* Pydantic
* SQLAlchemy
* async database access
* background workers

The AI orchestration layer should be modular and provider-independent.

---

## Database

Primary relational database:

* PostgreSQL

Use PostgreSQL for:

* users
* projects
* data sources
* documents
* chunks
* models
* experiments
* evaluation runs
* traces
* metrics
* configurations

---

## Vector Database

Initial recommendation:

**pgvector**

This reduces infrastructure complexity.

The architecture must nevertheless abstract the vector store so that Qdrant or another backend can be added later.

---

## Keyword Retrieval

Implement:

**BM25**

The initial implementation may use:

* PostgreSQL full-text search
* or a dedicated BM25 implementation

The retrieval interface must remain abstract.

---

## Cache / Queue

Use:

* Redis

Potential uses:

* caching
* background jobs
* ingestion status
* experiment jobs
* rate limiting
* temporary execution state

---

## Object Storage

Support an object-storage abstraction.

Possible implementations:

* S3
* Azure Blob Storage
* local filesystem for development

---

# 7. Model Provider Architecture

The system must not be hardcoded to one LLM.

Implement a provider abstraction.

Example:

```python
class ModelProvider:
    async def generate(...)
    async def stream(...)
    async def embed(...)
    async def estimate_cost(...)
```

Providers should eventually support:

```text
OpenAI
Anthropic
Google
OpenRouter
Azure OpenAI
Hugging Face
Ollama
vLLM
Custom OpenAI-compatible endpoint
```

Do not implement every provider immediately.

Start with:

1. OpenAI-compatible provider
2. Ollama/local provider

Then add others through the abstraction.

---

# 8. Model Configuration

Users must be able to configure:

```text
Provider
Model
Context Window
Temperature
Max Output Tokens
Embedding Model
Reranker
Reasoning Configuration
Timeout
Retry Policy
```

Sensitive credentials must:

* never be displayed in plaintext
* never be returned to the browser
* never be committed to Git
* never appear in logs

Use environment variables or secure server-side credential storage initially.

---

# 9. Data Lab

The Data Lab is the entry point for user-provided knowledge.

## Supported Sources

Initial:

```text
PDF
DOCX
PPTX
TXT
Markdown
CSV
XLSX
URL
```

Future:

```text
SharePoint
OneDrive
Google Drive
S3
Azure Blob
PostgreSQL
MySQL
MongoDB
REST APIs
```

---

# 10. Data Ingestion Pipeline

```text
SOURCE
  ↓
File Validation
  ↓
Parsing
  ↓
OCR if required
  ↓
Document Structure Extraction
  ↓
Metadata Extraction
  ↓
Text Extraction
  ↓
Tables
  ↓
Images / Figures
  ↓
Chunking
  ↓
Embeddings
  ↓
Keyword Index
  ↓
Vector Index
  ↓
Metadata Store
```

Each document should have an ingestion state:

```text
UPLOADED
PROCESSING
PARSED
INDEXING
READY
FAILED
```

---

# 11. Document Representation

Do not treat documents as plain text only.

Represent documents as structured objects.

Example:

```json
{
  "document_id": "doc_123",
  "title": "Example Report",
  "source_type": "pdf",
  "pages": 20,
  "sections": [],
  "chunks": [],
  "tables": [],
  "figures": [],
  "entities": [],
  "metadata": {},
  "quality": {}
}
```

Each chunk should contain:

```json
{
  "chunk_id": "chunk_123",
  "document_id": "doc_123",
  "page": 4,
  "section": "Introduction",
  "text": "...",
  "embedding": "...",
  "metadata": {},
  "source_location": {}
}
```

---

# 12. Data Visualization

The Data Lab must visually expose the dataset.

Show:

```text
Documents
Pages
Chunks
Tables
Figures
Entities
Languages
Duplicate Documents
Parsing Failures
OCR Required
Average Chunk Size
```

Visualizations:

### Dataset Overview

* document count
* page count
* chunk count
* source distribution

### Embedding Map

Provide:

```text
PCA
UMAP
t-SNE
```

where practical.

Users should be able to hover over points and identify the document/chunk.

### Knowledge Graph

Visualize:

```text
Entity → Entity
Entity → Document
Entity → Claim
Claim → Source
```

This is an optional advanced feature and should not block the core system.

---

# 13. AI Playground

The Playground is the main interactive experience.

User enters:

```text
Question
```

The system executes:

```text
Query Analyzer
      ↓
Retrieval
      ↓
Evidence Fusion
      ↓
Reranker
      ↓
Evidence Assessment
      ↓
Model Router
      ↓
Generation
      ↓
Verification
      ↓
Answer
```

The UI must display:

```text
Answer
Citations
Evidence
Retrieval strategy
Model used
Latency
Tokens
Cost
Confidence / evidence quality
```

---

# 14. Adaptive Query Analyzer

Every query should first pass through a query-analysis stage.

Possible properties:

```json
{
  "query_type": "conceptual",
  "difficulty": "medium",
  "requires_exact_match": false,
  "requires_multihop": false,
  "requires_numeric_reasoning": false,
  "ambiguity": 0.15,
  "estimated_evidence_difficulty": 0.42
}
```

Query types:

```text
CONCEPTUAL
EXACT_ENTITY
TECHNICAL
NUMERIC
MULTI_HOP
COMPARATIVE
TEMPORAL
AMBIGUOUS
```

The analyzer must influence downstream retrieval.

---

# 15. Retrieval Architecture

This is a major research component.

Implement three retrieval modes:

## A. Semantic Retrieval

Dense vector search.

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Top K
```

---

## B. Keyword Retrieval

BM25 / lexical search.

```text
Query
 ↓
Tokenizer
 ↓
BM25
 ↓
Top K
```

This is especially useful for:

* names
* acronyms
* identifiers
* error codes
* exact terminology
* numbers
* technical strings

---

## C. Hybrid Retrieval

```text
              QUERY
                │
        ┌───────┴────────┐
        ▼                ▼
    Semantic            BM25
    Retrieval          Retrieval
        │                │
        └───────┬────────┘
                ▼
          Result Fusion
                │
                ▼
              RRF
                │
                ▼
            Reranker
                │
                ▼
          Final Evidence
```

Use Reciprocal Rank Fusion initially.

---

# 16. Adaptive Hybrid Retrieval

This is one of the most important research components.

Do not assume:

```text
semantic = 50%
keyword = 50%
```

Instead:

```text
Query
 ↓
Query Analyzer
 ↓
Determine retrieval profile
 ↓
Semantic weight
Keyword weight
Top-K
Reranking depth
 ↓
Hybrid Retrieval
```

Example:

```text
Conceptual query
→ semantic-heavy

Exact product name
→ keyword-heavy

Technical query
→ hybrid

Multi-hop query
→ hybrid + larger candidate pool

Numeric query
→ keyword + semantic + verification
```

The first implementation can use deterministic rules.

Later, experiment with learned/adaptive policies.

---

# 17. Retrieval Result Schema

Each retrieval result should retain provenance.

```json
{
  "chunk_id": "chunk_123",
  "semantic_score": 0.87,
  "keyword_score": 12.4,
  "fusion_score": 0.91,
  "reranker_score": 0.94,
  "retrieval_source": [
    "semantic",
    "bm25"
  ],
  "rank_before_fusion": 3,
  "rank_after_fusion": 1
}
```

This data is essential for visualization and research.

---

# 18. Evidence Graph

Every answer should be traceable to evidence.

Example:

```text
                  USER QUERY
                      │
                      ▼
               Query Analyzer
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Chunk A       Chunk B       Chunk C
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                  Claim 1
                      │
                      ▼
                  Answer
```

The UI should allow users to click an answer claim and see:

```text
Claim
 ↓
Supporting chunks
 ↓
Document
 ↓
Page
 ↓
Original source
```

---

# 19. Model Router

The model router should choose the appropriate model.

Initial policy:

```text
Easy + high evidence confidence
→ cheaper/smaller model

Medium
→ medium model

Hard / ambiguous / conflicting
→ stronger model

Sensitive data
→ local/self-hosted model if configured
```

The router should consider:

```text
Query difficulty
Evidence quality
Conflict level
User budget
Latency requirement
Privacy requirement
Model availability
```

Do not expose hidden chain-of-thought.

Instead show structured routing information:

```text
Routing decision:
Difficulty: High
Evidence confidence: Low
Conflict: Detected
Selected model tier: High
Reason codes:
- conflicting evidence
- multi-hop query
- low retrieval confidence
```

---

# 20. Adaptive Inference Budget

The platform should eventually support:

```text
LOW
MEDIUM
HIGH
```

inference budgets.

The system can increase budget when:

```text
evidence confidence is low
query difficulty is high
sources conflict
answer verification fails
retrieval confidence is low
```

It should stop early when:

```text
evidence is sufficient
answer confidence is high
verification succeeds
additional computation is unlikely to help
```

The research goal is not:

> "Use more reasoning."

It is:

> **"Use additional reasoning only when it provides measurable benefit."**

---

# 21. Verification Layer

Before finalizing an answer, the system should optionally perform verification.

Verification should check:

```text
Groundedness
Citation support
Contradictions
Unsupported claims
Missing evidence
```

Example:

```json
{
  "grounded": true,
  "citation_correctness": 0.94,
  "unsupported_claims": 1,
  "contradictions_detected": false
}
```

---

# 22. Answer Generation

Answers must include citations to the actual source material.

The answer object should contain:

```json
{
  "answer": "...",
  "claims": [
    {
      "text": "...",
      "evidence": ["chunk_123", "chunk_456"]
    }
  ]
}
```

Do not rely solely on a single citation appended to the entire answer.

Where possible, support claim-level provenance.

---

# 23. Trace System

Every AI request should create a trace.

Example:

```text
TRACE
│
├── Query
│
├── Query Analysis
│
├── Semantic Retrieval
│   ├── candidate count
│   └── latency
│
├── BM25 Retrieval
│   ├── candidate count
│   └── latency
│
├── RRF Fusion
│
├── Reranking
│
├── Evidence Assessment
│
├── Model Routing
│
├── Generation
│
├── Verification
│
└── Final Answer
```

Each event should contain:

```json
{
  "trace_id": "...",
  "span_id": "...",
  "event_type": "...",
  "start_time": "...",
  "duration_ms": 123,
  "metadata": {}
}
```

---

# 24. Trace Visualization

Create a visual execution timeline.

Example:

```text
0ms       200ms      400ms      600ms      800ms      1000ms

Query     █
Analyze      ███
Semantic        █████
BM25             ████
Fusion                ██
Reranker                █████
LLM                         ███████████
Verify                                  ███
```

Users should be able to click an event and inspect its metadata.

---

# 25. System Metrics

Track at minimum:

```text
Total latency
Retrieval latency
Reranking latency
LLM latency
Verification latency

Input tokens
Output tokens
Total tokens

Estimated cost

Retrieval candidate count
Final evidence count

Model
Provider

Success/failure
Retry count
```

---

# 26. Evaluation Lab

Evaluation is a first-class module.

It must not be an afterthought.

Evaluation categories:

## Retrieval

```text
Recall@K
Precision@K
MRR
NDCG
Hit Rate
```

## Generation

```text
Answer Correctness
Faithfulness
Groundedness
Relevance
Completeness
```

## Citation

```text
Citation Correctness
Citation Completeness
Evidence Coverage
```

## Reliability

```text
Hallucination Rate
Contradiction Rate
Unsupported Claim Rate
```

## Systems

```text
Latency
Tokens
Cost
Failure Rate
```

---

# 27. Evaluation Dataset

Each evaluation item should support:

```json
{
  "question": "...",
  "expected_answer": "...",
  "gold_documents": [],
  "gold_chunks": [],
  "difficulty": "hard",
  "evidence_condition": "contradictory"
}
```

---

# 28. Evaluation Conditions

Create controlled evidence conditions:

```text
CLEAN
NOISY
CONFLICTING
OUTDATED
INCOMPLETE
ADVERSARIAL
```

This is critical for research.

---

# 29. AE-Bench

Create an internal benchmark called:

# AE-Bench

**Adaptive Evidence Benchmark**

The benchmark should evaluate AI systems under different evidence conditions.

Dimensions:

```text
Evidence Quality
├── Clean
├── Noisy
├── Conflicting
├── Outdated
├── Incomplete
└── Adversarial

Query Difficulty
├── Easy
├── Medium
├── Hard
└── Multi-hop
```

---

# 30. Benchmark Comparisons

At minimum compare:

```text
No RAG
Dense RAG
BM25 RAG
Hybrid RAG
Hybrid + Reranker
Adaptive Hybrid RAG
```

Later:

```text
Fixed model
Adaptive model routing

Fixed inference budget
Adaptive inference budget
```

---

# 31. Core Research Experiments

The system should make these experiments easy to run.

## Experiment 1 — Dense vs Keyword vs Hybrid

Question:

> Does hybrid retrieval recover evidence missed by semantic retrieval?

Compare:

```text
Dense
BM25
Hybrid
Hybrid + Reranker
Adaptive Hybrid
```

Metrics:

```text
Recall@5
Recall@10
MRR
NDCG
Answer accuracy
Citation accuracy
Latency
Cost
```

---

## Experiment 2 — Retrieval Under Noise

Question:

> How robust are retrieval strategies when irrelevant documents are introduced?

Increase noise:

```text
0%
10%
25%
50%
75%
```

Measure degradation.

---

## Experiment 3 — Conflicting Evidence

Question:

> How does the system behave when sources disagree?

Evaluate:

```text
Dense
Hybrid
Adaptive
```

Measure:

```text
contradiction detection
answer correctness
citation quality
hallucination
```

---

## Experiment 4 — More Reasoning vs Better Evidence

Important research experiment.

Compare:

```text
Better retrieval + low reasoning
Poor retrieval + high reasoning
Better retrieval + high reasoning
```

Research question:

> Does additional inference compensate for poor evidence?

Do not assume the answer.

Measure it.

---

## Experiment 5 — Adaptive Compute

Compare:

```text
Fixed low budget
Fixed medium budget
Fixed high budget
Adaptive budget
```

Measure:

```text
accuracy
faithfulness
cost
latency
```

Ideal outcome:

```text
Adaptive
→ similar or better quality
→ lower average cost
→ lower average latency
```

---

# 32. Experiment Lab

The Experiment Lab should allow users to configure:

```text
Dataset
Retrieval strategy
Top-K
Reranker
Model
Inference budget
Evaluation suite
Evidence condition
```

Then:

```text
RUN EXPERIMENT
```

The result should be reproducible.

---

# 33. Experiment Configuration

Example:

```json
{
  "experiment_name": "hybrid_vs_dense_v1",
  "dataset": "ae_bench_v1",
  "retrieval": {
    "semantic": true,
    "bm25": true,
    "fusion": "rrf",
    "reranker": true,
    "top_k": 10
  },
  "model": "example-model",
  "inference_budget": "medium",
  "evaluation": [
    "recall_at_10",
    "answer_correctness",
    "faithfulness",
    "citation_accuracy",
    "latency",
    "cost"
  ]
}
```

---

# 34. Experiment Results

Results should be visualized.

Examples:

```text
Accuracy vs Cost
Accuracy vs Latency
Recall vs K
Faithfulness vs Noise
Cost vs Evidence Quality
Performance vs Query Difficulty
```

Use interactive charts.

---

# 35. Model Comparison

Create a comparison interface.

Example:

| Metric       | Model A | Model B | Model C |
| ------------ | ------: | ------: | ------: |
| Accuracy     |       X |       X |       X |
| Faithfulness |       X |       X |       X |
| Citation     |       X |       X |       X |
| Latency      |       X |       X |       X |
| Tokens       |       X |       X |       X |
| Cost         |       X |       X |       X |

Users should be able to run the same evaluation suite against multiple models.

---

# 36. Failure Analysis

Create a dedicated failure-analysis interface.

Failure categories:

```text
Retrieval Miss
Wrong Ranking
Insufficient Evidence
Conflicting Evidence
Hallucination
Unsupported Claim
Incorrect Citation
Model Failure
Tool Failure
Timeout
```

Users should be able to click:

```text
FAILED QUERY
```

and inspect:

```text
Query
Expected Evidence
Retrieved Evidence
Missed Evidence
Model
Answer
Ground Truth
Failure Category
Trace
```

---

# 37. Visual Analytics

This is a major differentiator.

The platform should visually answer:

> "What actually happened inside my AI system?"

Create:

### 1. Evidence Graph

```text
Query
 ↓
Documents
 ↓
Chunks
 ↓
Claims
 ↓
Answer
```

### 2. Retrieval Comparison

```text
Semantic
    \
     → Fusion → Reranker
    /
BM25
```

### 3. Embedding Map

Visual cluster of document chunks.

### 4. Model Routing Graph

```text
Query
 ↓
Difficulty
 ↓
Evidence confidence
 ↓
Model choice
```

### 5. Trace Timeline

Full execution timeline.

### 6. Evaluation Heatmap

Example:

```text
             Clean  Noisy  Conflict  Outdated
Dense          92     81      64        70
BM25           86     80      68        73
Hybrid         95     89      79        84
Adaptive       96     92      87        89
```

Numbers above are placeholders only.

Never fabricate benchmark results.

---

# 38. Dashboard

The main dashboard should show:

```text
Documents
Queries
Experiments
Evaluation Runs
Average Accuracy
Average Retrieval Recall
Average Latency
Average Cost
Failure Rate
```

Also show recent activity.

---

# 39. Navigation

Recommended navigation:

```text
Adaptive Evidence AI Lab

├── Dashboard
│
├── Data Lab
│   ├── Sources
│   ├── Documents
│   ├── Dataset Explorer
│   ├── Embeddings
│   └── Knowledge Graph
│
├── Model Lab
│   ├── Providers
│   ├── Models
│   ├── Playground
│   └── Routing
│
├── AI Playground
│
├── Traces
│
├── Evaluations
│   ├── Overview
│   ├── Datasets
│   ├── Runs
│   └── Failure Analysis
│
├── Experiments
│
├── Benchmarks
│
└── Settings
```

---

# 40. AI Playground UX

The Playground should not resemble ChatGPT too closely.

Recommended layout:

```text
┌────────────────────────────────────────────────────────────┐
│ Query                                                      │
│                                                            │
│ [ Ask a question about your data...                    ]  │
│                                                            │
│ Retrieval: Adaptive Hybrid     Model: Auto                 │
│ Budget: Auto                   Dataset: Company Docs       │
└────────────────────────────────────────────────────────────┘

┌──────────────────────────┬─────────────────────────────────┐
│ ANSWER                   │ EXECUTION                       │
│                          │                                 │
│ Answer text              │ Query Analysis                  │
│                          │ Semantic Retrieval              │
│ [citation] [citation]    │ BM25 Retrieval                 │
│                          │ Fusion                          │
│                          │ Reranking                       │
│                          │ Model Routing                   │
│                          │ Generation                      │
│                          │ Verification                    │
└──────────────────────────┴─────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ EVIDENCE                                                   │
│                                                            │
│ Document A — Page 12                                       │
│ Document B — Page 4                                        │
│ Document C — Page 19                                       │
└────────────────────────────────────────────────────────────┘
```

---

# 41. Data Source UI

The user should be able to select:

```text
+ Add Data Source

[ PDF ]
[ DOCX ]
[ PPTX ]
[ CSV ]
[ XLSX ]
[ URL ]
```

Future:

```text
[ SharePoint ]
[ OneDrive ]
[ Google Drive ]
[ PostgreSQL ]
[ S3 ]
```

After ingestion:

```text
Source
Status
Documents
Chunks
Last Updated
Index Status
```

---

# 42. Security

Security is important because this platform can process private data.

Implement:

```text
Authentication
Authorization
Project isolation
Source isolation
API-key protection
Audit logging
```

Never expose:

```text
LLM API keys
Database credentials
Storage credentials
Internal tokens
```

to the browser.

---

# 43. Multi-Tenant Architecture

Do not over-engineer this initially.

However, database entities should support:

```text
user_id
organization_id
project_id
```

This makes future multi-tenancy possible.

---

# 44. API Design

Example endpoints:

```text
POST   /api/projects
GET    /api/projects

POST   /api/sources
GET    /api/sources
DELETE /api/sources/{id}

POST   /api/documents/upload
GET    /api/documents

POST   /api/query
POST   /api/query/stream

GET    /api/traces/{id}

POST   /api/evaluations
GET    /api/evaluations
POST   /api/evaluations/run

POST   /api/experiments
GET    /api/experiments
POST   /api/experiments/{id}/run

GET    /api/benchmarks
POST   /api/benchmarks/run

GET    /api/models
POST   /api/models
```

Use versioning:

```text
/api/v1/...
```

---

# 45. Streaming

AI responses should stream where practical.

Use:

```text
Server-Sent Events
```

or WebSockets where appropriate.

The frontend should be able to display execution progress:

```text
✓ Query analyzed
✓ Semantic retrieval
✓ Keyword retrieval
✓ Fusion
✓ Reranking
● Generating
○ Verification
```

---

# 46. Background Jobs

Use background workers for:

```text
Large document ingestion
OCR
Embedding generation
Indexing
Benchmark runs
Large evaluation runs
Experiments
```

Do not block HTTP requests for long-running operations.

---

# 47. Error Handling

Every stage must have explicit errors.

Example:

```text
INGESTION_FAILED
PARSING_FAILED
EMBEDDING_FAILED
RETRIEVAL_FAILED
RERANKING_FAILED
MODEL_TIMEOUT
MODEL_RATE_LIMIT
VERIFICATION_FAILED
EVALUATION_FAILED
```

Errors should appear in traces.

---

# 48. Observability

The application itself must be observable.

Track:

```text
Request ID
Trace ID
User ID
Project ID
Model
Provider
Latency
Tokens
Cost
Errors
Retries
```

Logs must be structured.

Do not log secrets.

---

# 49. Repository Structure

Recommended structure:

```text
adaptive-evidence-ai/
│
├── apps/
│   ├── web/
│   │   ├── app/
│   │   ├── components/
│   │   ├── features/
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── types/
│   │
│   └── api/
│       ├── app/
│       ├── routes/
│       ├── services/
│       ├── models/
│       ├── schemas/
│       └── workers/
│
├── core/
│   ├── ingestion/
│   ├── parsing/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranking/
│   ├── evidence/
│   ├── routing/
│   ├── reasoning/
│   └── verification/
│
├── models/
│   ├── providers/
│   ├── local/
│   ├── embeddings/
│   └── registry/
│
├── evaluation/
│   ├── datasets/
│   ├── metrics/
│   ├── judges/
│   ├── human_eval/
│   └── failure_analysis/
│
├── benchmark/
│   ├── ae_bench/
│   ├── clean/
│   ├── noisy/
│   ├── conflicting/
│   ├── outdated/
│   └── adversarial/
│
├── experiments/
│   ├── retrieval/
│   ├── routing/
│   ├── inference/
│   └── reports/
│
├── visualization/
│   ├── evidence_graph/
│   ├── embeddings/
│   ├── traces/
│   ├── retrieval/
│   ├── evaluations/
│   └── models/
│
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── api/
│   └── deployment/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── evaluation/
│   └── benchmark/
│
├── docker/
│
├── .env.example
├── docker-compose.yml
├── README.md
├── PROJECT_SPEC.md
└── LICENSE
```

---

# 50. Engineering Principles

The coding agent must follow these principles.

## Principle 1 — Modularity

Retrieval, models, evaluation, and storage must be replaceable.

Do not tightly couple business logic to a specific provider.

---

## Principle 2 — Reproducibility

Every experiment must store its configuration.

A result must be reproducible from:

```text
dataset
configuration
model
retrieval
prompt
evaluation
timestamp/version
```

---

## Principle 3 — Provenance

Every answer should be traceable back to source evidence.

---

## Principle 4 — Observable AI

AI operations must produce structured traces.

Do not create opaque "magic" functions.

---

## Principle 5 — Evaluation First

Every major AI component must be measurable.

---

## Principle 6 — No Fake Intelligence

Do not add meaningless "AI" labels.

If a routing decision is rule-based, explicitly identify it as rule-based.

---

# 51. What NOT to Build Initially

Do NOT begin with:

```text
Multi-agent swarm
Autonomous agents
Complex knowledge graph reasoning
Fine-tuning
Custom model training
Distributed GPU infrastructure
Complex Kubernetes deployment
Multi-tenant enterprise billing
```

These can be future extensions.

The core system is more important.

---

# 52. MVP

The first functional MVP must contain:

```text
✓ PDF upload
✓ Document parsing
✓ Chunking
✓ Embeddings
✓ PostgreSQL + pgvector
✓ BM25
✓ Dense retrieval
✓ Hybrid retrieval
✓ RRF
✓ Basic reranking
✓ One LLM provider
✓ Question answering
✓ Citations
✓ Trace generation
✓ Basic evaluation
✓ Basic dashboard
```

MVP flow:

```text
PDF
 ↓
Parse
 ↓
Chunk
 ↓
Embed
 ↓
Index
 ↓
Question
 ↓
Dense + BM25
 ↓
RRF
 ↓
Reranker
 ↓
LLM
 ↓
Citation
 ↓
Trace
```

---

# 53. Phase 2

Add:

```text
✓ Model abstraction
✓ Multiple providers
✓ Model Lab
✓ Query Analyzer
✓ Adaptive hybrid retrieval
✓ Evidence graph
✓ Trace visualization
✓ Retrieval comparison
✓ Evaluation dashboard
✓ Cost tracking
✓ Latency tracking
```

---

# 54. Phase 3

Add:

```text
✓ AE-Bench
✓ Controlled noise
✓ Contradictory evidence
✓ Outdated evidence
✓ Adversarial evidence
✓ Experiment management
✓ Model comparison
✓ Failure analysis
✓ Research dashboards
```

---

# 55. Phase 4

Add:

```text
✓ Adaptive model routing
✓ Adaptive inference budget
✓ Evidence-aware compute allocation
✓ Automatic experiment generation
✓ Advanced analytics
✓ Multimodal retrieval
```

---

# 56. Multimodal Extension

After the text pipeline is stable, support:

```text
PDF page images
Tables
Figures
Charts
Screenshots
```

The system should be able to retrieve visual evidence.

Potential future architecture:

```text
Document
    │
    ├── Text
    ├── Tables
    ├── Images
    └── Page Renderings
          │
          ▼
   Multimodal Index
          │
          ▼
   Multimodal Retrieval
```

The UI should allow a user to inspect the actual page/figure/table used as evidence.

---

# 57. Research Questions

The project should maintain an explicit research-question registry.

Initial questions:

### RQ1

> When does hybrid retrieval outperform semantic retrieval?

### RQ2

> Can query-aware retrieval weighting improve evidence recall?

### RQ3

> How does evidence noise affect retrieval and answer reliability?

### RQ4

> Does additional reasoning compensate for poor retrieval?

### RQ5

> Can evidence uncertainty predict when additional inference compute is useful?

### RQ6

> Can adaptive model routing reduce cost while preserving answer quality?

### RQ7

> How reliable are automated evaluators compared with human evaluation?

### RQ8

> How do retrieval strategies behave under conflicting or adversarial evidence?

---

# 58. Research Artifact

The final project should be capable of producing:

```text
Research Paper
+
Open-source Repository
+
Live Demo
+
AE-Bench Dataset/Benchmark
+
Experiment Results
+
Technical Documentation
```

---

# 59. Master's Application Positioning

The project should eventually support this narrative:

> I became interested in whether additional reasoning necessarily improves the reliability of retrieval-augmented AI systems. I developed an experimental platform that allows researchers to control evidence quality, retrieval strategy, model selection, and inference budget while measuring reliability, cost, and latency. The platform provides interactive provenance and failure analysis and includes a benchmark for evaluating RAG under noisy, conflicting, outdated, and adversarial evidence.

This should become the research narrative.

---

# 60. Resume Positioning

Do NOT describe the project as:

> Built a RAG chatbot.

Instead, eventually use language similar to:

> **Adaptive Evidence AI Lab — AI Systems Research Platform**
>
> Developed a multimodal AI experimentation platform combining adaptive hybrid retrieval, evidence-aware model routing, inference-budget allocation, provenance tracing, and automated evaluation; implemented semantic + BM25 retrieval, reranking, claim-level citations, and interactive execution traces while measuring correctness, faithfulness, latency, token usage, and inference cost.

Once real benchmark numbers exist, replace generic claims with measured results.

Example:

> Improved Recall@10 from **X% → Y%** using adaptive hybrid retrieval while reducing average inference cost by **Z%**.

Never invent these numbers.

---

# 61. Demo Strategy

The live demo should be understandable within 60 seconds.

A visitor should be able to:

```text
1. Upload a document
2. Ask a question
3. Receive an answer
4. Inspect citations
5. Open evidence
6. View retrieval methods
7. Open execution trace
8. See evaluation metrics
```

The "wow" moment should be:

> **Clicking an answer claim and visually seeing exactly which documents, pages, chunks, retrieval methods, and model decisions produced it.**

---

# 62. UI Design Direction

Visual style:

```text
Modern
Technical
Research-oriented
Minimal
Dense but understandable
Professional
```

Think:

```text
AI research laboratory
+
developer observability platform
+
data visualization system
```

Avoid:

```text
Generic ChatGPT clone
Excessive gradients
Unnecessary animations
Huge empty cards
Marketing-style landing page inside the app
```

---

# 63. Development Rules for AI Coding Agents

When an AI coding agent works on this repository:

1. Read `PROJECT_SPEC.md` before modifying architecture.
2. Inspect the existing repository before creating files.
3. Do not rewrite working code unnecessarily.
4. Do not introduce dependencies without justification.
5. Prefer modular interfaces.
6. Keep provider-specific logic isolated.
7. Add tests for important logic.
8. Never hardcode API keys.
9. Never fabricate evaluation results.
10. Never fabricate benchmark results.
11. Never claim a feature works unless it has been tested.
12. Update documentation when architecture changes.
13. Preserve type safety.
14. Use structured logging.
15. Keep database migrations versioned.
16. Keep experiment configurations reproducible.
17. Every AI operation should be traceable.
18. Every retrieval result should preserve provenance.
19. Prefer deterministic logic where possible.
20. Do not use an LLM when ordinary software logic is sufficient.

---

# 64. Agent Execution Strategy

The coding agent should work incrementally.

Do not attempt to implement the entire platform in one pass.

Follow:

```text
PHASE 0
Repository inspection
        ↓
PHASE 1
Architecture + scaffolding
        ↓
PHASE 2
Data ingestion
        ↓
PHASE 3
Dense retrieval
        ↓
PHASE 4
BM25 retrieval
        ↓
PHASE 5
Hybrid + RRF
        ↓
PHASE 6
Reranking
        ↓
PHASE 7
Generation + citations
        ↓
PHASE 8
Tracing
        ↓
PHASE 9
Evaluation
        ↓
PHASE 10
Visualization
        ↓
PHASE 11
Adaptive retrieval
        ↓
PHASE 12
Adaptive routing
        ↓
PHASE 13
AE-Bench
        ↓
PHASE 14
Experiments
        ↓
PHASE 15
Deployment
```

At the end of every phase:

```text
Run tests
Run application
Verify feature
Update documentation
Commit changes
```

---

# 65. Phase 0 — Repository Inspection

Before coding:

1. Inspect repository.
2. Identify existing code.
3. Identify current framework.
4. Identify current database.
5. Identify existing environment configuration.
6. Identify existing AI integrations.
7. Identify reusable components.
8. Identify technical debt.
9. Propose minimal changes.

Do not immediately delete or rewrite existing code.

Create:

```text
docs/INITIAL_ARCHITECTURE.md
```

containing the discovered architecture.

---

# 66. Phase 1 — Project Foundation

Create:

```text
apps/web
apps/api
core
models
evaluation
benchmark
experiments
visualization
docs
tests
```

Set up:

```text
TypeScript
Python
FastAPI
Next.js
PostgreSQL
pgvector
Redis
Docker
```

Create:

```text
.env.example
docker-compose.yml
```

Ensure the application can start locally.

---

# 67. Phase 2 — Data Pipeline

Implement:

```text
Upload
 ↓
Validation
 ↓
Parsing
 ↓
Chunking
 ↓
Metadata
 ↓
Storage
```

Initially prioritize:

```text
PDF
```

Then:

```text
DOCX
PPTX
TXT
Markdown
```

---

# 68. Phase 3 — Dense Retrieval

Implement:

```text
Embedding provider
 ↓
Vector storage
 ↓
Similarity search
 ↓
Top-K
```

Expose retrieval scores.

Create tests.

---

# 69. Phase 4 — BM25

Implement keyword retrieval.

Expose:

```text
keyword score
rank
matched terms
```

Create tests around:

```text
exact names
acronyms
technical identifiers
numbers
rare terminology
```

---

# 70. Phase 5 — Hybrid Retrieval

Implement:

```text
Dense
+
BM25
↓
RRF
↓
Unified ranking
```

Create a retrieval interface:

```python
class Retriever:
    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None
    ):
        ...
```

Implement:

```text
DenseRetriever
BM25Retriever
HybridRetriever
```

---

# 71. Phase 6 — Reranking

Add a reranker interface.

Example:

```python
class Reranker:
    async def rerank(query, candidates):
        ...
```

Initial implementation may use a cross-encoder or provider-supported reranking mechanism.

Keep it replaceable.

---

# 72. Phase 7 — Generation

Implement:

```text
Evidence Set
 ↓
Prompt Construction
 ↓
LLM
 ↓
Structured Answer
 ↓
Citation Mapping
```

Require structured output wherever practical.

---

# 73. Phase 8 — Trace System

Every query must create:

```text
trace
 ├── query
 ├── analysis
 ├── retrieval
 ├── fusion
 ├── reranking
 ├── routing
 ├── generation
 └── verification
```

Persist traces.

---

# 74. Phase 9 — Evaluation

Implement initial metrics:

```text
Recall@K
MRR
NDCG
Answer correctness
Faithfulness
Citation correctness
Latency
Tokens
Cost
```

The evaluation framework should be independent of the UI.

---

# 75. Phase 10 — Visualization

Implement:

```text
Trace Viewer
Retrieval Comparison
Evidence Graph
Evaluation Dashboard
```

Do not build every visualization simultaneously.

Prioritize:

1. Trace Viewer
2. Evidence Graph
3. Retrieval Comparison
4. Evaluation Dashboard

---

# 76. Phase 11 — Adaptive Retrieval

Implement query classification.

Start with deterministic rules.

Example:

```python
if exact_entity_detected:
    keyword_weight = 0.75
    semantic_weight = 0.25

elif conceptual_query:
    keyword_weight = 0.25
    semantic_weight = 0.75

else:
    keyword_weight = 0.5
    semantic_weight = 0.5
```

These values are initial heuristics only.

Make them configurable.

Then run experiments to determine whether adaptive weighting actually improves results.

---

# 77. Phase 12 — Adaptive Model Routing

Implement:

```text
Query difficulty
+
Evidence confidence
+
Conflict level
+
Latency budget
+
Cost budget
```

→ model selection.

Initially use deterministic policies.

Later experiment with learned routing.

---

# 78. Phase 13 — AE-Bench

Create a benchmark generation and evaluation pipeline.

Support:

```text
clean
noisy
conflicting
outdated
incomplete
adversarial
```

Each benchmark result must record:

```text
configuration
model
retriever
dataset version
metrics
runtime
timestamp
```

---

# 79. Phase 14 — Research Experiments

Implement reusable experiment runners.

Example:

```bash
python -m experiments.retrieval.compare \
  --dataset ae_bench_v1 \
  --retrievers dense,bm25,hybrid,adaptive
```

Output:

```text
experiments/results/<experiment_id>/
```

with:

```text
config.json
results.json
metrics.json
report.md
```

---

# 80. Phase 15 — Deployment

Target architecture:

```text
                     Internet
                        │
                        ▼
                  Reverse Proxy
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
        Next.js                  FastAPI
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
              PostgreSQL          Redis          Object Storage
                  │
                  ▼
               pgvector
```

The application should be deployable through Docker.

---

# 81. Environment Variables

Provide:

```text
DATABASE_URL=
REDIS_URL=

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

OBJECT_STORAGE_ENDPOINT=
OBJECT_STORAGE_BUCKET=
OBJECT_STORAGE_ACCESS_KEY=
OBJECT_STORAGE_SECRET_KEY=

NEXT_PUBLIC_API_URL=
```

Only include provider keys that are actually implemented.

---

# 82. Testing Requirements

## Unit Tests

Test:

```text
chunking
retrieval
RRF
query classification
routing
metrics
citation mapping
cost calculation
```

## Integration Tests

Test:

```text
upload → ingestion
ingestion → indexing
query → retrieval
query → generation
query → trace
evaluation → result
```

## Research Tests

Test benchmark reproducibility.

---

# 83. Quality Gates

A feature is not complete until:

```text
✓ Code implemented
✓ Unit tests pass
✓ Integration tests pass where relevant
✓ API documented
✓ UI verified
✓ Errors handled
✓ Logs structured
✓ No secrets exposed
✓ Documentation updated
```

---

# 84. Definition of Done — MVP

The MVP is complete when a user can:

```text
1. Open the web application.
2. Create a project.
3. Upload a PDF.
4. Wait for ingestion.
5. See document statistics.
6. Ask a question.
7. See semantic retrieval.
8. See keyword retrieval.
9. See hybrid ranking.
10. See reranked evidence.
11. Receive an answer.
12. Inspect citations.
13. Open the source page/chunk.
14. View execution trace.
15. See latency/tokens/cost.
16. Run an evaluation.
17. View basic evaluation metrics.
```

---

# 85. Definition of Done — Research Platform

The research platform is complete when:

```text
✓ Adaptive retrieval
✓ Model routing
✓ Adaptive inference budget
✓ AE-Bench
✓ Experiment management
✓ Failure analysis
✓ Evidence visualization
✓ Model comparison
✓ Retrieval comparison
✓ Reproducible experiments
✓ Exportable experiment reports
```

---

# 86. Future Extensions

Potential future work:

```text
Multimodal RAG
Graph RAG
Agentic retrieval
Knowledge graph reasoning
Learned retrieval routing
Learned model routing
Reinforcement learning for routing
Automatic benchmark generation
Human evaluation workflows
Federated/private retrieval
SharePoint connector
Enterprise identity
Local-only privacy mode
GPU telemetry
Distributed inference
```

These should remain outside the MVP unless required.

---

# 87. Important Research Constraint

The system must distinguish between:

```text
SYSTEM BEHAVIOR
```

and:

```text
RESEARCH HYPOTHESIS
```

Do not claim:

> "Adaptive retrieval improves accuracy."

until experiments demonstrate it.

Instead say:

> "Adaptive retrieval is being evaluated for its effect on accuracy, cost, and latency."

The platform itself should make falsifiable experiments possible.

---

# 88. Central Success Metric

The final system should optimize more than accuracy.

Think in terms of:

```text
RELIABILITY
      ×
EVIDENCE QUALITY
      ×
COST EFFICIENCY
      ×
LATENCY
```

The objective is not:

> Maximum intelligence at any cost.

The objective is:

> **The right amount of evidence and computation for the task.**

---

# 89. Final Product Vision

The finished application should feel like:

```text
                     ADAPTIVE EVIDENCE AI LAB
                              │
             ┌────────────────┼────────────────┐
             │                │                │
          DATA LAB         MODEL LAB       EVALUATION
             │                │                │
             └────────────────┼────────────────┘
                              │
                         AI PLAYGROUND
                              │
                    ┌─────────┴─────────┐
                    │                   │
               AI EXECUTION        EVIDENCE
                    │                   │
                    ▼                   ▼
                 TRACE             PROVENANCE
                    │                   │
                    └─────────┬─────────┘
                              │
                        EXPERIMENT LAB
                              │
                              ▼
                           AE-Bench
                              │
                              ▼
                       RESEARCH RESULTS
```

The application should allow someone to go from:

```text
"I have some documents."
```

to:

```text
"I can ask questions about them."
```

to:

```text
"I can see exactly what the AI did."
```

to:

```text
"I can measure whether it was correct."
```

to:

```text
"I can compare different retrieval and model strategies."
```

to:

```text
"I can run reproducible experiments."
```

to:

```text
"I can investigate a genuine AI systems research question."
```

---

# 90. Immediate Next Action for the Coding Agent

Do **not** start implementing every feature.

Start with:

```text
STEP 1
Inspect the existing repository.

STEP 2
Create docs/INITIAL_ARCHITECTURE.md.

STEP 3
Compare the existing architecture against PROJECT_SPEC.md.

STEP 4
Identify reusable code.

STEP 5
Identify missing infrastructure.

STEP 6
Propose Phase 1 implementation plan.

STEP 7
Implement only Phase 1.

STEP 8
Run tests and verify the application starts.

STEP 9
Document the implementation.

STEP 10
Stop and report what was completed, what remains, and any architectural decisions.
```

The coding agent must **not silently expand scope**.

If an implementation decision is ambiguous, choose the simplest modular solution that preserves the architecture described in this document.

---

# 91. Long-Term End State

The ultimate artifact is not merely an application.

It is:

```text
                  ONE FLAGSHIP AI SYSTEM
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       PRODUCT          RESEARCH        ENGINEERING
          │                │                │
       Live Demo        AE-Bench        Architecture
       Data Lab         Experiments     APIs
       Model Lab        Metrics         Distributed jobs
       AI Playground    Papers          Observability
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                     PORTFOLIO ARTIFACT
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          GitHub         Paper        Live Demo
```

The project should ultimately demonstrate that you can work across:

```text
AI Research
+
Information Retrieval
+
LLMs
+
Multimodal AI
+
Backend Engineering
+
Distributed Systems
+
Evaluation
+
Observability
+
Data Engineering
+
Visualization
+
Experimentation
```

without turning the project into an unmaintainable collection of disconnected features.

---

# 92. Final Principle

> **Build one deep system. Measure everything. Visualize everything important. Make every research claim experimentally testable.**

The goal is not to build the largest AI application.

The goal is to build a system where someone can open the application and immediately understand:

> **What evidence did the AI see?**

> **How did it retrieve that evidence?**

> **Why did it choose that model?**

> **How much computation did it use?**

> **Was the answer actually reliable?**

> **And can we prove that with experiments?**

That is the core identity of Adaptive Evidence AI Lab.
