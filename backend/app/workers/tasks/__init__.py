"""Background task definitions.

`TASK_FUNCTIONS` is the registry the arq worker exposes; a task is not
runnable until it is listed here.
"""

from app.workers.tasks.benchmark import run_benchmark
from app.workers.tasks.embedding import embed_document_chunks
from app.workers.tasks.evaluation import run_evaluation
from app.workers.tasks.experiment import run_experiment
from app.workers.tasks.indexing import reindex_project
from app.workers.tasks.ingestion import ingest_document, sync_source

TASK_FUNCTIONS = [
    ingest_document,
    sync_source,
    embed_document_chunks,
    reindex_project,
    run_evaluation,
    run_experiment,
    run_benchmark,
]

__all__ = ["TASK_FUNCTIONS"]
