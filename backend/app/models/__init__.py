"""SQLAlchemy ORM models — the relational schema.

Every model module is imported here so that `Base.metadata` is complete when
Alembic autogenerates a migration. Add new modules to this list.
"""

from app.db.base import Base
from app.models.benchmark import BenchmarkRun
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationItem,
    EvaluationResult,
    EvaluationRun,
)
from app.models.experiment import Experiment, ExperimentRun
from app.models.metric import MetricRecord
from app.models.model_config import ModelConfig
from app.models.project import Project
from app.models.query import Query
from app.models.source import DataSource
from app.models.trace import Trace, TraceSpan
from app.models.user import Organization, User

__all__ = [
    "Base",
    "BenchmarkRun",
    "Chunk",
    "DataSource",
    "Document",
    "EvaluationDataset",
    "EvaluationItem",
    "EvaluationResult",
    "EvaluationRun",
    "Experiment",
    "ExperimentRun",
    "MetricRecord",
    "ModelConfig",
    "Organization",
    "Project",
    "Query",
    "Trace",
    "TraceSpan",
    "User",
]
