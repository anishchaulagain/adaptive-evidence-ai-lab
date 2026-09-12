"""Evaluation lab (spec sections 26-28, 44, 74).

Independent of the UI, as the spec requires: everything here is reachable
through the API and through the service layer directly, so benchmark and
experiment runners need not go through HTTP.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep, SettingsDep
from app.models.evaluation import EvaluationDataset, EvaluationRun
from app.schemas.evaluation import (
    DatasetCreate,
    DatasetRead,
    DatasetSummary,
    EvaluationItemRead,
    ResultRead,
    RunCreate,
    RunRead,
)
from app.services.evaluation_service import EvaluationService
from app.services.project_service import ProjectService
from app.workers.queue import enqueue
from evaluation.datasets.schema import EvaluationItemSpec

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _dataset_read(dataset: EvaluationDataset) -> DatasetRead:
    return DatasetRead(
        id=dataset.id,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        project_id=dataset.project_id,
        name=dataset.name,
        description=dataset.description,
        item_count=len(dataset.items),
        items=[
            EvaluationItemRead(
                id=item.id,
                created_at=item.created_at,
                updated_at=item.updated_at,
                question=item.question,
                expected_answer=item.expected_answer,
                gold_chunks=list(item.gold_chunk_ids),
                gold_documents=list(item.gold_document_ids),
                difficulty=item.difficulty,
                evidence_condition=item.evidence_condition,
            )
            for item in dataset.items
        ],
    )


def _run_read(run: EvaluationRun) -> RunRead:
    return RunRead(
        id=run.id,
        created_at=run.created_at,
        updated_at=run.updated_at,
        dataset_id=run.dataset_id,
        project_id=run.project_id,
        name=run.name,
        config=run.config,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        item_count=run.item_count,
        completed_count=run.completed_count,
        failed_count=run.failed_count,
        aggregate_metrics=run.aggregate_metrics,
        error_code=run.error_code,
        error_message=run.error_message,
    )


# --- datasets -------------------------------------------------------------


@router.post("/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    payload: DatasetCreate,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> DatasetRead:
    project = await ProjectService(session).get(payload.project_id, principal)
    dataset = await EvaluationService(session, settings).create_dataset(
        name=payload.name,
        description=payload.description,
        items=[
            EvaluationItemSpec(
                question=item.question,
                expected_answer=item.expected_answer,
                gold_chunks=tuple(item.gold_chunks),
                gold_documents=tuple(item.gold_documents),
                difficulty=item.difficulty,
                evidence_condition=item.evidence_condition,
            )
            for item in payload.items
        ],
        project_id=project.id,
        principal=principal,
    )
    return _dataset_read(dataset)


@router.get("/datasets", response_model=list[DatasetSummary])
async def list_datasets(
    project_id: UUID,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> list[DatasetSummary]:
    datasets = await EvaluationService(session, settings).list_datasets(project_id, principal)
    return [
        DatasetSummary(
            id=dataset.id,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            project_id=dataset.project_id,
            name=dataset.name,
            description=dataset.description,
            item_count=len(dataset.items),
        )
        for dataset in datasets
    ]


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: UUID,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> DatasetRead:
    dataset = await EvaluationService(session, settings).get_dataset(dataset_id, principal)
    return _dataset_read(dataset)


# --- runs -----------------------------------------------------------------


@router.post("", response_model=RunRead, status_code=status.HTTP_201_CREATED)
async def create_run(
    payload: RunCreate,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> RunRead:
    """Register a run without starting it. Use `/run` to execute."""
    service = EvaluationService(session, settings)
    dataset = await service.get_dataset(payload.dataset_id, principal)
    run = await service.create_run(
        dataset_id=dataset.id,
        project_id=dataset.project_id,
        principal=principal,
        name=payload.name,
        config={
            "strategy": str(payload.strategy),
            "top_k": payload.top_k,
            "generate": payload.generate,
        },
    )
    return _run_read(run)


@router.post("/run", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
async def run_evaluation(
    payload: RunCreate,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> RunRead:
    """Create a run and queue it.

    Returns 202: a dataset of any size takes far longer than a request should,
    so execution happens in a worker (spec section 46).
    """
    run = await create_run(payload, principal, session, settings)
    await enqueue("run_evaluation", run.id)
    return run


@router.get("", response_model=list[RunRead])
async def list_evaluations(
    project_id: UUID,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> list[RunRead]:
    runs = await EvaluationService(session, settings).list_runs(project_id, principal)
    return [_run_read(run) for run in runs]


@router.get("/{run_id}", response_model=RunRead)
async def get_evaluation(
    run_id: UUID, principal: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> RunRead:
    run = await EvaluationService(session, settings).get_run(run_id, principal)
    return _run_read(run)


@router.get("/{run_id}/results", response_model=list[ResultRead])
async def get_evaluation_results(
    run_id: UUID, principal: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> list[ResultRead]:
    """Per-item results, for inspecting *why* an aggregate looks as it does."""
    run = await EvaluationService(session, settings).get_run(run_id, principal)
    return [
        ResultRead(
            id=result.id,
            item_id=result.item_id,
            trace_id=result.trace_id,
            status=result.status,
            answer=result.answer,
            abstained=result.abstained,
            retrieved_chunk_ids=list(result.retrieved_chunk_ids),
            cited_chunk_ids=list(result.cited_chunk_ids),
            metrics=result.metrics,
            failure_category=result.failure_category,
            error_code=result.error_code,
        )
        for result in run.results
    ]
