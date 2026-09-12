"""Evaluation run orchestration (spec section 26).

Runs a dataset against a frozen configuration and records real, computed
metrics. Nothing here estimates or extrapolates a score (spec rule 9).

Retrieval metrics need no provider, so a run can measure retrieval quality
with no API key at all; generation is skipped in that case and its metrics are
reported as absent rather than zero.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.providers import get_answer_generator, get_embedding_pipeline
from app.core.security import Principal
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationItem,
    EvaluationResult,
    EvaluationRun,
)
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from app.services.base import Service
from core.errors import DomainError
from core.reasoning.base import InferenceBudget
from core.reasoning.grounded import GroundedAnswerGenerator
from core.retrieval.base import RetrievalQuery
from evaluation.datasets.schema import (
    Difficulty,
    EvaluationItemSpec,
    EvidenceCondition,
)
from evaluation.metrics.citation import ClaimCitations
from evaluation.scoring import ItemObservation, ItemScore, aggregate, aggregate_by, score_item

logger = get_logger(__name__)


class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _to_spec(item: EvaluationItem) -> EvaluationItemSpec:
    return EvaluationItemSpec(
        question=item.question,
        expected_answer=item.expected_answer,
        gold_chunks=tuple(item.gold_chunk_ids),
        gold_documents=tuple(item.gold_document_ids),
        difficulty=Difficulty(item.difficulty),
        evidence_condition=EvidenceCondition(item.evidence_condition),
    )


class EvaluationService(Service):
    """Creates datasets and executes evaluation runs."""

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        super().__init__(session)
        self.settings = settings

    # --- datasets --------------------------------------------------------

    async def create_dataset(
        self,
        *,
        name: str,
        description: str | None,
        items: list[EvaluationItemSpec],
        project_id: UUID,
        principal: Principal,
    ) -> EvaluationDataset:
        dataset = EvaluationDataset(
            name=name,
            description=description,
            project_id=project_id,
            organization_id=principal.organization_id,
            user_id=principal.user_id,
        )
        dataset.items = [
            EvaluationItem(
                question=spec.question,
                expected_answer=spec.expected_answer,
                gold_chunk_ids=list(spec.gold_chunks),
                gold_document_ids=list(spec.gold_documents),
                difficulty=str(spec.difficulty),
                evidence_condition=str(spec.evidence_condition),
                metadata_={},
            )
            for spec in items
        ]
        self.session.add(dataset)
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError(
                f"A dataset named {name!r} already exists in this project.",
                details={"name": name},
            ) from exc
        return dataset

    async def get_dataset(self, dataset_id: UUID, principal: Principal) -> EvaluationDataset:
        dataset = await self.session.scalar(
            select(EvaluationDataset).where(
                EvaluationDataset.id == dataset_id,
                EvaluationDataset.organization_id == principal.organization_id,
            )
        )
        if dataset is None:
            raise NotFoundError(
                "Evaluation dataset not found.", details={"dataset_id": str(dataset_id)}
            )
        return dataset

    async def list_datasets(
        self, project_id: UUID, principal: Principal
    ) -> list[EvaluationDataset]:
        result = await self.session.scalars(
            select(EvaluationDataset)
            .where(
                EvaluationDataset.project_id == project_id,
                EvaluationDataset.organization_id == principal.organization_id,
            )
            .order_by(EvaluationDataset.created_at.desc())
        )
        return list(result)

    # --- runs ------------------------------------------------------------

    async def create_run(
        self,
        *,
        dataset_id: UUID,
        project_id: UUID,
        principal: Principal,
        name: str | None,
        config: dict[str, object],
    ) -> EvaluationRun:
        """Register a run with its configuration frozen at creation time."""
        dataset = await self.get_dataset(dataset_id, principal)
        run = EvaluationRun(
            dataset_id=dataset.id,
            name=name,
            config=dict(config),
            status=RunStatus.PENDING,
            item_count=len(dataset.items),
            project_id=project_id,
            organization_id=principal.organization_id,
            user_id=principal.user_id,
        )
        self.session.add(run)
        # Committed before the job is enqueued, for the same reason ingestion
        # commits first: the worker may start before the transaction lands.
        await self.session.commit()
        return run

    async def get_run(self, run_id: UUID, principal: Principal) -> EvaluationRun:
        run = await self.session.scalar(
            select(EvaluationRun).where(
                EvaluationRun.id == run_id,
                EvaluationRun.organization_id == principal.organization_id,
            )
        )
        if run is None:
            raise NotFoundError("Evaluation run not found.", details={"run_id": str(run_id)})
        return run

    async def list_runs(self, project_id: UUID, principal: Principal) -> list[EvaluationRun]:
        result = await self.session.scalars(
            select(EvaluationRun)
            .where(
                EvaluationRun.project_id == project_id,
                EvaluationRun.organization_id == principal.organization_id,
            )
            .order_by(EvaluationRun.created_at.desc())
        )
        return list(result)

    async def execute(
        self,
        run_id: UUID,
        *,
        generator: GroundedAnswerGenerator | None = None,
    ) -> EvaluationRun:
        """Run every item and record per-item and aggregate metrics."""
        run = await self.session.get(EvaluationRun, run_id)
        if run is None:
            raise NotFoundError("Evaluation run not found.", details={"run_id": str(run_id)})

        dataset = await self.session.get(EvaluationDataset, run.dataset_id)
        if dataset is None:  # pragma: no cover - FK guarantees this
            raise NotFoundError("Evaluation dataset not found.")

        config = run.config
        strategy = str(config.get("strategy", "hybrid"))
        top_k = int(config.get("top_k", self.settings.RETRIEVAL_DEFAULT_TOP_K))
        generate = bool(config.get("generate", False))

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(UTC)
        await self.session.commit()

        owns_generator = generate and generator is None
        embeddings = None
        try:
            if strategy in {"semantic", "hybrid"}:
                embeddings = get_embedding_pipeline(self.settings)
            if owns_generator:
                generator = get_answer_generator(self.settings)

            scored: list[tuple[str, ItemScore]] = []
            for item in dataset.items:
                spec = _to_spec(item)
                observation = await self._run_item(
                    spec,
                    project_id=run.project_id,
                    strategy=strategy,
                    top_k=top_k,
                    embeddings=embeddings,
                    generator=generator if generate else None,
                )
                score = score_item(spec, observation)
                scored.append((str(spec.evidence_condition), score))

                self.session.add(
                    EvaluationResult(
                        run_id=run.id,
                        item_id=item.id,
                        status="error" if observation.error_code else "ok",
                        answer=observation.answer,
                        abstained=observation.abstained,
                        retrieved_chunk_ids=list(observation.retrieved),
                        cited_chunk_ids=sorted(observation.cited, key=str),
                        metrics=score.metrics,
                        failure_category=score.failure_category,
                        error_code=observation.error_code,
                    )
                )

            run.completed_count = sum(1 for _, score in scored if not score.metrics.get("failed"))
            run.failed_count = len(scored) - run.completed_count
            run.aggregate_metrics = {
                "overall": aggregate([score for _, score in scored]),
                # Reported per condition because pooling CLEAN with ADVERSARIAL
                # hides the contrast the platform exists to measure.
                "by_condition": aggregate_by(scored),
            }
            run.status = RunStatus.COMPLETED
            run.finished_at = datetime.now(UTC)
            await self.session.commit()

            logger.info(
                "evaluation.completed",
                run_id=str(run.id),
                items=len(scored),
                failed=run.failed_count,
            )
        except DomainError as exc:
            await self.session.rollback()
            run = await self.session.get(EvaluationRun, run_id)
            if run is not None:
                run.status = RunStatus.FAILED
                run.error_code = str(exc.code)
                run.error_message = exc.message
                run.finished_at = datetime.now(UTC)
                await self.session.commit()
            logger.warning("evaluation.failed", run_id=str(run_id), code=str(exc.code))
            raise
        finally:
            if embeddings is not None:
                await embeddings.aclose()
            if owns_generator and generator is not None:
                await generator.aclose()

        return run

    async def _run_item(
        self,
        spec: EvaluationItemSpec,
        *,
        project_id: UUID,
        strategy: str,
        top_k: int,
        embeddings: object,
        generator: GroundedAnswerGenerator | None,
    ) -> ItemObservation:
        """Execute one item, converting a failure into a recorded observation.

        One item failing must not abort the run: a partial result set with the
        failures counted is far more useful than no measurement at all.
        """
        query = RetrievalQuery(text=spec.question, project_id=project_id, top_k=top_k)
        keyword = PostgresFtsRetriever(self.session)
        dense = PgVectorRetriever(self.session, embeddings)  # type: ignore[arg-type]

        try:
            if strategy == "keyword":
                evidence = await keyword.retrieve(query)
            elif strategy == "semantic":
                evidence = await dense.retrieve(query)
            else:
                evidence = await HybridRetriever(dense, keyword).retrieve(query)

            retrieved = tuple(hit.provenance.chunk_id for hit in evidence)
            if generator is None:
                return ItemObservation(retrieved=retrieved)

            answer = await generator.generate(
                spec.question,
                evidence,
                InferenceBudget(
                    max_input_tokens=0,
                    max_output_tokens=self.settings.GENERATION_MAX_OUTPUT_TOKENS,
                ),
            )
            return ItemObservation(
                retrieved=retrieved,
                claims=tuple(
                    ClaimCitations(text=claim.text, cited=claim.evidence) for claim in answer.claims
                ),
                answer=answer.text,
                abstained=answer.abstained,
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
            )
        except DomainError as exc:
            logger.warning(
                "evaluation.item_failed", question=spec.question[:80], code=str(exc.code)
            )
            return ItemObservation(error_code=str(exc.code))
