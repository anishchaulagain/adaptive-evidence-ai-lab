"""Trace recording invariants (spec section 23).

The recorder is pure in-memory bookkeeping, so its behaviour — especially what
it does when a stage raises — is pinned here rather than inferred from an
end-to-end query.
"""

from __future__ import annotations

import pytest

from core.errors import ErrorCode, PipelineError
from core.tracing.base import SpanStatus, TraceStage
from core.tracing.recorder import TraceCollector

pytestmark = pytest.mark.unit


async def test_a_stage_produces_a_span() -> None:
    collector = TraceCollector()

    async with collector.span(TraceStage.RETRIEVAL) as span:
        span.set(candidate_count=7)

    record = collector.finish()
    assert len(record.spans) == 1
    assert record.spans[0].stage is TraceStage.RETRIEVAL
    assert record.spans[0].status is SpanStatus.OK
    assert record.spans[0].attributes["candidate_count"] == 7
    assert record.spans[0].duration_ms >= 0


async def test_attributes_can_be_set_before_and_after_the_stage_runs() -> None:
    """The interesting facts — candidate counts, model names — are only known
    once the stage has executed."""
    collector = TraceCollector()

    async with collector.span(TraceStage.GENERATION, model="m") as span:
        span.set(claims=3)

    attributes = collector.finish().spans[0].attributes
    assert attributes == {"model": "m", "claims": 3}


async def test_spans_nest_and_record_their_parent() -> None:
    """Hybrid retrieval runs arms inside a retrieval stage; the tree in spec
    section 23 is only reconstructable if parentage is recorded."""
    collector = TraceCollector()

    async with collector.span(TraceStage.RETRIEVAL):
        async with collector.span(TraceStage.SEMANTIC_RETRIEVAL):
            pass
        async with collector.span(TraceStage.KEYWORD_RETRIEVAL):
            pass

    record = collector.finish()
    by_stage = {span.stage: span for span in record.spans}
    parent = by_stage[TraceStage.RETRIEVAL]
    assert by_stage[TraceStage.SEMANTIC_RETRIEVAL].parent_span_id == parent.span_id
    assert by_stage[TraceStage.KEYWORD_RETRIEVAL].parent_span_id == parent.span_id
    assert parent.parent_span_id is None


async def test_sibling_stages_have_no_parent() -> None:
    collector = TraceCollector()

    async with collector.span(TraceStage.RETRIEVAL):
        pass
    async with collector.span(TraceStage.GENERATION):
        pass

    assert all(span.parent_span_id is None for span in collector.finish().spans)


async def test_spans_are_ordered_by_start_not_completion() -> None:
    """A nested span finishes before its parent; ordering by completion would
    render the timeline out of execution order."""
    collector = TraceCollector()

    # Written nested on purpose: the nesting is the behaviour under test, and
    # a combined `with` would obscure which stage contains which.
    async with collector.span(TraceStage.RETRIEVAL):  # noqa: SIM117
        async with collector.span(TraceStage.SEMANTIC_RETRIEVAL):
            pass

    stages = [span.stage for span in collector.finish().spans]
    assert stages == [TraceStage.RETRIEVAL, TraceStage.SEMANTIC_RETRIEVAL]


# --- failures -------------------------------------------------------------


async def test_a_failing_stage_is_still_recorded() -> None:
    """Spec section 47: errors must appear in traces. An untraced failure is
    the hardest kind to diagnose."""
    collector = TraceCollector()

    with pytest.raises(PipelineError):
        async with collector.span(TraceStage.GENERATION):
            raise PipelineError("boom", code=ErrorCode.GENERATION_FAILED, stage="generation")

    span = collector.finish().spans[0]
    assert span.status is SpanStatus.ERROR
    assert span.error_code == "GENERATION_FAILED"
    assert span.error_message == "boom"


async def test_an_unexpected_exception_is_recorded_too() -> None:
    collector = TraceCollector()

    with pytest.raises(ZeroDivisionError):
        async with collector.span(TraceStage.RETRIEVAL):
            _ = 1 / 0

    span = collector.finish().spans[0]
    assert span.status is SpanStatus.ERROR
    assert span.error_code == "INTERNAL_ERROR"


async def test_a_trace_with_a_failed_span_is_itself_failed() -> None:
    """A stage error that never surfaced at the top level is still a failed
    execution, and must not be recorded as a success."""
    collector = TraceCollector()

    with pytest.raises(PipelineError):
        async with collector.span(TraceStage.GENERATION):
            raise PipelineError("x", code=ErrorCode.GENERATION_FAILED, stage="g")

    assert collector.finish().status is SpanStatus.ERROR


async def test_an_explicit_failure_marks_the_trace() -> None:
    collector = TraceCollector()

    record = collector.finish(error_code="PROVIDER_NOT_CONFIGURED", error_message="no key")

    assert record.status is SpanStatus.ERROR
    assert record.error_code == "PROVIDER_NOT_CONFIGURED"


async def test_a_clean_run_is_ok() -> None:
    collector = TraceCollector()

    async with collector.span(TraceStage.RETRIEVAL):
        pass

    assert collector.finish().status is SpanStatus.OK


# --- metrics (spec section 25) --------------------------------------------


async def test_usage_accumulates_across_stages() -> None:
    collector = TraceCollector()
    collector.record_usage(input_tokens=100, output_tokens=20, cost=0.001)
    collector.record_usage(input_tokens=50, output_tokens=10, cost=0.0005)

    record = collector.finish()
    assert record.input_tokens == 150
    assert record.output_tokens == 30
    assert record.total_tokens == 180
    assert record.cost == pytest.approx(0.0015)


async def test_cost_stays_none_when_never_recorded() -> None:
    """Null means "no pricing configured". A zero would silently understate
    what a benchmark run cost."""
    collector = TraceCollector()
    collector.record_usage(input_tokens=100, output_tokens=20)

    assert collector.finish().cost is None


async def test_stage_latency_is_queryable_by_name() -> None:
    collector = TraceCollector()

    async with collector.span(TraceStage.RETRIEVAL):
        pass

    record = collector.finish()
    assert record.latency_of(TraceStage.RETRIEVAL) is not None
    assert record.latency_of(TraceStage.RERANKING) is None


async def test_total_latency_covers_more_than_the_spans() -> None:
    collector = TraceCollector()

    async with collector.span(TraceStage.RETRIEVAL):
        pass

    record = collector.finish()
    span_total = sum(span.duration_ms for span in record.spans)
    assert record.total_latency_ms >= span_total


async def test_trace_level_attributes_are_kept() -> None:
    collector = TraceCollector()
    collector.set(strategy="hybrid", top_k=8)

    assert collector.finish().attributes == {"strategy": "hybrid", "top_k": 8}


async def test_each_trace_gets_its_own_id() -> None:
    assert TraceCollector().trace_id != TraceCollector().trace_id
