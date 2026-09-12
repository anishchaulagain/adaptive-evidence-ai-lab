"""Visualization payload builders (spec sections 18, 24, 37).

Pure functions, so the shapes a viewer depends on are pinned here rather than
inferred from a live response.
"""

from __future__ import annotations

import pytest

from visualization.evaluations.heatmap import RunInput, build_heatmap
from visualization.evidence_graph.builder import (
    ChunkInput,
    ClaimInput,
    build_evidence_graph,
    graph_summary,
)
from visualization.retrieval.comparison import StrategyResult, build_comparison
from visualization.traces.timeline import SpanInput, build_timeline
from visualization.types import EdgeKind, NodeType

pytestmark = pytest.mark.unit


# --- timeline -------------------------------------------------------------


def _span(span_id: str, parent: str | None, stage: str, offset: float) -> SpanInput:
    return SpanInput(
        span_id=span_id,
        parent_span_id=parent,
        stage=stage,
        status="ok",
        offset_ms=offset,
        duration_ms=10.0,
    )


def test_timeline_resolves_nesting_depth() -> None:
    """A viewer indents by depth; walking parent links itself is where it would
    get the nesting wrong."""
    timeline = build_timeline(
        trace_id="t",
        total_ms=100.0,
        status="ok",
        spans=[
            _span("a", None, "retrieval", 0.0),
            _span("b", "a", "semantic_retrieval", 1.0),
            _span("c", "b", "deeper", 2.0),
            _span("d", None, "generation", 20.0),
        ],
    )

    assert [(bar.stage, bar.depth) for bar in timeline.bars] == [
        ("retrieval", 0),
        ("semantic_retrieval", 1),
        ("deeper", 2),
        ("generation", 0),
    ]


def test_timeline_tolerates_a_missing_parent() -> None:
    """A truncated trace must still render rather than fail the whole view."""
    timeline = build_timeline(
        trace_id="t",
        total_ms=10.0,
        status="ok",
        spans=[_span("b", "gone", "orphan", 0.0)],
    )

    assert timeline.bars[0].depth == 0


def test_timeline_does_not_loop_on_a_cyclic_parent_link() -> None:
    """Corrupt data must not hang the endpoint."""
    spans = [_span("a", "b", "one", 0.0), _span("b", "a", "two", 1.0)]

    timeline = build_timeline(trace_id="t", total_ms=1.0, status="ok", spans=spans)

    assert len(timeline.bars) == 2


def test_timeline_preserves_span_order_and_offsets() -> None:
    timeline = build_timeline(
        trace_id="t",
        total_ms=50.0,
        status="ok",
        spans=[_span("a", None, "retrieval", 0.0), _span("b", None, "generation", 12.5)],
    )

    assert [bar.offset_ms for bar in timeline.bars] == [0.0, 12.5]
    assert timeline.total_ms == 50.0


# --- evidence graph -------------------------------------------------------


def _chunks() -> list[ChunkInput]:
    return [
        ChunkInput("c1", "d1", "Report A", "First passage.", page=1, rank=0),
        ChunkInput("c2", "d1", "Report A", "Second passage.", page=2, rank=1),
        ChunkInput("c3", "d2", "Report B", "Third passage.", page=1, rank=2),
    ]


def test_the_graph_chains_query_to_answer() -> None:
    """Spec section 18: query -> documents -> chunks -> claims -> answer."""
    graph = build_evidence_graph(
        query="why?",
        answer="because.",
        chunks=_chunks(),
        claims=[ClaimInput("A claim.", ["c1"])],
    )

    types = {node.id: node.type for node in graph.nodes}
    assert types["query"] is NodeType.QUERY
    assert types["document:d1"] is NodeType.DOCUMENT
    assert types["chunk:c1"] is NodeType.CHUNK
    assert types["claim:0"] is NodeType.CLAIM
    assert types["answer"] is NodeType.ANSWER

    kinds = {(edge.source, edge.target): edge.kind for edge in graph.edges}
    assert kinds[("query", "document:d1")] is EdgeKind.RETRIEVED
    assert kinds[("document:d1", "chunk:c1")] is EdgeKind.CONTAINS
    assert kinds[("chunk:c1", "claim:0")] is EdgeKind.CITES
    assert kinds[("claim:0", "answer")] is EdgeKind.SUPPORTS


def test_documents_are_deduplicated() -> None:
    graph = build_evidence_graph(query="q", answer="a", chunks=_chunks(), claims=[])

    documents = [n for n in graph.nodes if n.type is NodeType.DOCUMENT]
    assert len(documents) == 2


def test_uncited_chunks_still_appear() -> None:
    """What the model was given and chose *not* to use is the interesting part."""
    graph = build_evidence_graph(
        query="q", answer="a", chunks=_chunks(), claims=[ClaimInput("c", ["c1"])]
    )

    cited = {n.id: n.metadata["cited"] for n in graph.nodes if n.type is NodeType.CHUNK}
    assert cited == {"chunk:c1": True, "chunk:c2": False, "chunk:c3": False}


def test_a_citation_to_an_unretrieved_chunk_creates_no_edge() -> None:
    """A dangling citation must not produce an edge to a node that is absent."""
    graph = build_evidence_graph(
        query="q", answer="a", chunks=_chunks(), claims=[ClaimInput("c", ["missing"])]
    )

    assert not any(edge.source == "chunk:missing" for edge in graph.edges)


def test_an_unsupported_claim_is_marked() -> None:
    graph = build_evidence_graph(
        query="q", answer="a", chunks=_chunks(), claims=[ClaimInput("bare", [])]
    )

    claim = next(n for n in graph.nodes if n.type is NodeType.CLAIM)
    assert claim.metadata["supported"] is False


def test_the_summary_counts_what_matters() -> None:
    graph = build_evidence_graph(
        query="q",
        answer="a",
        chunks=_chunks(),
        claims=[ClaimInput("cited", ["c1"]), ClaimInput("bare", [])],
    )

    summary = graph_summary(graph)
    assert summary["uncited_chunks"] == 2
    assert summary["unsupported_claims"] == 1
    assert summary["node_counts"]["chunk"] == 3


def test_an_abstention_still_produces_a_graph() -> None:
    graph = build_evidence_graph(query="q", answer=None, chunks=[], claims=[], abstained=True)

    answer = next(n for n in graph.nodes if n.type is NodeType.ANSWER)
    assert answer.metadata["abstained"] is True


def test_long_labels_are_truncated() -> None:
    graph = build_evidence_graph(query="x" * 500, answer="a", chunks=[], claims=[])

    query_node = next(n for n in graph.nodes if n.type is NodeType.QUERY)
    assert len(query_node.label) < 200


# --- retrieval comparison -------------------------------------------------


def test_comparison_reports_each_strategys_rank_per_chunk() -> None:
    comparison = build_comparison(
        [
            StrategyResult("semantic", 10.0, [("a", "A", 0.9), ("b", "B", 0.8)]),
            StrategyResult("keyword", 1.0, [("b", "B", 0.5)]),
        ]
    )

    rows = {row["chunk_id"]: row for row in comparison["rows"]}
    assert rows["a"]["ranks"] == {"semantic": 0, "keyword": None}
    assert rows["b"]["ranks"] == {"semantic": 1, "keyword": 0}
    assert rows["b"]["found_by"] == ["semantic", "keyword"]


def test_rows_are_ordered_by_the_best_rank_any_strategy_gave() -> None:
    """Sorting by one strategy's ranking would bury the disagreements."""
    comparison = build_comparison(
        [
            StrategyResult("semantic", 1.0, [("a", "A", 0.9), ("b", "B", 0.8)]),
            StrategyResult("keyword", 1.0, [("c", "C", 0.5)]),
        ]
    )

    assert [row["chunk_id"] for row in comparison["rows"]][:2] == ["a", "c"]


def test_overlap_counts_what_each_strategy_uniquely_contributed() -> None:
    """A strategy contributing nothing of its own could be removed."""
    comparison = build_comparison(
        [
            StrategyResult("semantic", 1.0, [("a", "A", 0.9), ("shared", "S", 0.7)]),
            StrategyResult("keyword", 1.0, [("shared", "S", 0.4)]),
        ]
    )

    overlap = comparison["overlap"]
    assert overlap["total_chunks"] == 2
    assert overlap["found_by_all"] == 1
    assert overlap["unique_to"] == {"semantic": 1, "keyword": 0}


def test_an_empty_strategy_is_represented() -> None:
    comparison = build_comparison(
        [
            StrategyResult("semantic", 1.0, [("a", "A", 0.9)]),
            StrategyResult("keyword", 0.5, []),
        ]
    )

    assert comparison["overlap"]["unique_to"]["semantic"] == 1
    assert comparison["strategies"][1]["result_count"] == 0


# --- evaluation heatmap ---------------------------------------------------


def _run(run_id: str, strategy: str, **conditions: float) -> RunInput:
    return RunInput(
        run_id=run_id,
        strategy=strategy,
        by_condition={
            condition: {"recall@5": {"mean": value, "n": 3}}
            for condition, value in conditions.items()
        },
    )


def test_heatmap_builds_a_strategy_by_condition_matrix() -> None:
    heatmap = build_heatmap(
        [_run("r1", "hybrid", clean=0.9, noisy=0.7), _run("r2", "keyword", clean=0.4)],
        metric="recall@5",
    )

    assert heatmap.rows == ["hybrid", "keyword"]
    assert heatmap.columns == ["clean", "noisy"]
    cells = {(cell.row, cell.column): cell.value for cell in heatmap.cells}
    assert cells[("hybrid", "clean")] == pytest.approx(0.9)
    assert cells[("keyword", "clean")] == pytest.approx(0.4)


def test_a_combination_never_run_is_null_not_zero() -> None:
    """Spec section 37 forbids fabricating benchmark numbers; a gap must read
    as a gap."""
    heatmap = build_heatmap(
        [_run("r1", "hybrid", clean=0.9, noisy=0.7), _run("r2", "keyword", clean=0.4)],
        metric="recall@5",
    )

    cells = {(cell.row, cell.column): cell for cell in heatmap.cells}
    missing = cells[("keyword", "noisy")]
    assert missing.value is None
    assert missing.n == 0
    assert missing.run_id is None


def test_the_matrix_is_complete_for_every_row_and_column() -> None:
    heatmap = build_heatmap(
        [_run("r1", "hybrid", clean=0.9, noisy=0.7), _run("r2", "keyword", clean=0.4)],
        metric="recall@5",
    )

    assert len(heatmap.cells) == len(heatmap.rows) * len(heatmap.columns)


def test_the_most_recent_run_wins_for_a_repeated_combination() -> None:
    """Averaging runs would silently mix different configurations."""
    heatmap = build_heatmap(
        [_run("newest", "hybrid", clean=0.9), _run("older", "hybrid", clean=0.1)],
        metric="recall@5",
    )

    cell = next(c for c in heatmap.cells if c.row == "hybrid")
    assert cell.value == pytest.approx(0.9)
    assert cell.run_id == "newest"


def test_an_unknown_metric_yields_an_empty_matrix_not_an_error() -> None:
    heatmap = build_heatmap([_run("r1", "hybrid", clean=0.9)], metric="nonexistent")

    assert all(cell.value is None for cell in heatmap.cells)


def test_no_runs_yields_an_empty_heatmap() -> None:
    heatmap = build_heatmap([], metric="recall@5")

    assert heatmap.rows == []
    assert heatmap.cells == []
