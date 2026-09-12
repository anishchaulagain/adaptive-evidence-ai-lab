"""Shared visualization payload types.

Plain dataclasses, free of the ORM and of FastAPI, so the builders can be
tested directly and reused by the experiment and benchmark reports.

These are render-ready: offsets, depths and matrix cells are computed here
rather than left for the client to derive, because deriving them is exactly
where a viewer gets the zero point or the nesting wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    """Node kinds in the evidence graph (spec section 18)."""

    QUERY = "query"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CLAIM = "claim"
    ANSWER = "answer"


class EdgeKind(StrEnum):
    RETRIEVED = "retrieved"
    CONTAINS = "contains"
    CITES = "cites"
    SUPPORTS = "supports"


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    type: NodeType
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    kind: EdgeKind
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Graph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TimelineBar:
    """One stage on the execution timeline (spec section 24)."""

    span_id: str
    stage: str
    status: str
    # Nesting depth, so a viewer indents a child stage under its parent without
    # having to walk parent links itself.
    depth: int
    offset_ms: float
    duration_ms: float
    error_code: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Timeline:
    trace_id: str
    total_ms: float
    status: str
    bars: list[TimelineBar] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HeatmapCell:
    row: str
    column: str
    # None where a combination was never run, which a viewer must render as a
    # gap rather than as a zero score.
    value: float | None
    n: int = 0
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class Heatmap:
    metric: str
    rows: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    cells: list[HeatmapCell] = field(default_factory=list)
