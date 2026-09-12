"""Evidence graph payload (spec section 18).

Builds the chain the spec requires: query -> documents -> chunks -> claims ->
answer, so a reader can click a claim and walk back to the exact span of the
source that supports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visualization.types import EdgeKind, Graph, GraphEdge, GraphNode, NodeType


@dataclass(frozen=True, slots=True)
class ChunkInput:
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    rank: int | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ClaimInput:
    text: str
    cited_chunk_ids: list[str]


def _preview(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_evidence_graph(
    *,
    query: str,
    answer: str | None,
    chunks: list[ChunkInput],
    claims: list[ClaimInput],
    abstained: bool = False,
) -> Graph:
    """Assemble the graph for one answered query.

    A chunk that was retrieved but never cited still appears, connected to the
    query but not to any claim. That absence is the interesting part: it shows
    what the model was given and chose not to use.
    """
    nodes: list[GraphNode] = [GraphNode(id="query", type=NodeType.QUERY, label=_preview(query))]
    edges: list[GraphEdge] = []

    documents: dict[str, str] = {}
    for chunk in chunks:
        documents.setdefault(chunk.document_id, chunk.document_title)

    for document_id, title in documents.items():
        node_id = f"document:{document_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type=NodeType.DOCUMENT,
                label=title,
                metadata={"document_id": document_id},
            )
        )
        edges.append(GraphEdge(source="query", target=node_id, kind=EdgeKind.RETRIEVED))

    cited = {chunk_id for claim in claims for chunk_id in claim.cited_chunk_ids}
    for chunk in chunks:
        node_id = f"chunk:{chunk.chunk_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type=NodeType.CHUNK,
                label=_preview(chunk.text),
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page": chunk.page,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "rank": chunk.rank,
                    "score": chunk.score,
                    "cited": chunk.chunk_id in cited,
                },
            )
        )
        edges.append(
            GraphEdge(
                source=f"document:{chunk.document_id}",
                target=node_id,
                kind=EdgeKind.CONTAINS,
            )
        )

    answer_id = "answer"
    nodes.append(
        GraphNode(
            id=answer_id,
            type=NodeType.ANSWER,
            label=_preview(answer or "(no answer)"),
            metadata={"abstained": abstained},
        )
    )

    known = {chunk.chunk_id for chunk in chunks}
    for index, claim in enumerate(claims):
        node_id = f"claim:{index}"
        supported = bool(claim.cited_chunk_ids)
        nodes.append(
            GraphNode(
                id=node_id,
                type=NodeType.CLAIM,
                label=_preview(claim.text),
                metadata={"supported": supported},
            )
        )
        edges.append(GraphEdge(source=node_id, target=answer_id, kind=EdgeKind.SUPPORTS))
        for chunk_id in claim.cited_chunk_ids:
            # A citation to a chunk not in the retrieved set would dangle; the
            # API drops those before they reach here, so this is a guard rather
            # than an expected case.
            if chunk_id in known:
                edges.append(
                    GraphEdge(source=f"chunk:{chunk_id}", target=node_id, kind=EdgeKind.CITES)
                )

    return Graph(nodes=nodes, edges=edges)


def graph_summary(graph: Graph) -> dict[str, Any]:
    """Counts a viewer shows above the graph without walking every node."""
    counts: dict[str, int] = {}
    for node in graph.nodes:
        counts[str(node.type)] = counts.get(str(node.type), 0) + 1
    unused = sum(
        1 for node in graph.nodes if node.type is NodeType.CHUNK and not node.metadata.get("cited")
    )
    unsupported = sum(
        1
        for node in graph.nodes
        if node.type is NodeType.CLAIM and not node.metadata.get("supported")
    )
    return {
        "node_counts": counts,
        "edge_count": len(graph.edges),
        # Retrieved but never cited: evidence paid for and not used.
        "uncited_chunks": unused,
        "unsupported_claims": unsupported,
    }
