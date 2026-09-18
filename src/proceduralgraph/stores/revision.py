# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Default layer stores over any :class:`RevisionStore` (H3): the JSON-document implementation of the store protocols.

Chain kinds are ``graph``, ``rejections``, ``traces``, ``checkpoints``. They do not collide with skillwiki's
(``wiki``, ``skills``, ``raw``, ``checkpoint``), so a host may point both packages at one table.
"""

from __future__ import annotations

from typing import Any

from ..documents import SCHEMA_VERSION, HeadMoved, Revision, check_schema
from ..edits import EditSet
from ..graph import Graph
from ..rejections import RejectionMemory
from ..traces import Trace
from .base import Candidate, RevisionStore

GRAPH, REJECTIONS, TRACES, CHECKPOINTS = "graph", "rejections", "traces", "checkpoints"
CHAIN_KINDS = (GRAPH, REJECTIONS, TRACES, CHECKPOINTS)


class RevisionGraphStore:
    """Candidates are held in memory until accepted; only accepted graphs join the chain (Alg. 1 line 17)."""

    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def load(self) -> tuple[str | None, Graph]:
        head = await self.revisions.head(self.workspace, GRAPH)
        if head is None:
            return None, Graph.skeleton()
        head.verify()
        return head.digest, Graph.from_document(head.document)

    async def seed(self, graph: Graph, *, meta: dict[str, Any]) -> str:
        """Revision 0 of an empty chain (G1a). ``HeadMoved`` when a graph already exists."""
        head = await self.revisions.head(self.workspace, GRAPH)
        if head is not None:
            raise HeadMoved(self.workspace, GRAPH, None, head.digest)
        revision = await self.revisions.append(self.workspace, GRAPH, graph.to_document(), expected_head=None, meta={"iteration": 0, **meta})
        return revision.digest

    async def propose(
        self, current_ref: str | None, current: Graph, edits: EditSet, candidate: Graph, *, iteration: int, rejections_ref: str | None
    ) -> Candidate:
        return Candidate(
            ref=candidate.digest,
            graph=candidate,
            edits=edits,
            meta={"iteration": iteration, "rejections_ref": rejections_ref, "base_ref": current_ref},
        )

    async def accept(self, candidate: Candidate, *, expected_ref: str | None) -> str:
        revision = await self.revisions.append(
            self.workspace,
            GRAPH,
            candidate.graph.to_document(),
            expected_head=expected_ref,
            meta={**candidate.meta, "edits": candidate.edits.to_dict(candidate.graph.attribute_fields), "origin": "accepted"},
        )
        return revision.digest

    async def history(self, *, limit: int = 50) -> list[Revision]:
        return await self.revisions.list(self.workspace, GRAPH, limit=limit)


class RevisionRejectionStore:
    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def load(self) -> tuple[str | None, RejectionMemory]:
        head = await self.revisions.head(self.workspace, REJECTIONS)
        if head is None:
            return None, RejectionMemory()
        head.verify()
        return head.digest, RejectionMemory.from_document(head.document)

    async def save(self, memory: RejectionMemory, *, expected_ref: str | None, iteration: int) -> str:
        revision = await self.revisions.append(
            self.workspace, REJECTIONS, memory.to_document(), expected_head=expected_ref, meta={"iteration": iteration}
        )
        return revision.digest


class RevisionTraceStore:
    """Each iteration's traces become one immutable ``traces`` revision; individual traces are looked up by id."""

    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def put(self, traces: list[Trace], *, iteration: int) -> None:
        head = await self.revisions.head(self.workspace, TRACES)
        document: dict[str, Any] = {"schema": SCHEMA_VERSION, "iteration": iteration, "traces": [t.to_document() for t in traces]}
        await self.revisions.append(self.workspace, TRACES, document, expected_head=head.digest if head else None, meta={"iteration": iteration})

    async def get(self, trace_id: str) -> Trace | None:
        for revision in await self.revisions.list(self.workspace, TRACES, limit=10_000):
            check_schema(revision.document, what="traces")
            for value in revision.document.get("traces", []):
                if value.get("id") == trace_id:
                    return Trace.from_document(value)
        return None


class RevisionCheckpointStore:
    """Each save appends one ``checkpoints`` revision; ``load`` returns the newest one for the iteration."""

    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def load(self, iteration: int) -> dict[str, Any] | None:
        revisions = sorted(await self.revisions.list(self.workspace, CHECKPOINTS, limit=200), key=lambda r: -r.seq)
        for revision in revisions:
            if revision.document.get("iteration") == iteration:
                return dict(revision.document.get("payload", {}))
        return None

    async def save(self, iteration: int, payload: dict[str, Any]) -> None:
        head = await self.revisions.head(self.workspace, CHECKPOINTS)
        document: dict[str, Any] = {"schema": SCHEMA_VERSION, "iteration": iteration, "payload": payload}
        await self.revisions.append(self.workspace, CHECKPOINTS, document, expected_head=head.digest if head else None, meta={"iteration": iteration})


__all__ = [
    "CHAIN_KINDS",
    "CHECKPOINTS",
    "GRAPH",
    "REJECTIONS",
    "TRACES",
    "RevisionCheckpointStore",
    "RevisionGraphStore",
    "RevisionRejectionStore",
    "RevisionTraceStore",
]
