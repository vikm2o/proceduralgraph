# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Store protocols. One primitive (``RevisionStore``) and four layer stores built on it. REQUIREMENTS G3, G4, G7, H2-H4.

``GraphStore`` is the abstraction (H4a); the JSON document in a revision chain is its first implementation
(:mod:`proceduralgraph.stores.revision`). A host can implement the layer protocols directly against its own tables,
or implement only ``RevisionStore`` and use the default ``Revision*Store`` classes. ``RevisionStore`` is byte-for-byte
skillwiki's, so one host adapter serves both packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..documents import Revision
from ..edits import EditSet
from ..graph import Graph
from ..rejections import RejectionMemory
from ..traces import Trace


class RevisionStore(Protocol):
    """An append-only chain per (workspace, kind). ``append`` must refuse to fork: when ``expected_head`` is not
    the digest of the current head (or ``None`` for an empty chain) it raises :class:`proceduralgraph.documents.HeadMoved`."""

    async def head(self, workspace: str, kind: str) -> Revision | None: ...

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision: ...

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None: ...

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]: ...


@dataclass
class Candidate:
    """A materialised candidate graph G_k^cand, identified by a host-opaque ``ref`` (a digest, a bundle id, ...)."""

    ref: str
    graph: Graph
    edits: EditSet
    meta: dict[str, Any] = field(default_factory=dict)


class GraphStore(Protocol):
    """The live artefact. ``propose`` MUST be idempotent for a given ``(iteration, edits)``: a resumed run calls it
    again for the same candidate (G7), so a host that mints a durable artefact there must key it on the iteration."""

    async def load(self) -> tuple[str | None, Graph]:
        """Current head: (ref, graph). An empty chain returns ``(None, Graph.skeleton())``; the ``None`` ref tells the
        harness to seed (G1a)."""
        ...

    async def seed(self, graph: Graph, *, meta: dict[str, Any]) -> str:
        """Write revision 0 of an empty chain and return its ref. Raises ``HeadMoved`` when a graph already exists."""
        ...

    async def propose(
        self, current_ref: str | None, current: Graph, edits: EditSet, candidate: Graph, *, iteration: int, rejections_ref: str | None
    ) -> Candidate:
        """Materialise the candidate so it can be evaluated. Rejected candidates never join the chain."""
        ...

    async def accept(self, candidate: Candidate, *, expected_ref: str | None) -> str:
        """Make the candidate the new head and return its ref."""
        ...


class RejectionStore(Protocol):
    """The persistent memory H_rejected. Saved once per iteration, after the outcome is known (G4)."""

    async def load(self) -> tuple[str | None, RejectionMemory]: ...

    async def save(self, memory: RejectionMemory, *, expected_ref: str | None, iteration: int) -> str: ...


class CheckpointStore(Protocol):
    """Resume support (G7): the harness records where an iteration got to so a restart skips the paid stages already
    done. Hosts without a resume path use :class:`NullCheckpointStore`."""

    async def load(self, iteration: int) -> dict[str, Any] | None:
        """The newest checkpoint recorded for ``iteration``, or ``None``."""
        ...

    async def save(self, iteration: int, payload: dict[str, Any]) -> None: ...


class NullCheckpointStore:
    async def load(self, iteration: int) -> dict[str, Any] | None:
        return None

    async def save(self, iteration: int, payload: dict[str, Any]) -> None:
        return None


class TraceStore(Protocol):
    """Immutable; the harness writes each iteration's collected traces once."""

    async def put(self, traces: list[Trace], *, iteration: int) -> None: ...

    async def get(self, trace_id: str) -> Trace | None: ...


class NullTraceStore:
    """For hosts whose traces already live elsewhere."""

    async def put(self, traces: list[Trace], *, iteration: int) -> None:
        return None

    async def get(self, trace_id: str) -> Trace | None:
        return None


__all__ = [
    "Candidate",
    "CheckpointStore",
    "GraphStore",
    "NullCheckpointStore",
    "NullTraceStore",
    "RejectionStore",
    "RevisionStore",
    "TraceStore",
]
