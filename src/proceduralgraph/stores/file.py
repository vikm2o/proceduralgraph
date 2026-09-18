# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Filesystem revision store and the exported workspace rendering (H6).

For local runs, examples and tests. Not for clustered deployments: writers on one machine are serialised with an
advisory file lock per chain (``fcntl`` where available), but there is no cross-host locking. Layout:
``<root>/<workspace>/<kind>/HEAD`` holding ``<seq:06d>-<digest>.json`` and one JSON file per revision, written
atomically (temp file + ``os.replace``). Workspace and kind names are percent-encoded reversibly, so ``team/a`` and
``team_a`` are different directories and ``..`` cannot escape the root.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:  # POSIX advisory locks; on platforms without fcntl the per-instance asyncio lock is the only serialisation
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from ..documents import HeadMoved, Revision
from ..graph import Graph
from ..rejections import RejectionMemory
from ..serialize import render_markdown, render_mermaid

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _segment(value: str) -> str:
    """One path segment per workspace / kind name: ``[A-Za-z0-9_-]`` kept, everything else ``%XX``-encoded, so the
    mapping is injective and never contains ``.``, ``/`` or a separator."""
    if not value:
        return "%00"
    return _UNSAFE.sub(lambda m: "".join(f"%{b:02X}" for b in m.group(0).encode("utf-8")), value)


@contextlib.contextmanager
def _chain_lock(directory: Path) -> Iterator[None]:
    """Serialise writers to one chain across store instances and processes on this machine."""
    if fcntl is None:  # pragma: no cover
        yield
        return
    with open(directory / ".lock", "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FileRevisionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = asyncio.Lock()

    def _dir(self, workspace: str, kind: str) -> Path:
        return self.root / _segment(workspace) / _segment(kind)

    def _read_head(self, directory: Path) -> Revision | None:
        head = directory / "HEAD"
        if not head.exists():
            return None
        path = directory / head.read_text().strip()
        return Revision.from_document(json.loads(path.read_text(encoding="utf-8")))

    async def head(self, workspace: str, kind: str) -> Revision | None:
        return await asyncio.to_thread(self._read_head, self._dir(workspace, kind))

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision:
        async with self._lock:
            return await asyncio.to_thread(self._append, workspace, kind, document, expected_head, meta)

    def _append(self, workspace, kind, document, expected_head, meta) -> Revision:
        directory = self._dir(workspace, kind)
        directory.mkdir(parents=True, exist_ok=True)
        with _chain_lock(directory):  # the head is re-read under the lock, so two writers cannot both see the same head
            current = self._read_head(directory)
            actual = current.digest if current else None
            if actual != expected_head:
                raise HeadMoved(workspace, kind, expected_head, actual)
            revision = Revision.build(workspace, kind, document, parent=current, meta=meta)
            name = f"{revision.seq:06d}-{revision.digest}.json"
            (directory / name).write_text(json.dumps(revision.to_document(), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp = directory / f"HEAD.{os.getpid()}.tmp"
            tmp.write_text(name)
            os.replace(tmp, directory / "HEAD")  # atomic on POSIX and Windows
        return revision

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None:
        for revision in await self.list(workspace, kind, limit=1_000_000):
            if revision.digest == digest:
                return revision
        return None

    def _chain(self, directory: Path, limit: int) -> list[Revision]:
        """HEAD, then parents by ``parent_digest``: only committed revisions, never a crashed writer's orphan file."""
        current = self._read_head(directory)
        chain: list[Revision] = []
        while current is not None and len(chain) < limit:
            chain.append(current)
            if current.parent_digest is None:
                break
            parent = directory / f"{current.seq - 1:06d}-{current.parent_digest}.json"
            current = Revision.from_document(json.loads(parent.read_text(encoding="utf-8"))) if parent.exists() else None
        return chain

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]:
        directory = self._dir(workspace, kind)
        if not directory.exists():
            return []
        return await asyncio.to_thread(self._chain, directory, limit)


def export_workspace(root: str | Path, *, graph: Graph, rejections: RejectionMemory | None = None) -> list[Path]:
    """Write ``graph.json`` (refiner shape), ``graph.md``, ``graph.mmd`` and ``rejections.md`` for humans and diff
    tools (H6). This is the rendering; the storage is the chain. Returns the written paths."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    files = {
        "graph.json": json.dumps(graph.to_refiner_json(), ensure_ascii=False, indent=1) + "\n",
        "graph.md": render_markdown(graph),
        "graph.mmd": render_mermaid(graph),
        "rejections.md": (rejections or RejectionMemory()).render_markdown(),
    }
    written: list[Path] = []
    for name, text in files.items():
        path = base / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


__all__ = ["FileRevisionStore", "export_workspace"]
