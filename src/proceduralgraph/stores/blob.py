# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Revision chains on object stores, kept fork-free with backend preconditions.

Layout under ``prefix``::

    <workspace>/<kind>/HEAD                       {"digest": ..., "seq": ...}
    <workspace>/<kind>/revisions/<seq>-<digest>.json

``append`` writes the revision object create-only, then advances ``HEAD`` with an etag/generation precondition. If
two writers race, exactly one advances HEAD; the loser gets :class:`HeadMoved` and its orphaned revision object is
harmless: ``get`` and ``list`` follow the committed chain from HEAD through ``parent_digest`` and never see it.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..documents import HeadMoved, Revision


class BlobStore(Protocol):
    async def get(self, key: str) -> tuple[bytes, str] | None:
        """(data, version tag) or None when absent. The tag is whatever the backend uses for preconditions."""
        ...

    async def put_if_absent(self, key: str, data: bytes) -> bool: ...

    async def put_if_match(self, key: str, data: bytes, *, tag: str) -> bool: ...

    async def list(self, prefix: str) -> list[str]: ...


class MemoryBlobStore:
    """Reference implementation of the precondition semantics; also used by the contract test's race case."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, int]] = {}

    async def get(self, key: str) -> tuple[bytes, str] | None:
        item = self._objects.get(key)
        return (item[0], str(item[1])) if item else None

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        if key in self._objects:
            return False
        self._objects[key] = (data, 1)
        return True

    async def put_if_match(self, key: str, data: bytes, *, tag: str) -> bool:
        item = self._objects.get(key)
        if item is None or str(item[1]) != tag:
            return False
        self._objects[key] = (data, item[1] + 1)
        return True

    async def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self._objects if k.startswith(prefix))


class BlobRevisionStore:
    def __init__(self, blobs: BlobStore, *, prefix: str = "proceduralgraph"):
        self.blobs, self.prefix = blobs, prefix.strip("/")

    def _base(self, workspace: str, kind: str) -> str:
        return f"{self.prefix}/{workspace}/{kind}"

    async def _head(self, workspace: str, kind: str) -> tuple[Revision | None, str | None]:
        found = await self.blobs.get(f"{self._base(workspace, kind)}/HEAD")
        if found is None:
            return None, None
        pointer = json.loads(found[0])
        revision = await self.blobs.get(f"{self._base(workspace, kind)}/revisions/{pointer['seq']:06d}-{pointer['digest']}.json")
        if revision is None:
            raise ValueError(f"HEAD of {workspace}/{kind} points at a missing revision object")
        return Revision.from_document(json.loads(revision[0])), found[1]

    async def head(self, workspace: str, kind: str) -> Revision | None:
        return (await self._head(workspace, kind))[0]

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision:
        current, tag = await self._head(workspace, kind)
        actual = current.digest if current else None
        if actual != expected_head:
            raise HeadMoved(workspace, kind, expected_head, actual)
        revision = Revision.build(workspace, kind, document, parent=current, meta=meta)
        base = self._base(workspace, kind)
        body = json.dumps(revision.to_document(), ensure_ascii=False).encode("utf-8")
        key = f"{base}/revisions/{revision.seq:06d}-{revision.digest}.json"
        if not await self.blobs.put_if_absent(key, body):
            # Keyed by seq AND content digest, so an existing object with this key is this same document: a retry after
            # a transient failure between the two writes, not another writer. Resume at HEAD.
            existing = await self.blobs.get(key)
            if existing is None or json.loads(existing[0].decode("utf-8")).get("digest") != revision.digest:
                raise HeadMoved(workspace, kind, expected_head, "another writer created this revision")
        pointer = json.dumps({"digest": revision.digest, "seq": revision.seq}).encode("utf-8")
        advanced = (
            await self.blobs.put_if_absent(f"{base}/HEAD", pointer) if tag is None else await self.blobs.put_if_match(f"{base}/HEAD", pointer, tag=tag)
        )
        if not advanced:
            raise HeadMoved(workspace, kind, expected_head, "HEAD advanced concurrently")
        return revision

    async def _revision(self, workspace: str, kind: str, seq: int, digest: str) -> Revision | None:
        found = await self.blobs.get(f"{self._base(workspace, kind)}/revisions/{seq:06d}-{digest}.json")
        return Revision.from_document(json.loads(found[0])) if found else None

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None:
        for revision in await self.list(workspace, kind, limit=1_000_000):
            if revision.digest == digest:
                return revision
        return None

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]:
        """HEAD, then parents by ``parent_digest``: the committed chain only, so a racing loser's orphan object is
        never returned."""
        current = await self.head(workspace, kind)
        chain: list[Revision] = []
        while current is not None and len(chain) < limit:
            chain.append(current)
            if current.parent_digest is None:
                break
            current = await self._revision(workspace, kind, current.seq - 1, current.parent_digest)
        return chain


__all__ = ["BlobRevisionStore", "BlobStore", "MemoryBlobStore"]
