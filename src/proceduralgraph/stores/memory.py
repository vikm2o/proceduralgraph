# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""In-memory revision store for tests and examples."""

from __future__ import annotations

import asyncio
from typing import Any

from ..documents import HeadMoved, Revision


class MemoryRevisionStore:
    def __init__(self) -> None:
        self._chains: dict[tuple[str, str], list[Revision]] = {}
        self._lock = asyncio.Lock()

    async def head(self, workspace: str, kind: str) -> Revision | None:
        chain = self._chains.get((workspace, kind))
        return chain[-1] if chain else None

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision:
        async with self._lock:
            chain = self._chains.setdefault((workspace, kind), [])
            actual = chain[-1].digest if chain else None
            if actual != expected_head:
                raise HeadMoved(workspace, kind, expected_head, actual)
            revision = Revision.build(workspace, kind, document, parent=chain[-1] if chain else None, meta=meta)
            chain.append(revision)
            return revision

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None:
        for revision in self._chains.get((workspace, kind), []):
            if revision.digest == digest:
                return revision
        return None

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]:
        chain = self._chains.get((workspace, kind), [])
        return list(reversed(chain))[:limit]


__all__ = ["MemoryRevisionStore"]
