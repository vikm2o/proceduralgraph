# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Seed one workspace's graph from another's (H7): a new project starts from what a sibling already validated."""

from __future__ import annotations

from .base import RevisionStore
from .revision import RevisionGraphStore


async def seed_workspace(revisions: RevisionStore, *, source: str, target: str) -> str | None:
    """Copy ``source``'s head graph into an **empty** ``target`` as revision 0 with ``meta.origin = "transfer:<source>"``.

    Returns the new revision digest, or ``None`` when the source has no graph or the target already has one. Rejection
    memory is never copied: it records what the *source* validated, and the target's validation set may differ.
    """
    source_ref, graph = await RevisionGraphStore(revisions, source).load()
    if source_ref is None:
        return None
    target_store = RevisionGraphStore(revisions, target)
    target_ref, _ = await target_store.load()
    if target_ref is not None:
        return None
    return await target_store.seed(graph, meta={"origin": f"transfer:{source}", "source_ref": source_ref})


__all__ = ["seed_workspace"]
