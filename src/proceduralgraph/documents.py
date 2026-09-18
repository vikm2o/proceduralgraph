# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Canonical JSON, content digests and the immutable revision record.

Every artefact of a workspace (the graph, rejection memory, traces, checkpoints) is stored as a JSON document chained
by digest. The exported files (``graph.json``, ``graph.md``, ``graph.mmd``, ``rejections.md``) are a *rendering* of
these documents, never the storage. The shapes in this module are byte-for-byte compatible with ``skillwiki`` 0.2.0
so one host adapter can back both packages (REQUIREMENTS §6).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1
"""Version of the graph, rejection-memory and trace documents. Stored as ``"schema"`` in every document so a later
change can migrate old chains instead of guessing (REQUIREMENTS H10)."""


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in (sorted(value) if isinstance(value, (set, frozenset)) else value)]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    """Stable serialisation: sorted keys, compact separators, no NaN, UTF-8 preserved."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON form."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def check_schema(document: dict[str, Any], *, what: str) -> None:
    """Refuse documents written by a newer package version with a plain message (H10)."""
    version = document.get("schema", 1)
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise ValueError(f"{what} document has schema {version}; this proceduralgraph reads up to {SCHEMA_VERSION}. Upgrade the package.")


class HeadMoved(Exception):
    """The chain advanced since the caller read its head; the append was refused."""

    def __init__(self, workspace: str, kind: str, expected: str | None, actual: str | None):
        super().__init__(f"{workspace}/{kind}: expected head {expected!r}, found {actual!r}")
        self.workspace, self.kind, self.expected, self.actual = workspace, kind, expected, actual


@dataclass(frozen=True)
class Revision:
    """One immutable node in a per-(workspace, kind) chain.

    ``digest`` covers ``document`` only, so identical content written twice has the same digest; ``seq`` and
    ``parent_digest`` give the chain its order.
    """

    workspace: str
    kind: str
    seq: int
    digest: str
    parent_digest: str | None
    document: dict[str, Any]
    created_at: str = field(default_factory=now_iso)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        workspace: str,
        kind: str,
        document: dict[str, Any],
        *,
        parent: Revision | None,
        meta: dict[str, Any] | None = None,
    ) -> Revision:
        return cls(
            workspace=workspace,
            kind=kind,
            seq=0 if parent is None else parent.seq + 1,
            digest=digest(document),
            parent_digest=None if parent is None else parent.digest,
            document=_plain(document),
            meta=dict(meta or {}),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "kind": self.kind,
            "seq": self.seq,
            "digest": self.digest,
            "parent_digest": self.parent_digest,
            "document": self.document,
            "created_at": self.created_at,
            "meta": self.meta,
        }

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Revision:
        return cls(**{k: value[k] for k in ("workspace", "kind", "seq", "digest", "parent_digest", "document", "created_at", "meta")})

    def verify(self) -> None:
        actual = digest(self.document)
        if actual != self.digest:
            raise ValueError(f"revision {self.workspace}/{self.kind}#{self.seq} is corrupt: {actual} != {self.digest}")


__all__ = ["SCHEMA_VERSION", "HeadMoved", "Revision", "canonical_json", "check_schema", "digest", "now_iso"]
