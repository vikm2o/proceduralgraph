# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The refiner's edit set ΔG, PrepareCandidate and the structural checks (paper §3.3 Step 2, App. B.5, App. B.6).
REQUIREMENTS B.

    G_k^cand = G_{k-1} ⊕ ΔG_k, where ⊕ applies edits to a copy and performs any configured cycle repair.

Order (App. B.6): ``delete_edges`` (every relation between the endpoints), ``delete_nodes`` (with incident edges),
``add_nodes``, ``add_edges``. Then cycle repair when the policy asks for it, then :meth:`Graph.validate`. Any
error diagnostic makes the candidate invalid: no validation rollout, retained graph unchanged (Alg. 1 lines 11-13).
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .graph import END, START, Diagnostic, Edge, Graph, GraphError, Node, errors
from .serialize import render_markdown

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class EditError(GraphError):
    """The refiner's output is not a usable edit set. ``diagnostics`` carry ``malformed_edit`` entries."""


@dataclass
class EditSet:
    """The App. B.5 JSON: four arrays. ``delete_edges`` entries are ``(source, target)`` pairs and remove every
    relation between the endpoints; attribute revision is delete + add (B6)."""

    add_nodes: list[Node] = field(default_factory=list)
    delete_nodes: list[str] = field(default_factory=list)
    add_edges: list[Edge] = field(default_factory=list)
    delete_edges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.add_nodes or self.delete_nodes or self.add_edges or self.delete_edges)

    def counts(self) -> dict[str, int]:
        return {
            "add_nodes": len(self.add_nodes),
            "delete_nodes": len(self.delete_nodes),
            "add_edges": len(self.add_edges),
            "delete_edges": len(self.delete_edges),
        }

    def summary(self) -> str:
        c = self.counts()
        return f"+{c['add_nodes']} node(s) -{c['delete_nodes']} node(s) +{c['add_edges']} edge(s) -{c['delete_edges']} edge(s)"

    def to_dict(self, attribute_fields: tuple[str, ...] | None = None) -> dict[str, Any]:
        return {
            "add_nodes": [n.to_dict(extras=False) for n in self.add_nodes],
            "delete_nodes": list(self.delete_nodes),
            "add_edges": [e.to_dict(attribute_fields) for e in self.add_edges],
            "delete_edges": [{"source": s, "target": t} for s, t in self.delete_edges],
        }

    def to_json(self, attribute_fields: tuple[str, ...] | None = None) -> str:
        return json.dumps(self.to_dict(attribute_fields), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, value: Any, attribute_fields: tuple[str, ...] | None = None) -> EditSet:
        """Shape-check the four arrays; anything off is an :class:`EditError` with ``malformed_edit`` diagnostics."""
        found: list[Diagnostic] = []
        if not isinstance(value, dict):
            raise EditError([Diagnostic("malformed_edit", "the edit set must be a JSON object with four arrays")])
        known = {"add_nodes", "delete_nodes", "add_edges", "delete_edges"}
        unknown = sorted(set(value) - known)
        if not (set(value) & known):
            raise EditError([Diagnostic("malformed_edit", f"the edit set has none of the four arrays add_nodes / delete_nodes / add_edges / "
                                                          f"delete_edges (keys found: {sorted(value)})", "reply")])
        if unknown:
            found.append(Diagnostic("malformed_edit", f"unknown keys in the edit set: {unknown}", ",".join(unknown), "warning"))
        arrays: dict[str, list[Any]] = {}
        for key in ("add_nodes", "delete_nodes", "add_edges", "delete_edges"):
            raw = value.get(key, [])
            if raw is None:
                raw = []
            if not isinstance(raw, list):
                found.append(Diagnostic("malformed_edit", f"{key} must be an array", key))
                raw = []
            arrays[key] = raw
        add_nodes: list[Node] = []
        for raw in arrays["add_nodes"]:
            if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
                found.append(Diagnostic("malformed_edit", f"add_nodes entries need an id: {raw!r}", "add_nodes"))
                continue
            if not str(raw.get("type") or "").strip():
                found.append(Diagnostic("malformed_edit", f"add_nodes entry {raw.get('id')!r} has no type", str(raw.get("id"))))
                continue
            add_nodes.append(Node(id=raw["id"], type=raw["type"], description=raw.get("description") or ""))
        delete_nodes: list[str] = []
        for raw in arrays["delete_nodes"]:
            if isinstance(raw, dict) and "id" in raw:  # tolerate {"id": ...}
                raw = raw["id"]
            if not isinstance(raw, str) or not raw.strip():
                found.append(Diagnostic("malformed_edit", f"delete_nodes entries must be node ids: {raw!r}", "delete_nodes"))
                continue
            delete_nodes.append(raw)
        add_edges: list[Edge] = []
        for raw in arrays["add_edges"]:
            if not isinstance(raw, dict) or not raw.get("source") or not raw.get("target"):
                found.append(Diagnostic("malformed_edit", f"add_edges entries need source and target: {raw!r}", "add_edges"))
                continue
            if not str(raw.get("relation") or "").strip():
                found.append(Diagnostic("malformed_edit", f"add_edges entry {raw['source']}→{raw['target']} has no relation",
                                        f"{raw['source']}→{raw['target']}"))
                continue
            add_edges.append(Edge.from_dict(raw, attribute_fields))
        delete_edges: list[tuple[str, str]] = []
        for raw in arrays["delete_edges"]:
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                raw = {"source": raw[0], "target": raw[1]}
            if not isinstance(raw, dict) or not raw.get("source") or not raw.get("target"):
                found.append(Diagnostic("malformed_edit", f"delete_edges entries need source and target: {raw!r}", "delete_edges"))
                continue
            delete_edges.append((str(raw["source"]), str(raw["target"])))
        if errors(found):
            raise EditError(found)
        return cls(add_nodes=add_nodes, delete_nodes=delete_nodes, add_edges=add_edges, delete_edges=delete_edges)

    @classmethod
    def parse(cls, text: str, attribute_fields: tuple[str, ...] | None = None) -> EditSet:
        """Parse the refiner's reply: the raw JSON block, tolerating surrounding prose and code fences (B1)."""
        block = extract_json_object(text)
        if block is None:
            raise EditError([Diagnostic("malformed_edit", "no JSON object found in the refiner's reply", "reply")])
        try:
            value = json.loads(block)
        except json.JSONDecodeError as exc:
            raise EditError([Diagnostic("malformed_edit", f"the JSON block does not parse: {exc.msg} at position {exc.pos}", "reply")]) from exc
        return cls.from_dict(value, attribute_fields)


def extract_json_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text`` (inside a code fence first, if any), or ``None``."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            depth, in_string, escape = 0, False, False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        block = candidate[start : i + 1]
                        try:
                            json.loads(block)
                            return block
                        except json.JSONDecodeError:
                            break
            start = candidate.find("{", start + 1)
    return None


def repair_cycles(graph: Graph) -> tuple[Graph, list[Edge]]:
    """Remove the cycle-closing back edges a deterministic DFS finds (B3). Returns the repaired graph and the edges
    removed, in removal order."""
    removed: list[Edge] = []
    current = graph
    for _ in range(len(graph.edges) + 1):
        back = current.find_cycles()
        if not back:
            break
        drop = set(e.key for e in back)
        removed.extend(back)
        current = current.with_edges([e for e in current.edges if e.key not in drop])
    return current, removed


def prepare_candidate(
    graph: Graph,
    edits: EditSet,
    *,
    cycle_policy: str = "repair",
    available_tools: Sequence[str] | None = None,
) -> tuple[Graph | None, list[Diagnostic]]:
    """PrepareCandidate(G_{k-1}, ΔG_k, c) (App. B.6). Returns ``(candidate, diagnostics)``; the candidate is ``None``
    when any diagnostic is an error. Warnings (``cycle_repaired``, ``unreachable_from_start``, ``unknown_action``,
    ``no_such_edge``, ``no_such_node``, ``missing_guidance``) ride along and never block.

    ``available_tools`` (B4) only produces ``unknown_action`` warnings for ACTION nodes outside the list: tool-catalog
    membership is a refiner-prompt rule, not a structural check.
    """
    if cycle_policy not in ("allow", "repair"):
        raise ValueError("cycle_policy must be 'allow' or 'repair'")
    found: list[Diagnostic] = []
    nodes = dict(graph.nodes)
    edges = list(graph.edges)

    for source, target in edits.delete_edges:
        before = len(edges)
        edges = [e for e in edges if not (e.source == source and e.target == target)]
        if len(edges) == before:
            found.append(Diagnostic("no_such_edge", f"delete_edges: no edge from {source!r} to {target!r}", f"{source}→{target}", "warning"))
    for node_id in edits.delete_nodes:
        if node_id in (START, END):
            found.append(Diagnostic("reserved_node", f"delete_nodes: {node_id!r} is a sentinel and cannot be deleted", node_id))
            continue
        if node_id not in nodes:
            found.append(Diagnostic("no_such_node", f"delete_nodes: no node {node_id!r}", node_id, "warning"))
            continue
        del nodes[node_id]
        edges = [e for e in edges if e.source != node_id and e.target != node_id]
    for node in edits.add_nodes:
        if node.id in nodes:
            found.append(Diagnostic("duplicate_node", f"add_nodes: node {node.id!r} already exists", node.id))
            continue
        if not node.id.strip():
            found.append(Diagnostic("malformed_edit", "add_nodes: empty node id", node.id))
            continue
        nodes[node.id] = node
    keys = {e.key for e in edges}
    for edge in edits.add_edges:
        label = f"{edge.source} -{edge.relation}-> {edge.target}"
        normalized = Edge(edge.source, edge.relation, edge.target, {**dict.fromkeys(graph.attribute_fields), **edge.attributes})
        if normalized.key in keys:
            found.append(Diagnostic("duplicate_edge", f"add_edges: edge {label} already exists", label))
            continue
        if "guidance" in graph.attribute_fields and not normalized.guidance:
            found.append(Diagnostic("missing_guidance", f"add_edges: edge {label} has no guidance (refiner rule 3)", label, "warning"))
        keys.add(normalized.key)
        edges.append(normalized)

    candidate = replace(graph, nodes=nodes, edges=tuple(edges), meta={k: v for k, v in graph.meta.items() if k != "repairs"})
    if errors(found):
        return None, found
    if cycle_policy == "repair":
        candidate, removed = repair_cycles(candidate)
        if removed:
            for edge in removed:
                found.append(Diagnostic("cycle_repaired", f"removed the cycle-closing edge {edge.source} -{edge.relation}-> {edge.target}",
                                        f"{edge.source}→{edge.target}", "warning"))
            candidate = candidate.with_meta(repairs=[e.to_dict(graph.attribute_fields) for e in removed])
    found.extend(candidate.validate())
    if available_tools is not None:
        allowed = set(available_tools)
        for node in candidate.nodes.values():
            if node.type == "ACTION" and node.id not in allowed:
                found.append(Diagnostic("unknown_action", f"ACTION node {node.id!r} is not in the available tools list", node.id, "warning"))
    if errors(found):
        return None, found
    return candidate, found


def unified_diff(before: Graph, after: Graph, *, context: int = 3) -> str:
    """A unified diff of the two graphs' ``graph.md`` renderings (B5), for the CLI and iteration reports."""
    a = render_markdown(before).splitlines(keepends=True)
    b = render_markdown(after).splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=f"graph {before.digest[:12]}", tofile=f"graph {after.digest[:12]}", n=context))


__all__ = ["EditError", "EditSet", "extract_json_object", "prepare_candidate", "repair_cycles", "unified_diff"]
