# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The Procedural Graph data model (paper §3.1, App. B.4, App. B.5). REQUIREMENTS A.

    G = (V, R, E, Φ): nodes V, relation vocabulary R, attributed directed triplets E, attribute mapping Φ.

:class:`Graph` is an immutable value object. Every operation returns a new graph; the online :class:`Guide` and the
offline loop never mutate one in place. The stored document (``to_document``) is a superset of the refiner-facing
JSON (``to_refiner_json``): it adds ``schema``, the vocabularies and the optional ``ref`` / ``meta`` on nodes.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .documents import SCHEMA_VERSION, check_schema, digest

START, END = "Start", "End"
DEFAULT_RELATIONS = ("LEADS_TO", "TRIGGERS", "PROVIDES_INPUT_FOR", "CONVERGES_TO")
DEFAULT_ATTRIBUTE_FIELDS = ("condition", "guidance", "pitfalls")
DEFAULT_NODE_TYPES = ("ACTION", "REASONING", "STATUS")
STRUCTURAL_EDGE_KEYS = ("source", "target", "relation")
_RESERVED_ATTRIBUTE_NAMES = frozenset(STRUCTURAL_EDGE_KEYS) | {"id"}


@dataclass(frozen=True)
class Diagnostic:
    """One structural finding. ``severity`` is ``"error"`` (the candidate is invalid, Alg. 1 line 11) or
    ``"warning"`` (recorded, never blocking). Codes are stable identifiers tests and hosts may match on (A6)."""

    code: str
    message: str
    subject: str | None = None
    severity: str = "error"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "subject": self.subject, "severity": self.severity}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Diagnostic:
        return cls(code=value["code"], message=value.get("message", ""), subject=value.get("subject"), severity=value.get("severity", "error"))

    def __str__(self) -> str:
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity}:{self.code}{where}: {self.message}"


def errors(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if d.is_error]


def warnings(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if not d.is_error]


def _vocabulary(value: Mapping[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """A stored vocabulary list; ``None`` / absent means the default, an explicit empty list means empty (A4)."""
    found = value.get(key)
    return default if found is None else tuple(found)


class GraphError(ValueError):
    """A graph or edit that fails the structural checks. ``diagnostics`` says why."""

    def __init__(self, diagnostics: list[Diagnostic], message: str | None = None):
        self.diagnostics = list(diagnostics)
        super().__init__(message or "; ".join(str(d) for d in self.diagnostics) or "invalid graph")


@dataclass(frozen=True)
class Node:
    """``type`` is stored upper-case (A6: case-insensitive matching). ``ref`` and ``meta`` are host extras (A8)."""

    id: str
    type: str
    description: str = ""
    ref: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "type", str(self.type).strip().upper())
        object.__setattr__(self, "description", "" if self.description is None else str(self.description))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))  # read-only: a Node is shared between graphs

    def to_dict(self, *, extras: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.id, "type": self.type, "description": self.description}
        if extras:
            value["ref"] = self.ref
            value["meta"] = dict(self.meta)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Node:
        return cls(
            id=value["id"],
            type=value.get("type") or "",
            description=value.get("description") or "",
            ref=value.get("ref"),
            meta=dict(value.get("meta") or {}),
        )


@dataclass(frozen=True)
class Edge:
    """A directed attributed triplet ``(source, relation, target)`` with Φ(e) in ``attributes``. Relation labels are
    stored upper-case. ``condition`` / ``guidance`` / ``pitfalls`` read the paper's default fields."""

    source: str
    relation: str
    target: str
    attributes: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "relation", str(self.relation).strip().upper())
        # read-only: an Edge is shared between a graph and the candidates derived from it (A1, C4)
        object.__setattr__(self, "attributes", MappingProxyType({str(k): (None if v is None else str(v)) for k, v in dict(self.attributes).items()}))

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.relation, self.target)

    @property
    def condition(self) -> str | None:
        return self.attributes.get("condition")

    @property
    def guidance(self) -> str | None:
        return self.attributes.get("guidance")

    @property
    def pitfalls(self) -> str | None:
        return self.attributes.get("pitfalls")

    def to_dict(self, attribute_fields: tuple[str, ...] | None = None) -> dict[str, Any]:
        """The refiner's flat shape: structural keys first, then every attribute field (missing ones as ``null``)."""
        fields = attribute_fields if attribute_fields is not None else tuple(self.attributes)
        value: dict[str, Any] = {"source": self.source, "target": self.target, "relation": self.relation}
        for name in fields:
            value[name] = self.attributes.get(name)
        for name, v in self.attributes.items():  # attributes outside the schema still travel, after the schema ones
            if name not in value:
                value[name] = v
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any], attribute_fields: tuple[str, ...] | None = None) -> Edge:
        """Inverse of :meth:`to_dict`. With ``attribute_fields`` every field is present (``None`` when absent);
        without it, every non-structural key becomes an attribute."""
        attributes: dict[str, str | None] = {}
        if attribute_fields is not None:
            for name in attribute_fields:
                attributes[name] = value.get(name)
        for name, v in value.items():
            if name not in STRUCTURAL_EDGE_KEYS and name not in attributes:
                attributes[name] = v
        return cls(source=value["source"], relation=value.get("relation") or "", target=value["target"], attributes=attributes)


@dataclass(frozen=True)
class Neighborhood:
    """N_h(u) (eq. 2): the active node plus the outgoing transitions reached in up to ``hops`` steps, each edge tagged
    with its hop distance (A7). ``edges`` is ordered by hop, then edge sort order."""

    active: str
    hops: int
    edges: tuple[tuple[int, Edge], ...]
    node_ids: tuple[str, ...]

    @property
    def digest(self) -> str:
        return digest({"active": self.active, "hops": self.hops, "edges": [[h, e.to_dict()] for h, e in self.edges]})

    def edges_at(self, hop: int) -> list[Edge]:
        return [e for h, e in self.edges if h == hop]


@dataclass(frozen=True)
class Graph:
    """An immutable Procedural Graph. Construct with ``Graph(nodes=..., edges=...)`` or the class methods; nodes are
    kept sorted by id and edges by ``(source, relation, target)`` so ``digest`` is order-independent (A4)."""

    nodes: Mapping[str, Node] = field(default_factory=dict)
    edges: tuple[Edge, ...] = ()
    relations: tuple[str, ...] = DEFAULT_RELATIONS
    attribute_fields: tuple[str, ...] = DEFAULT_ATTRIBUTE_FIELDS
    node_types: tuple[str, ...] = DEFAULT_NODE_TYPES
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = self.nodes.values() if isinstance(self.nodes, Mapping) else self.nodes
        ordered = {n.id: n for n in sorted(nodes, key=lambda n: n.id)}
        object.__setattr__(self, "nodes", MappingProxyType(ordered))
        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda e: e.key)))
        object.__setattr__(self, "relations", tuple(dict.fromkeys(str(r).strip().upper() for r in self.relations)))
        object.__setattr__(self, "node_types", tuple(dict.fromkeys(str(t).strip().upper() for t in self.node_types)))
        fields = tuple(dict.fromkeys(str(f) for f in self.attribute_fields))
        clash = [f for f in fields if f in _RESERVED_ATTRIBUTE_NAMES]
        if clash:
            raise ValueError(f"attribute_fields may not use the structural names {sorted(_RESERVED_ATTRIBUTE_NAMES)}: {clash}")
        object.__setattr__(self, "attribute_fields", fields)
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    # -- construction -------------------------------------------------------------------------------------------------

    @classmethod
    def skeleton(cls, **overrides: Any) -> Graph:
        """G_skeleton = (Start → End) (App. D.2): one ``LEADS_TO`` edge with empty attributes (A3)."""
        base = cls(**overrides)
        return replace(
            base,
            nodes={START: Node(START, "STATUS", "Task start"), END: Node(END, "STATUS", "Task end")},
            edges=(Edge(START, base.relations[0] if base.relations else "LEADS_TO", END, dict.fromkeys(base.attribute_fields)),),
        )

    @property
    def is_skeleton(self) -> bool:
        """True iff the node set is exactly ``{Start, End}`` and there is one edge (structural test, A3)."""
        return set(self.nodes) == {START, END} and len(self.edges) == 1

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Graph:
        """Strict inverse of :meth:`to_document`: no completion, no repair (A4, G1a)."""
        check_schema(value, what="graph")
        attribute_fields = _vocabulary(value, "attribute_fields", DEFAULT_ATTRIBUTE_FIELDS)
        return cls(
            nodes={n["id"]: Node.from_dict(n) for n in value.get("nodes", [])},
            edges=tuple(Edge.from_dict(e, attribute_fields) for e in value.get("edges", [])),
            relations=_vocabulary(value, "relations", DEFAULT_RELATIONS),
            attribute_fields=attribute_fields,
            node_types=_vocabulary(value, "node_types", DEFAULT_NODE_TYPES),
            meta=dict(value.get("meta") or {}),
        )

    @classmethod
    def from_human(cls, value: dict[str, Any] | Path, **vocab: Any) -> tuple[Graph, list[Diagnostic]]:
        """Load a hand-written graph in the refiner JSON shape ``{"nodes": [...], "edges": [...]}`` (G1a).

        ``value`` is a dict or a :class:`Path` to a JSON file. Missing ``Start`` / ``End`` nodes are added
        (``skeleton_completed`` warning) and missing attribute fields become ``null``. Nothing else is completed: a
        node without a type or an edge without a relation fails :meth:`validate` and raises :class:`GraphError`, as
        does any other structural error. Human input is refused with diagnostics, never silently repaired. ``vocab``
        may override ``relations`` / ``attribute_fields`` / ``node_types`` when the file does not carry them. Returns
        the graph and the warnings.
        """
        if isinstance(value, Path):
            value = json.loads(value.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise GraphError([Diagnostic("malformed_edit", "a graph document must be a JSON object")])
        check_schema(value, what="graph")
        notes: list[Diagnostic] = []
        relations = _vocabulary(value, "relations", _vocabulary(vocab, "relations", DEFAULT_RELATIONS))
        attribute_fields = _vocabulary(value, "attribute_fields", _vocabulary(vocab, "attribute_fields", DEFAULT_ATTRIBUTE_FIELDS))
        node_types = _vocabulary(value, "node_types", _vocabulary(vocab, "node_types", DEFAULT_NODE_TYPES))
        nodes: dict[str, Node] = {}
        for raw in value.get("nodes") or []:
            if not isinstance(raw, dict) or not raw.get("id"):
                raise GraphError([Diagnostic("malformed_edit", f"node entries need an id: {raw!r}")])
            nodes[str(raw["id"])] = Node.from_dict(raw)
        empty_input = not (value.get("nodes") or value.get("edges"))  # "empty" means the skeleton (G1a): no warnings
        for sentinel, description in ((START, "Task start"), (END, "Task end")):
            if sentinel not in nodes:
                nodes[sentinel] = Node(sentinel, "STATUS", description)
                if not empty_input:
                    notes.append(Diagnostic("skeleton_completed", f"added the missing {sentinel} node", sentinel, "warning"))
        edges: list[Edge] = []
        for raw in value.get("edges") or []:
            if not isinstance(raw, dict) or "source" not in raw or "target" not in raw:
                raise GraphError([Diagnostic("malformed_edit", f"edge entries need source and target: {raw!r}")])
            edges.append(Edge.from_dict(raw, attribute_fields))
        if not edges and set(nodes) == {START, END}:
            edges.append(Edge(START, relations[0], END, dict.fromkeys(attribute_fields)))
            if not empty_input:
                notes.append(Diagnostic("skeleton_completed", "added the Start → End edge", f"{START}→{END}", "warning"))
        graph = cls(nodes=nodes, edges=tuple(edges), relations=relations, attribute_fields=attribute_fields, node_types=node_types,
                    meta=dict(value.get("meta") or {}))
        found = graph.validate()
        if errors(found):
            raise GraphError(errors(found), "the hand-written graph fails the structural checks: " + "; ".join(str(d) for d in errors(found)))
        return graph, notes + warnings(found)

    # -- documents ----------------------------------------------------------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict(self.attribute_fields) for e in self.edges],
            "relations": list(self.relations),
            "attribute_fields": list(self.attribute_fields),
            "node_types": list(self.node_types),
            "meta": dict(self.meta),
        }

    def to_refiner_json(self) -> dict[str, Any]:
        """The App. B.5 ``{current_graph_json}`` shape: nodes as ``{id, type, description}``, edges flat."""
        return {
            "nodes": [n.to_dict(extras=False) for n in self.nodes.values()],
            "edges": [e.to_dict(self.attribute_fields) for e in self.edges],
        }

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical document; content-addressed and order-independent (A4)."""
        return digest(self.to_document())

    def content_digest(self, *, exclude_meta: bool = False) -> str:
        """``digest`` by default; with ``exclude_meta`` the node ``ref`` / ``meta`` and graph ``meta`` are left out (A8)."""
        if not exclude_meta:
            return self.digest
        value = self.to_document()
        value["nodes"] = [n.to_dict(extras=False) for n in self.nodes.values()]
        value.pop("meta", None)
        return digest(value)

    # -- queries ------------------------------------------------------------------------------------------------------

    def node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def successors(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def predecessors(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def edges_between(self, source: str, target: str) -> list[Edge]:
        return [e for e in self.edges if e.source == source and e.target == target]

    def out_degree(self, node_id: str) -> int:
        return sum(1 for e in self.edges if e.source == node_id)

    def terminals(self) -> list[str]:
        """Nodes with zero out-degree (App. B.6: the terminal check is not specifically about ``End``)."""
        return [n for n in self.nodes if self.out_degree(n) == 0]

    def neighborhood(self, node_id: str, hops: int) -> Neighborhood:
        """N_h(u): BFS over outgoing edges for up to ``hops`` steps; deterministic (A7)."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        seen = {node_id}
        frontier = [node_id]
        tagged: list[tuple[int, Edge]] = []
        order = [node_id]
        for hop in range(1, max(hops, 0) + 1):
            next_frontier: list[str] = []
            for source in frontier:
                for edge in self.successors(source):
                    tagged.append((hop, edge))
                    if edge.target not in seen:
                        seen.add(edge.target)
                        next_frontier.append(edge.target)
                        order.append(edge.target)
            frontier = next_frontier
            if not frontier:
                break
        return Neighborhood(active=node_id, hops=hops, edges=tuple(tagged), node_ids=tuple(order))

    def reachable_from(self, node_id: str) -> set[str]:
        seen = {node_id}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self.successors(current):
                if edge.target not in seen:
                    seen.add(edge.target)
                    queue.append(edge.target)
        return seen

    # -- invariants ---------------------------------------------------------------------------------------------------

    def validate(self) -> list[Diagnostic]:
        """A6 plus App. B.6: endpoints exist, relation and node types are in the vocabularies, ``Start`` and ``End``
        exist, no duplicate triple, every node has a directed path to a zero-out-degree node. Nodes unreachable from
        ``Start`` are a warning, never a failure (§3.4)."""
        found: list[Diagnostic] = []
        for sentinel in (START, END):
            if sentinel not in self.nodes:
                found.append(Diagnostic("missing_sentinel", f"the graph has no {sentinel} node", sentinel))
        for node in self.nodes.values():
            if not node.id.strip():
                found.append(Diagnostic("malformed_edit", "a node has an empty id", node.id))
            if node.type not in self.node_types:
                found.append(Diagnostic("unknown_node_type", f"node {node.id!r} has type {node.type!r}; allowed: {', '.join(self.node_types)}", node.id))
        seen_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            label = f"{edge.source} -{edge.relation}-> {edge.target}"
            for endpoint in (edge.source, edge.target):
                if endpoint not in self.nodes:
                    found.append(Diagnostic("missing_endpoint", f"edge {label} references the unknown node {endpoint!r}", label))
            if edge.relation not in self.relations:
                found.append(Diagnostic("unknown_relation", f"edge {label} uses relation {edge.relation!r}; allowed: {', '.join(self.relations)}", label))
            if edge.key in seen_keys:
                found.append(Diagnostic("duplicate_edge", f"edge {label} appears more than once", label))
            seen_keys.add(edge.key)
            extra = [k for k in edge.attributes if k not in self.attribute_fields]
            if extra:
                found.append(Diagnostic("unknown_attribute", f"edge {label} carries attributes outside the schema: {extra}", label, "warning"))
        # Path to a terminal: reverse BFS from every zero-out-degree node.
        terminals = set(self.terminals())
        can_finish = set(terminals)
        queue = deque(terminals)
        while queue:
            current = queue.popleft()
            for edge in self.predecessors(current):
                if edge.source in self.nodes and edge.source not in can_finish:
                    can_finish.add(edge.source)
                    queue.append(edge.source)
        for node_id in self.nodes:
            if node_id not in can_finish:
                found.append(Diagnostic("no_path_to_terminal", f"node {node_id!r} has no directed path to a terminal (zero out-degree) node", node_id))
        if START in self.nodes:
            reachable = self.reachable_from(START)
            for node_id in self.nodes:
                if node_id not in reachable:
                    found.append(Diagnostic("unreachable_from_start", f"node {node_id!r} is not reachable from Start", node_id, "warning"))
        return found

    def is_valid(self) -> bool:
        return not errors(self.validate())

    # -- derivation ---------------------------------------------------------------------------------------------------

    def with_meta(self, **meta: Any) -> Graph:
        return replace(self, meta={**self.meta, **meta})

    def with_nodes(self, nodes: Mapping[str, Node]) -> Graph:
        return replace(self, nodes=dict(nodes))

    def with_edges(self, edges: tuple[Edge, ...] | list[Edge]) -> Graph:
        return replace(self, edges=tuple(edges))

    def find_cycles(self) -> list[Edge]:
        """The back edges a deterministic DFS (nodes in sorted id order, edges in sort order) finds; removing them
        makes the graph acyclic (B3)."""
        colour: dict[str, int] = {}
        back: list[Edge] = []

        def visit(node_id: str) -> None:
            colour[node_id] = 1
            for edge in self.successors(node_id):
                state = colour.get(edge.target, 0)
                if state == 1:
                    back.append(edge)
                elif state == 0 and edge.target in self.nodes:
                    visit(edge.target)
            colour[node_id] = 2

        for node_id in self.nodes:
            if colour.get(node_id, 0) == 0:
                visit(node_id)
        return back

    def summary(self) -> str:
        return f"{len(self.nodes)} node(s), {len(self.edges)} edge(s), digest {self.digest[:12]}"


__all__ = [
    "DEFAULT_ATTRIBUTE_FIELDS",
    "DEFAULT_NODE_TYPES",
    "DEFAULT_RELATIONS",
    "END",
    "START",
    "Diagnostic",
    "Edge",
    "Graph",
    "GraphError",
    "Neighborhood",
    "Node",
    "errors",
    "warnings",
]
