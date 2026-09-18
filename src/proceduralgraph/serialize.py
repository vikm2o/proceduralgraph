# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Human- and model-readable renderings of a graph (paper App. B.5 "Serialized Local Graph Context"). REQUIREMENTS C5, H6.

``serialize_context`` reproduces the paper's serializer layout for the guidance prompt's ``{subgraph_summary}`` /
``{graph_summary}`` slot. ``render_markdown`` and ``render_mermaid`` are the operator's views written by
``export_workspace``. The storage is the JSON document; these are renderings.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .graph import Edge, Graph
from .traces import Step

_LABELS = {"guidance": "Guidance", "pitfalls": "Pitfalls to Avoid"}


def _label(field: str) -> str:
    return _LABELS.get(field, field.replace("_", " ").strip().capitalize())


def _transition_line(edge: Edge, *, include_relations: bool) -> str:
    condition = edge.condition if edge.condition is not None else "unconditional"
    inner = f"{edge.relation}; Condition: {condition}" if include_relations else f"Condition: {condition}"
    return f"- Transition: [{edge.source}] → [{edge.target}] ({inner})"


def _attribute_lines(edge: Edge, attribute_fields: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for name in attribute_fields:
        if name == "condition":
            continue
        value = edge.attributes.get(name)
        if name == "guidance":
            lines.append(f"* Guidance: {value if value is not None else '(none)'}")
        elif value is not None:
            lines.append(f"* {_label(name)}: {value}")
    for name, value in edge.attributes.items():  # attributes outside the schema, if a host stored any
        if name not in attribute_fields and name != "condition" and value is not None:
            lines.append(f"* {_label(name)}: {value}")
    return lines


def _hop_header(hop: int) -> str:
    if hop == 1:
        return "Immediate Transition Options (Hop 1):"
    if hop == 2:
        return "Subsequent Horizon (Hop 2):"
    return f"Further Horizon (Hop {hop}):"


def serialize_context(graph: Graph, active: str | None, hops: int = 2, *, include_relations: bool = True) -> str:
    """The App. B.5 layout. With ``active`` set: the active-node header, then transitions grouped by hop. With
    ``active=None``: the complete graph (eq. 2's fallback and the full-graph ablation).

    Decision (C5): the relation label is printed on each transition line because the refiner wrote it and it carries
    information (``[A] → [B] (LEADS_TO; Condition: …)``). ``include_relations=False`` reproduces the paper's serializer
    exactly (App. B.5 notes that its serializer does not print the stored relation labels).
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if active is None:
        return _serialize_full(graph, include_relations=include_relations)
    node = graph.node(active)
    if node is None:
        raise KeyError(active)
    lines = [f"Active Cognitive Node: [{node.id}] (Type: {node.type})", f"Description: {node.description}"]
    neighborhood = graph.neighborhood(active, hops)
    for hop in range(1, hops + 1):
        edges = neighborhood.edges_at(hop)
        if hop > 1 and not edges:
            break
        lines.append(_hop_header(hop))
        if not edges:
            lines.append("- (none: this node has no outgoing transitions; it is a terminal node)")
        for edge in edges:
            lines.append(_transition_line(edge, include_relations=include_relations))
            lines.extend(_attribute_lines(edge, graph.attribute_fields))
    return "\n".join(lines)


def _serialize_full(graph: Graph, *, include_relations: bool) -> str:
    lines = [f"Complete Procedural Graph: {len(graph.nodes)} nodes, {len(graph.edges)} transitions", "Nodes:"]
    for node in graph.nodes.values():
        lines.append(f"- [{node.id}] (Type: {node.type}): {node.description}" if node.description else f"- [{node.id}] (Type: {node.type})")
    lines.append("Transitions:")
    if not graph.edges:
        lines.append("- (none)")
    for edge in graph.edges:
        lines.append(_transition_line(edge, include_relations=include_relations))
        lines.extend(_attribute_lines(edge, graph.attribute_fields))
    return "\n".join(lines)


def render_trajectory(steps: Sequence[Step], *, window: int | None = None) -> str:
    """``{recent_context}`` (eq. 2's T_{t-w:t}): the last ``window`` steps, numbered from their absolute position."""
    if not steps:
        return "(no steps yet; the agent is at Start)"
    total = len(steps)
    if window == 0:
        return f"({total} step(s) taken; the trajectory window is 0, so none are shown)"
    chosen = list(steps) if window is None else list(steps)[-window:]
    offset = total - len(chosen)
    lines = []
    if offset:
        lines.append(f"[... {offset} earlier step(s) omitted ...]")
    for i, step in enumerate(chosen, start=offset + 1):
        lines.append(f"Step {i}:\n{step.rendered()}")
    return "\n".join(lines)


def render_markdown(graph: Graph, *, include_relations: bool = True) -> str:
    """``graph.md`` (H6): a count header, the node table, then every node serialized as the active node once with its
    immediate transitions. Also the text :func:`proceduralgraph.edits.unified_diff` compares."""
    lines = [
        "# Procedural Graph",
        "",
        f"{len(graph.nodes)} node(s), {len(graph.edges)} edge(s). Relations: {', '.join(graph.relations)}. "
        f"Attributes: {', '.join(graph.attribute_fields)}.",
        "",
        "## Nodes",
        "",
    ]
    for node in graph.nodes.values():
        extra = f" (ref: {node.ref})" if node.ref else ""
        lines.append(f"- `{node.id}` ({node.type}){extra}: {node.description}")
    lines += ["", "## Transitions by node", ""]
    for node in graph.nodes.values():
        lines.append("```")
        lines.append(serialize_context(graph, node.id, hops=1, include_relations=include_relations))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_MERMAID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def render_mermaid(graph: Graph) -> str:
    """``graph.mmd`` (H6): ``flowchart LR`` with relation labels on the edges."""
    ids = {node_id: f"n{i}" for i, node_id in enumerate(graph.nodes)}
    lines = ["flowchart LR"]
    for node_id, node in graph.nodes.items():
        label = node_id.replace('"', "'")
        shape = ("([", "])") if node.type == "STATUS" else ("[", "]")
        lines.append(f'    {ids[node_id]}{shape[0]}"{label}"{shape[1]}')
    for edge in graph.edges:
        source = ids.get(edge.source, _MERMAID_SAFE.sub("_", edge.source))
        target = ids.get(edge.target, _MERMAID_SAFE.sub("_", edge.target))
        lines.append(f"    {source} -->|{edge.relation}| {target}")
    return "\n".join(lines) + "\n"


__all__ = ["render_markdown", "render_mermaid", "render_trajectory", "serialize_context"]
