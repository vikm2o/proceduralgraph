# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""A: the graph data model, its document round trip, digest stability, invariants and neighbourhoods."""

import json

import pytest

from proceduralgraph.graph import END, START, Edge, Graph, GraphError, Node, errors, warnings


def two_hop() -> Graph:
    nodes = {
        START: Node(START, "STATUS", "start"),
        "search": Node("search", "action", "Search the index"),
        "read": Node("read", "ACTION", "Read a document"),
        "answer": Node("answer", "ACTION", "Answer"),
        END: Node(END, "STATUS", "end"),
    }
    edges = (
        Edge(START, "leads_to", "search", {"condition": None, "guidance": "search first", "pitfalls": "no guessing"}),
        Edge("search", "LEADS_TO", "read", {"condition": "results found", "guidance": "read the best hit", "pitfalls": "skim"}),
        Edge("read", "PROVIDES_INPUT_FOR", "search", {"condition": "bridge term found", "guidance": "search again", "pitfalls": None}),
        Edge("read", "LEADS_TO", "answer", {"condition": "answer known", "guidance": "answer", "pitfalls": None}),
        Edge("answer", "LEADS_TO", END, {}),
    )
    return Graph(nodes=nodes, edges=edges)


def test_skeleton_and_is_skeleton():
    g = Graph.skeleton()
    assert g.is_skeleton and set(g.nodes) == {START, END}
    assert g.edges[0].key == (START, "LEADS_TO", END) and g.edges[0].attributes == {"condition": None, "guidance": None, "pitfalls": None}
    assert g.is_valid()
    assert not two_hop().is_skeleton
    custom = Graph.skeleton(relations=("NEXT",))
    assert custom.edges[0].relation == "NEXT" and custom.is_skeleton


def test_case_insensitive_types_and_relations_and_sorted_document():
    g = two_hop()
    assert g.nodes["search"].type == "ACTION"
    assert g.edges[0].relation == "LEADS_TO"
    doc = g.to_document()
    assert doc["schema"] == 1
    assert [n["id"] for n in doc["nodes"]] == sorted(g.nodes)
    assert [(e["source"], e["relation"], e["target"]) for e in doc["edges"]] == sorted(e.key for e in g.edges)
    assert set(doc["edges"][0]) == {"source", "target", "relation", "condition", "guidance", "pitfalls"}
    shuffled = Graph(nodes=dict(reversed(list(g.nodes.items()))), edges=tuple(reversed(g.edges)))
    assert shuffled.digest == g.digest
    assert Graph.from_document(doc).digest == g.digest
    assert Graph.from_document(json.loads(json.dumps(doc))).to_document() == doc


def test_refiner_json_shape_and_meta_digest():
    g = two_hop()
    refiner = g.to_refiner_json()
    assert set(refiner) == {"nodes", "edges"}
    assert set(refiner["nodes"][0]) == {"id", "type", "description"}
    with_ref = g.with_nodes({**g.nodes, "read": Node("read", "ACTION", "Read a document", ref="skill:reader", meta={"owner": "x"})})
    assert with_ref.digest != g.digest
    assert with_ref.content_digest(exclude_meta=True) == g.content_digest(exclude_meta=True)
    assert with_ref.to_refiner_json() == g.to_refiner_json()


def test_validate_reports_each_invariant():
    g = two_hop()
    assert errors(g.validate()) == []
    bad = Graph(
        nodes={START: Node(START, "STATUS"), "a": Node("a", "WIDGET"), "b": Node("b", "ACTION"), "c": Node("c", "ACTION")},
        edges=(
            Edge(START, "LEADS_TO", "a"),
            Edge("a", "FLIES_TO", "ghost"),
            Edge("b", "LEADS_TO", "c"),
            Edge("c", "LEADS_TO", "b"),
            Edge("b", "LEADS_TO", "c"),
        ),
    )
    codes = [d.code for d in errors(bad.validate())]
    assert set(codes) == {"duplicate_edge", "missing_endpoint", "missing_sentinel", "no_path_to_terminal", "unknown_node_type", "unknown_relation"}
    assert codes.count("no_path_to_terminal") == 4  # Start and a dangle into a missing endpoint; b and c cycle with no exit
    assert {d.code for d in warnings(bad.validate())} == {"unreachable_from_start"}


def test_terminal_is_zero_out_degree_not_end():
    g = Graph(
        nodes={START: Node(START, "STATUS"), END: Node(END, "STATUS"), "sink": Node("sink", "ACTION")},
        edges=(Edge(START, "LEADS_TO", "sink"), Edge(START, "LEADS_TO", END)),
    )
    assert g.terminals() == [END, "sink"]
    assert g.is_valid()


def test_neighborhood_two_hops_tags_edges():
    g = two_hop()
    n = g.neighborhood("search", 2)
    assert n.active == "search"
    assert [(h, e.target) for h, e in n.edges] == [(1, "read"), (2, "answer"), (2, "search")]
    assert n.node_ids == ("search", "read", "answer")
    assert n.edges_at(1)[0].guidance == "read the best hit"
    assert g.neighborhood("search", 1).digest != n.digest
    assert g.neighborhood(END, 2).edges == ()


def test_find_cycles_is_deterministic():
    g = two_hop()
    back = g.find_cycles()
    assert [e.key for e in back] == [("read", "PROVIDES_INPUT_FOR", "search")]


def test_from_human_completes_sentinels_and_refuses_invalid(tmp_path):
    raw = {"nodes": [{"id": "search", "type": "ACTION", "description": "s"}], "edges": [{"source": "Start", "target": "search", "relation": "LEADS_TO", "guidance": "go"},
                                                                                  {"source": "search", "target": "End", "relation": "LEADS_TO"}]}
    g, notes = Graph.from_human(raw)
    assert set(g.nodes) == {START, END, "search"}
    assert {d.code for d in notes} == {"skeleton_completed"}
    assert g.edges_between("search", END)[0].attributes == {"condition": None, "guidance": None, "pitfalls": None}
    # a missing type or relation is refused, not defaulted (G1a: "never silently repaired")
    with pytest.raises(GraphError) as info:
        Graph.from_human({"nodes": [{"id": "x"}], "edges": [{"source": "Start", "target": "x", "relation": "LEADS_TO"}, {"source": "x", "target": "End", "relation": "LEADS_TO"}]})
    assert "unknown_node_type" in {d.code for d in info.value.diagnostics}
    with pytest.raises(GraphError) as info:
        Graph.from_human({"nodes": [], "edges": [{"source": "Start", "target": "End"}]})
    assert "unknown_relation" in {d.code for d in info.value.diagnostics}
    assert g.edges_between(START, "search")[0].guidance == "go"
    path = tmp_path / "g.json"
    path.write_text(json.dumps(raw))
    again, _ = Graph.from_human(path)
    assert again.digest == g.digest
    empty, notes = Graph.from_human({"nodes": [], "edges": []})
    assert empty.is_skeleton and notes == []
    with pytest.raises(GraphError) as info:
        Graph.from_human({"nodes": [{"id": "x", "type": "ACTION"}], "edges": [{"source": "x", "target": "nowhere", "relation": "LEADS_TO"}]})
    assert "missing_endpoint" in {d.code for d in info.value.diagnostics}


def test_attribute_field_collision_is_refused():
    with pytest.raises(ValueError):
        Graph(attribute_fields=("source", "guidance"))
