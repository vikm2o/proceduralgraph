# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""B: edit parsing, PrepareCandidate order and the structural checks (App. B.6). Acceptance §8.6 and half of §8.5."""

import pytest

from proceduralgraph.edits import (
    EditError,
    EditSet,
    extract_json_object,
    prepare_candidate,
    repair_cycles,
    unified_diff,
)
from proceduralgraph.graph import END, START, Edge, Node, errors, warnings

from .test_graph import two_hop


def test_parse_tolerates_prose_and_fences():
    text = 'Here are my edits:\n```json\n{"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": []}\n```\nDone.'
    edits = EditSet.parse(text)
    assert edits.is_empty
    raw = '{"add_nodes": [{"id": "x", "type": "action", "description": "d"}], "delete_nodes": ["y"], ' \
          '"add_edges": [{"source": "a", "target": "b", "relation": "leads_to", "condition": null, "guidance": "g", "pitfalls": "p"}], ' \
          '"delete_edges": [{"source": "a", "target": "c"}]}'
    edits = EditSet.parse("Sure. " + raw + " trailing { not json")
    assert edits.add_nodes[0].type == "ACTION" and edits.delete_nodes == ["y"]
    assert edits.add_edges[0].key == ("a", "LEADS_TO", "b") and edits.add_edges[0].guidance == "g"
    assert edits.delete_edges == [("a", "c")]
    assert edits.counts() == {"add_nodes": 1, "delete_nodes": 1, "add_edges": 1, "delete_edges": 1}
    assert EditSet.parse(edits.to_json()).to_dict() == edits.to_dict()


def test_parse_reports_malformed_edits():
    with pytest.raises(EditError) as info:
        EditSet.parse("I cannot decide.")
    assert info.value.diagnostics[0].code == "malformed_edit"
    with pytest.raises(EditError):
        EditSet.parse('{"add_nodes": [{"type": "ACTION"}], "delete_nodes": [], "add_edges": [], "delete_edges": []}')
    with pytest.raises(EditError):
        EditSet.parse('{"add_nodes": "nope"}')
    assert extract_json_object("nothing here") is None
    assert extract_json_object('text {"a": {"b": "}"}} more') == '{"a": {"b": "}"}}'


def test_delete_edges_removes_every_relation_and_add_applies_after():
    """§8.6: one delete_edges entry removes all relations between the endpoints; add_edges is applied afterwards."""
    g = two_hop().with_edges(list(two_hop().edges) + [Edge("search", "TRIGGERS", "read", {"guidance": "also"})])
    assert len(g.edges_between("search", "read")) == 2
    edits = EditSet(
        delete_edges=[("search", "read")],
        add_edges=[Edge("search", "PROVIDES_INPUT_FOR", "read", {"condition": "c", "guidance": "kept, rewritten", "pitfalls": None})],
    )
    candidate, diagnostics = prepare_candidate(g, edits)
    assert candidate is not None and errors(diagnostics) == []
    kept = candidate.edges_between("search", "read")
    assert [e.relation for e in kept] == ["PROVIDES_INPUT_FOR"] and kept[0].guidance == "kept, rewritten"
    # attribute revision = delete + re-add of the same triple (B6): not a duplicate because deletes run first
    revise = EditSet(delete_edges=[("read", "answer")], add_edges=[Edge("read", "LEADS_TO", "answer", {"guidance": "new"})])
    revised, diagnostics = prepare_candidate(g, revise)
    assert revised is not None and revised.edges_between("read", "answer")[0].guidance == "new"


def test_structural_failures_are_errors_and_candidate_is_none():
    g = two_hop()
    missing = EditSet(add_edges=[Edge("read", "LEADS_TO", "ghost", {"guidance": "g"})])
    candidate, diagnostics = prepare_candidate(g, missing)
    assert candidate is None and {d.code for d in errors(diagnostics)} >= {"missing_endpoint"}
    reserved = EditSet(delete_nodes=[END])
    candidate, diagnostics = prepare_candidate(g, reserved)
    assert candidate is None and [d.code for d in errors(diagnostics)] == ["reserved_node"]
    duplicate_node = EditSet(add_nodes=[Node("search", "ACTION")])
    assert prepare_candidate(g, duplicate_node)[0] is None
    duplicate_edge = EditSet(add_edges=[Edge("search", "LEADS_TO", "read", {"guidance": "again"})])
    candidate, diagnostics = prepare_candidate(g, duplicate_edge)
    assert candidate is None and [d.code for d in errors(diagnostics)] == ["duplicate_edge"]
    unknown_type = EditSet(add_nodes=[Node("x", "WIDGET")], add_edges=[Edge("search", "LEADS_TO", "x", {"guidance": "g"}), Edge("x", "LEADS_TO", END, {"guidance": "g"})])
    candidate, diagnostics = prepare_candidate(g, unknown_type)
    assert candidate is None and "unknown_node_type" in {d.code for d in errors(diagnostics)}
    unknown_relation = EditSet(add_edges=[Edge("search", "FLIES_TO", "answer", {"guidance": "g"})])
    assert "unknown_relation" in {d.code for d in errors(prepare_candidate(g, unknown_relation)[1])}
    # A node left with no path to a terminal
    dead_end = EditSet(add_nodes=[Node("loop", "REASONING")], add_edges=[Edge("search", "LEADS_TO", "loop", {"guidance": "g"}), Edge("loop", "LEADS_TO", "loop", {"guidance": "g"})])
    candidate, diagnostics = prepare_candidate(g, dead_end, cycle_policy="allow")
    assert candidate is None and "no_path_to_terminal" in {d.code for d in errors(diagnostics)}


def test_cycle_policy_repair_removes_back_edges_as_warnings():
    g = two_hop()  # read -> search closes a cycle
    candidate, diagnostics = prepare_candidate(g, EditSet(), cycle_policy="repair")
    assert candidate is not None
    assert [d.code for d in warnings(diagnostics) if d.code == "cycle_repaired"] == ["cycle_repaired"]
    assert candidate.edges_between("read", "search") == []
    assert candidate.meta["repairs"][0]["source"] == "read"
    allowed, diagnostics = prepare_candidate(g, EditSet(), cycle_policy="allow")
    assert allowed is not None and len(allowed.edges_between("read", "search")) == 1
    assert not any(d.code == "cycle_repaired" for d in diagnostics)
    repaired, removed = repair_cycles(g)
    assert [e.key for e in removed] == [("read", "PROVIDES_INPUT_FOR", "search")] and repaired.find_cycles() == []


def test_available_tools_only_warns_and_delete_node_removes_incident_edges():
    g = two_hop()
    candidate, diagnostics = prepare_candidate(g, EditSet(), cycle_policy="allow", available_tools=["search", "read"])
    assert candidate is not None
    assert [d.subject for d in diagnostics if d.code == "unknown_action"] == ["answer"]
    dropped, diagnostics = prepare_candidate(g, EditSet(delete_nodes=["read"], add_edges=[Edge("search", "LEADS_TO", "answer", {"guidance": "skip"})]))
    assert dropped is not None and "read" not in dropped.nodes and not any("read" in (e.source, e.target) for e in dropped.edges)
    assert {d.code for d in warnings(diagnostics)} <= {"unreachable_from_start", "missing_guidance", "cycle_repaired"}


def test_unified_diff_mentions_changed_node():
    g = two_hop()
    after, _ = prepare_candidate(g, EditSet(add_nodes=[Node("verify", "REASONING", "Check")], add_edges=[Edge("read", "LEADS_TO", "verify", {"guidance": "check"}), Edge("verify", "LEADS_TO", END, {"guidance": "done"})]))
    diff = unified_diff(g, after)
    assert "+- `verify` (REASONING): Check" in diff and diff.startswith("--- graph ")
    assert unified_diff(g, g) == ""
    assert START in g.nodes
