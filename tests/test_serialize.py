# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""C5 / H6: the App. B.5 serializer layout, the full-graph rendering, markdown and Mermaid. Acceptance §8.9."""

from proceduralgraph.graph import END, START, Edge, Graph, Node
from proceduralgraph.serialize import render_markdown, render_mermaid, render_trajectory, serialize_context
from proceduralgraph.traces import Step


def hotpot() -> Graph:
    """The App. B.5 excerpt's graph: First_Hop_Retrieve -> Scan_Index -> Bridge_Extract."""
    nodes = {
        START: Node(START, "STATUS", "start"),
        "First_Hop_Retrieve": Node("First_Hop_Retrieve", "ACTION", "Execute first_hop_retrieve to fetch primary evidence passages."),
        "Scan_Index": Node("Scan_Index", "ACTION", "Scan the retrieved passages."),
        "Bridge_Extract": Node("Bridge_Extract", "REASONING", "Extract the bridge entity."),
        END: Node(END, "STATUS", "end"),
    }
    edges = (
        Edge(START, "LEADS_TO", "First_Hop_Retrieve", {"condition": None, "guidance": "Retrieve first.", "pitfalls": None}),
        Edge("First_Hop_Retrieve", "LEADS_TO", "Scan_Index", {
            "condition": "first_hop_retrieve",
            "guidance": "Review the retrieved primary passages via Scan_Index to locate specific bridge terms (such as birth dates, locations, or associated entities).",
            "pitfalls": "Do not skip reading evidence details; missing the exact bridge entity name causes second-hop search failure.",
        }),
        Edge("Scan_Index", "PROVIDES_INPUT_FOR", "Bridge_Extract", {
            "condition": "scan_index",
            "guidance": "Extract the explicit connecting entity or bridge term linking the first passage to the target question.",
            "pitfalls": "Ensure the extracted bridge term matches exact Wikipedia capitalization conventions.",
        }),
        Edge("Bridge_Extract", "LEADS_TO", END, {"condition": None, "guidance": "Finish.", "pitfalls": None}),
    )
    return Graph(nodes=nodes, edges=edges)


EXPECTED_PAPER_LAYOUT = """Active Cognitive Node: [First_Hop_Retrieve] (Type: ACTION)
Description: Execute first_hop_retrieve to fetch primary evidence passages.
Immediate Transition Options (Hop 1):
- Transition: [First_Hop_Retrieve] → [Scan_Index] (Condition: first_hop_retrieve)
* Guidance: Review the retrieved primary passages via Scan_Index to locate specific bridge terms (such as birth dates, locations, or associated entities).
* Pitfalls to Avoid: Do not skip reading evidence details; missing the exact bridge entity name causes second-hop search failure.
Subsequent Horizon (Hop 2):
- Transition: [Scan_Index] → [Bridge_Extract] (Condition: scan_index)
* Guidance: Extract the explicit connecting entity or bridge term linking the first passage to the target question.
* Pitfalls to Avoid: Ensure the extracted bridge term matches exact Wikipedia capitalization conventions."""


def test_paper_layout_without_relations_matches_app_b5():
    """§8.9: headers, Hop 1, Hop 2, Guidance, Pitfalls to Avoid, with include_relations=False."""
    text = serialize_context(hotpot(), "First_Hop_Retrieve", hops=2, include_relations=False)
    assert text == EXPECTED_PAPER_LAYOUT


def test_relation_labels_are_printed_by_default_and_null_condition_is_unconditional():
    text = serialize_context(hotpot(), START, hops=1)
    assert "- Transition: [Start] → [First_Hop_Retrieve] (LEADS_TO; Condition: unconditional)" in text
    assert "Subsequent Horizon" not in text
    terminal = serialize_context(hotpot(), END, hops=2)
    assert "terminal node" in terminal


def test_full_graph_rendering_lists_every_node_and_transition():
    text = serialize_context(hotpot(), None)
    assert text.startswith("Complete Procedural Graph: 5 nodes, 4 transitions")
    assert "- [Scan_Index] (Type: ACTION): Scan the retrieved passages." in text
    assert text.count("- Transition:") == 4


def test_markdown_and_mermaid():
    md = render_markdown(hotpot())
    assert md.startswith("# Procedural Graph\n\n5 node(s), 4 edge(s).")
    assert md.count("Active Cognitive Node:") == 5
    mmd = render_mermaid(hotpot())
    assert mmd.startswith("flowchart LR\n")
    assert '-->|PROVIDES_INPUT_FOR|' in mmd and '(["Start"])' in mmd


def test_render_trajectory_window():
    steps = [Step("search", {"q": "a"}, "r1"), Step("read", "doc1", "text"), Step("search", {"q": "b"}, "r2")]
    text = render_trajectory(steps, window=2)
    assert text.startswith("[... 1 earlier step(s) omitted ...]\nStep 2:\nAction: read(doc1)\nObservation: text")
    assert render_trajectory([]) == "(no steps yet; the agent is at Start)"
