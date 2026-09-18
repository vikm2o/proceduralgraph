# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""F: rejection memory as a persistent artefact, SerializeRejections and the operator rendering."""

from proceduralgraph.edits import EditSet
from proceduralgraph.gates import Evaluation
from proceduralgraph.graph import Diagnostic, Edge, Node
from proceduralgraph.rejections import RejectionEntry, RejectionMemory

from .test_graph import two_hop


def entry(iteration: int, kind: str, **extra) -> RejectionEntry:
    g = two_hop()
    edits = EditSet(add_nodes=[Node(f"n{iteration}", "REASONING", "x")], add_edges=[Edge("read", "LEADS_TO", f"n{iteration}", {"guidance": "g"})])
    return RejectionEntry(
        iteration=iteration,
        kind=kind,
        edits=edits,
        candidate=g if kind in ("accepted", "rejected") else None,
        candidate_digest=f"{iteration:064d}" if kind != "no_action" else None,
        base_digest=g.digest,
        trace_ids=[f"t{iteration}a", f"t{iteration}b"],
        trace_scores={f"t{iteration}a": 1.0, f"t{iteration}b": 0.0},
        **extra,
    )


def test_document_round_trip_and_digest_lookup():
    memory = RejectionMemory()
    memory.record(entry(1, "accepted", validation=Evaluation("r", 0.6), retained_score=0.5))
    memory.record(entry(2, "rejected", validation=Evaluation("r", 0.4), retained_score=0.6))
    memory.record(entry(3, "structural_failure", diagnostics=[Diagnostic("missing_endpoint", "m", "x")]))
    memory.record(entry(4, "no_action"))
    memory.record(entry(5, "duplicate_candidate", duplicate_of=2))
    assert memory.iteration == 5
    again = RejectionMemory.from_document(memory.to_document())
    assert again.digest == memory.digest and len(again.entries) == 5
    assert again.entries[1].candidate is not None and again.entries[2].candidate is None
    assert again.rejected_digests() == {f"{2:064d}": 2, f"{3:064d}": 3}
    assert again.counts()["duplicate_candidate"] == 1
    assert again.entries[1].edits.counts()["add_nodes"] == 1 and again.entries[0].validation.score == 0.6


def test_render_for_refiner_full_then_one_line_then_capped():
    memory = RejectionMemory()
    for k in range(1, 9):
        memory.record(entry(k, "rejected", validation=Evaluation("r", 0.1 * k), retained_score=0.9))
    memory.record(entry(9, "accepted", validation=Evaluation("r", 0.95), retained_score=0.9))
    text = memory.render_for_refiner(full_entries=3, char_cap=100_000)
    assert text.count("### iteration") == 3 and "### iteration 8" in text and "### iteration 5" not in text
    assert "- iteration 1: REJECTED" in text and "- iteration 9: ACCEPTED" in text and "validation 0.9500" in text
    assert "Edits (JSON)" in text and "Do not repeat rejected edits" in text
    capped = memory.render_for_refiner(full_entries=3, char_cap=900)
    assert len(capped) <= 900
    assert "iteration 8" in capped  # the newest survives the trimming
    assert RejectionMemory().render_for_refiner() == "(none yet)"


def test_markdown_and_headlines():
    memory = RejectionMemory()
    memory.record(entry(1, "structural_failure", diagnostics=[Diagnostic("missing_endpoint", "no node ghost", "a→ghost")]))
    memory.record(entry(2, "accepted", validation=Evaluation("r", 0.7), retained_score=0.5))
    md = memory.render_markdown()
    assert md.startswith("# Rejection memory")
    assert "## Iteration 1: structural failure" in md and "- error:missing_endpoint [a→ghost]: no node ghost" in md
    assert "## Iteration 2: accepted" in md and "```json" in md
    assert memory.last_lines(1) == [memory.entries[-1].headline()]
    assert "diagnostics missing_endpoint" in memory.entries[0].headline()
