# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""H7: seeding one workspace's graph from another's. Rejection memory never travels."""

from proceduralgraph import (
    EvolveConfig,
    MemoryRevisionStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    evolve,
    seed_workspace,
)
from proceduralgraph.stores.revision import GRAPH

from .test_harness_end_to_end import BUILD, ScoringEvaluator, traces


async def test_transfer_copies_head_into_empty_target_only():
    revisions = MemoryRevisionStore()
    assert await seed_workspace(revisions, source="a", target="b") is None  # nothing to give
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([BUILD]),
                          graph_store=RevisionGraphStore(revisions, "a"), rejection_store=RevisionRejectionStore(revisions, "a"),
                          trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator())
    assert report.outcomes() == ["accepted"]
    digest = await seed_workspace(revisions, source="a", target="b")
    assert digest == report.graph.digest
    head = await revisions.head("b", GRAPH)
    assert head.seq == 0 and head.meta["origin"] == "transfer:a" and head.meta["source_ref"] == report.graph_ref
    ref, graph = await RevisionGraphStore(revisions, "b").load()
    assert ref == digest and "search" in graph.nodes
    _, memory = await RevisionRejectionStore(revisions, "b").load()
    assert memory.entries == []  # rejection memory is not transferred
    assert await seed_workspace(revisions, source="a", target="b") is None  # target already has a graph
    # the transferred workspace evolves in static mode from its inherited graph
    model = ScriptedChatModel(['{"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": []}'])
    report = await evolve(config=EvolveConfig(max_rounds=1), model=model, graph_store=RevisionGraphStore(revisions, "b"),
                          rejection_store=RevisionRejectionStore(revisions, "b"), trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator())
    assert report.outcomes() == ["no_action"] and "Refinement mode: static_incremental" in model.calls[0].text
