# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""G: Algorithm 1 end to end with a scripted refiner. Acceptance §8.4 (tie), §8.5 (structural failure never evaluated),
§8.12 (host-canonical candidate), §8.13 (stale traces) and §8.13a (initialization)."""

import json

import pytest

from proceduralgraph import (
    END,
    START,
    Edge,
    Evaluation,
    EvolveConfig,
    Graph,
    GraphError,
    HeadMoved,
    MemoryRevisionStore,
    Node,
    RevisionGraphStore,
    RevisionRejectionStore,
    RevisionTraceStore,
    ScriptedChatModel,
    StaticTraceSource,
    StrictImprovementGate,
    TaskOutcome,
    Trace,
    evolve,
)
from proceduralgraph.stores.base import Candidate
from proceduralgraph.stores.revision import GRAPH, REJECTIONS


def edits(**arrays) -> str:
    return json.dumps({"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": [], **arrays})


def edge(source, target, guidance="do it", relation="LEADS_TO"):
    return {"source": source, "target": target, "relation": relation, "condition": None, "guidance": guidance, "pitfalls": None}


BUILD = edits(
    add_nodes=[{"id": "search", "type": "ACTION", "description": "s"}, {"id": "read", "type": "ACTION", "description": "r"}, {"id": "answer", "type": "ACTION", "description": "a"}],
    delete_edges=[{"source": START, "target": END}],
    add_edges=[edge(START, "search"), edge("search", "read"), edge("read", "search", relation="PROVIDES_INPUT_FOR"), edge("read", "answer"), edge("answer", END)],
)
BROKEN = edits(add_edges=[edge("answer", "ghost")])
GUESS = edits(add_nodes=[{"id": "guess", "type": "REASONING", "description": "g"}], add_edges=[edge("search", "guess"), edge("guess", END)])
EMPTY = edits()
VERIFY = edits(add_nodes=[{"id": "verify", "type": "REASONING", "description": "v"}], add_edges=[edge("read", "verify"), edge("verify", END)])


class ScoringEvaluator:
    """Deterministic validation: 0.5 for the skeleton, 0.8 once search→read exists, 0.3 when 'guess' is present."""

    def __init__(self):
        self.calls: list[tuple[str, int, str]] = []

    async def evaluate(self, ref, graph, *, iteration, purpose):
        self.calls.append((purpose, iteration, graph.digest))
        if "guess" in graph.nodes:
            score = 0.3
        elif graph.edges_between("search", "read"):
            score = 0.8
        else:
            score = 0.5
        return Evaluation(ref=ref, score=score, per_task=[TaskOutcome(f"t{i}", score, score >= 0.5) for i in range(3)])


def traces(graph_ref=None):
    return [
        Trace("pass-1", TaskOutcome("q1", 1.0, True), "searched, read, answered", graph_ref=graph_ref, nodes_visited=["search", "read", "answer"]),
        Trace("fail-1", TaskOutcome("q2", 0.0, False), "searched then guessed", graph_ref=graph_ref, nodes_visited=["search", "answer"]),
    ]


def stores(revisions=None, workspace="ws"):
    revisions = revisions or MemoryRevisionStore()
    return revisions, RevisionGraphStore(revisions, workspace), RevisionRejectionStore(revisions, workspace), RevisionTraceStore(revisions, workspace)


async def test_full_search_trace_with_tie_acceptance():
    revisions, graph_store, rejection_store, trace_store = stores()
    model = ScriptedChatModel([BUILD, BROKEN, GUESS, GUESS, EMPTY, VERIFY])
    evaluator = ScoringEvaluator()
    report = await evolve(
        config=EvolveConfig(max_rounds=6, role_retries=0),
        model=model,
        graph_store=graph_store,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(traces()),
        evaluator=evaluator,
        trace_store=trace_store,
    )
    assert report.outcomes() == ["accepted", "structural_failure", "rejected", "duplicate_candidate", "no_action", "accepted"]
    assert report.stopped_reason == "max_rounds" and report.best.score == 0.8
    # §8.5: the structurally invalid candidate never reached the evaluator; head and cached score unchanged
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate", "candidate", "candidate"]
    assert [c[1] for c in evaluator.calls] == [0, 1, 3, 6]
    failure = report.iterations[1]
    assert failure.candidate_digest is None and "missing_endpoint" in {d["code"] for d in failure.diagnostics}
    assert failure.graph_ref == report.iterations[0].graph_ref
    # F3: the duplicate points at the earlier rejection and skipped validation
    dup = report.rejections.entries[3]
    assert dup.kind == "duplicate_candidate" and dup.duplicate_of == 3
    # §8.4: a tie (0.8 == 0.8) is accepted by the default gate
    assert report.iterations[5].evaluation["score"] == 0.8 and "verify" in report.graph.nodes
    # storage: one graph revision per acceptance plus the seed; one rejection revision per round
    assert [r.seq for r in await revisions.list("ws", GRAPH)] == [2, 1, 0]
    head = await revisions.head("ws", GRAPH)
    assert head.meta["validation_score"] == 0.8 and head.meta["iteration"] == 6 and head.meta["origin"] == "accepted" and head.meta["edits"]["add_nodes"][0]["id"] == "verify"
    seed = (await revisions.list("ws", GRAPH))[-1]
    assert seed.meta["origin"] == "skeleton"
    assert len(await revisions.list("ws", REJECTIONS)) == 6
    assert report.rejections.counts() == {"accepted": 2, "rejected": 1, "structural_failure": 1, "no_action": 1, "duplicate_candidate": 1}
    # the refiner saw scratch mode first (skeleton), static afterwards, and the rejection block grew
    assert "Refinement mode: scratch_incremental" in model.calls[0].text
    assert "Refinement mode: static_incremental" in model.calls[1].text
    assert "REJECTED" in model.calls[3].text and "STRUCTURAL_FAILURE" in model.calls[3].text
    assert "=== Trace fail-1" in model.calls[0].text and "PASSED" in model.calls[0].text
    assert await trace_store.get("fail-1") is not None
    assert report.to_dict()["outcomes"][0] == "accepted"


async def test_strict_gate_rejects_the_tie_and_streak_stops_the_run():
    _, graph_store, rejection_store, _ = stores()
    model = ScriptedChatModel([BUILD, VERIFY, EMPTY, EMPTY])
    report = await evolve(
        config=EvolveConfig(max_rounds=10, max_rejected_streak=2, role_retries=0),
        model=model,
        graph_store=graph_store,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(traces()),
        evaluator=ScoringEvaluator(),
        gate=StrictImprovementGate(perfect=None),
    )
    assert report.outcomes() == ["accepted", "rejected", "no_action"] and report.stopped_reason == "rejected_streak"
    assert "verify" not in report.graph.nodes


async def test_perfect_stop_and_gate_stop():
    class Perfect(ScoringEvaluator):
        async def evaluate(self, ref, graph, *, iteration, purpose):
            e = await super().evaluate(ref, graph, iteration=iteration, purpose=purpose)
            e.score = 1.0 if "search" in graph.nodes else 0.5
            return e

    _, graph_store, rejection_store, _ = stores()
    report = await evolve(
        config=EvolveConfig(max_rounds=5, role_retries=0),
        model=ScriptedChatModel([BUILD, VERIFY]),
        graph_store=graph_store,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(traces()),
        evaluator=Perfect(),
        gate=StrictImprovementGate(perfect=1.0),
    )
    assert report.outcomes() == ["accepted"] and report.stopped_reason == "perfect"
    # a head that already scores ``perfect`` stops before round one (probe on the baseline); the default gate never does
    _, graph_store, rejection_store, _ = stores()
    model = ScriptedChatModel([BUILD])
    report = await evolve(config=EvolveConfig(max_rounds=3, role_retries=0), model=model, graph_store=graph_store, rejection_store=rejection_store,
                          trace_source=StaticTraceSource(traces()), evaluator=Perfect(), gate=StrictImprovementGate(perfect=1.0),
                          baseline=Evaluation(ref=None, score=1.0))
    assert report.iterations == [] and report.stopped_reason == "perfect" and model.calls == []
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=graph_store, rejection_store=rejection_store,
                          trace_source=StaticTraceSource(traces()), evaluator=Perfect(), baseline=Evaluation(ref=None, score=1.0))
    assert report.outcomes() == ["accepted"] and report.stopped_reason == "max_rounds"


class Normalising:
    """A GraphStore whose propose() returns a graph that differs from the module's: it adds a host annotation node."""

    def __init__(self, store):
        self.store = store
        self.proposals: list[int] = []

    async def load(self):
        return await self.store.load()

    async def seed(self, graph, *, meta):
        return await self.store.seed(graph, meta=meta)

    async def propose(self, current_ref, current, edits, candidate, *, iteration, rejections_ref):
        self.proposals.append(iteration)
        extra = candidate.with_nodes({**candidate.nodes, "host_note": Node("host_note", "STATUS", "added by the host")})
        if not extra.edges_between("host_note", END):
            extra = extra.with_edges(list(extra.edges) + [Edge("host_note", "LEADS_TO", END, {"guidance": "x"})])
        return Candidate(ref=f"host-ref-{iteration}", graph=extra, edits=edits, meta={"iteration": iteration})

    async def accept(self, candidate, *, expected_ref):
        return await self.store.accept(candidate, expected_ref=expected_ref)


async def test_host_candidate_is_canonical_with_one_warning():
    """§8.12: what GraphStore.propose returns is what gets evaluated."""
    revisions, inner, rejection_store, _ = stores()
    host = Normalising(inner)
    evaluator = ScoringEvaluator()
    report = await evolve(
        config=EvolveConfig(max_rounds=1, role_retries=0),
        model=ScriptedChatModel([BUILD]),
        graph_store=host,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(traces()),
        evaluator=evaluator,
    )
    assert report.outcomes() == ["accepted"] and "host_note" in report.graph.nodes
    evaluated_digest = evaluator.calls[1][2]
    assert evaluated_digest == report.graph.digest
    host_warnings = [w for w in report.iterations[0].warnings if "host materialised" in w]
    assert len(host_warnings) == 1
    assert report.rejections.entries[0].candidate_digest == report.graph.digest  # the host's digest is what memory records


async def test_duplicate_refusal_uses_the_host_digest():
    """F3 with a normalising host: the digest compared is the host's, so the repeat is refused without validation."""
    revisions, inner, rejection_store, _ = stores()
    host = Normalising(inner)
    evaluator = ScoringEvaluator()
    report = await evolve(
        config=EvolveConfig(max_rounds=3, role_retries=0),
        model=ScriptedChatModel([BUILD, GUESS, GUESS]),
        graph_store=host,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(traces()),
        evaluator=evaluator,
    )
    assert report.outcomes() == ["accepted", "rejected", "duplicate_candidate"]
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate", "candidate"]
    assert report.rejections.entries[2].duplicate_of == 2


async def test_stale_trace_policy_drop_excludes_and_reports():
    """§8.13."""
    _, graph_store, rejection_store, _ = stores()
    model = ScriptedChatModel([BUILD])
    mixed = traces() + [Trace("old-1", TaskOutcome("q9", 0.0, False), "ran under an older graph", graph_ref="0" * 64)]
    report = await evolve(
        config=EvolveConfig(max_rounds=1, role_retries=0, stale_trace_policy="drop"),
        model=model,
        graph_store=graph_store,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(mixed),
        evaluator=ScoringEvaluator(),
    )
    it = report.iterations[0]
    assert it.stale_trace_ids == ["old-1"] and "old-1" not in it.trace_ids
    assert "old-1" not in model.calls[0].text and any("stale" in w for w in it.warnings)
    # warn (default) keeps the trace but reports it
    _, graph_store, rejection_store, _ = stores()
    model = ScriptedChatModel([BUILD])
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=graph_store,
                          rejection_store=rejection_store, trace_source=StaticTraceSource(mixed), evaluator=ScoringEvaluator())
    it = report.iterations[0]
    assert it.stale_trace_ids == ["old-1"] and "old-1" in it.trace_ids and "old-1" in model.calls[0].text


async def test_initialization_paths(tmp_path):
    """§8.13a: skeleton by default, hand-written graph completed with a warning, invalid refused, seed refuses a populated chain."""
    # empty workspace, nothing supplied -> skeleton, scratch mode
    _, graph_store, rejection_store, _ = stores()
    model = ScriptedChatModel([EMPTY])
    report = await evolve(config=EvolveConfig(max_rounds=1), model=model, graph_store=graph_store, rejection_store=rejection_store,
                          trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator())
    assert report.graph.is_skeleton and "Refinement mode: scratch_incremental" in model.calls[0].text
    # hand-written JSON missing End -> completed, warned, static mode
    revisions, graph_store, rejection_store, _ = stores()
    human = {"nodes": [{"id": "search", "type": "ACTION", "description": "s"}], "edges": [{"source": START, "target": "search", "relation": "LEADS_TO", "guidance": "go"}]}
    path = tmp_path / "expert.json"
    path.write_text(json.dumps(human))
    model = ScriptedChatModel([EMPTY])
    report = await evolve(config=EvolveConfig(max_rounds=1), model=model, graph_store=graph_store, rejection_store=rejection_store,
                          trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator(), initial_graph=path)
    assert END in report.graph.nodes and any("skeleton_completed" in w for w in report.warnings)
    assert "Refinement mode: static_incremental" in model.calls[0].text
    assert (await revisions.head("ws", GRAPH)).meta["origin"] == "onboarded"
    # a graph that fails validate() is refused before any model call
    _, graph_store, rejection_store, _ = stores()
    model = ScriptedChatModel([EMPTY])
    with pytest.raises(GraphError):
        await evolve(config=EvolveConfig(max_rounds=1), model=model, graph_store=graph_store, rejection_store=rejection_store,
                     trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator(),
                     initial_graph={"nodes": [{"id": "x", "type": "ACTION"}], "edges": [{"source": "x", "target": "nowhere", "relation": "LEADS_TO"}]})
    assert model.calls == []
    # seed on a non-empty chain raises HeadMoved; initial_graph on a populated workspace is an error
    _, graph_store, rejection_store, _ = stores()
    await graph_store.seed(Graph.skeleton(), meta={"origin": "skeleton"})
    with pytest.raises(HeadMoved):
        await graph_store.seed(Graph.skeleton(), meta={"origin": "skeleton"})
    with pytest.raises(ValueError, match="already has a graph"):
        await evolve(config=EvolveConfig(max_rounds=1), model=ScriptedChatModel([EMPTY]), graph_store=graph_store, rejection_store=rejection_store,
                     trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator(), initial_graph=Graph.skeleton())


async def test_baseline_supplied_and_available_tools_warning():
    _, graph_store, rejection_store, _ = stores()
    evaluator = ScoringEvaluator()
    report = await evolve(
        config=EvolveConfig(max_rounds=1, role_retries=0),
        model=ScriptedChatModel([BUILD]),
        graph_store=graph_store,
        rejection_store=rejection_store,
        trace_source=StaticTraceSource(traces()),
        evaluator=evaluator,
        baseline=Evaluation(ref=None, score=0.5),
        available_tools=["search", "read"],
    )
    assert [c[0] for c in evaluator.calls] == ["candidate"]
    assert "unknown_action" in {d["code"] for d in report.iterations[0].diagnostics}
    assert report.outcomes() == ["accepted"]
