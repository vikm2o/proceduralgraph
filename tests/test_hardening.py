# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Second-review fixes: storage safety (file lock, name encoding, committed-chain reads), shared budgets, host-candidate
validation, immutability, crash-recovery ordering, baseline budget handling, refiner schema, empty vocabularies, stale
provenance and the zero window."""

import asyncio
import json

import pytest

from proceduralgraph import (
    Budget,
    BudgetExceeded,
    EditError,
    EditSet,
    EvolveConfig,
    FileRevisionStore,
    Graph,
    GuidanceConfig,
    Guide,
    HeadMoved,
    HookedModel,
    Hooks,
    MemoryRevisionStore,
    Node,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    Step,
    StrictImprovementGate,
    TaskOutcome,
    Trace,
    evolve,
    render_attempts_block,
)
from proceduralgraph.graph import END, START, Edge
from proceduralgraph.serialize import render_trajectory
from proceduralgraph.stores.base import Candidate
from proceduralgraph.stores.blob import BlobRevisionStore, MemoryBlobStore
from proceduralgraph.stores.file import _segment
from proceduralgraph.stores.revision import GRAPH, REJECTIONS

from .test_guidance import two_hop
from .test_harness_end_to_end import BUILD, ScoringEvaluator, traces


# -- 1: concurrent file writers cannot fork --------------------------------------------------------------------------------
async def test_file_store_concurrent_instances_cannot_both_write_revision_zero(tmp_path):
    a, b = FileRevisionStore(tmp_path), FileRevisionStore(tmp_path)
    results = await asyncio.gather(
        *[s.append("ws", "graph", {"n": i}, expected_head=None) for i, s in enumerate([a, b] * 4)], return_exceptions=True
    )
    ok = [r for r in results if not isinstance(r, Exception)]
    assert len(ok) == 1 and all(isinstance(r, HeadMoved) for r in results if r is not ok[0])
    assert (await a.head("ws", "graph")).digest == ok[0].digest
    assert [r.seq for r in await b.list("ws", "graph")] == [0]


# -- 2: workspace names are encoded reversibly and cannot escape the root ---------------------------------------------------
async def test_file_store_segments_are_injective_and_stay_under_root(tmp_path):
    assert _segment("team/a") != _segment("team_a") and _segment("..") == "%2E%2E" and _segment("plain-name_1") == "plain-name_1"
    store = FileRevisionStore(tmp_path / "root")
    await store.append("team/a", "graph", {"n": "slash"}, expected_head=None)
    await store.append("team_a", "graph", {"n": "underscore"}, expected_head=None)
    await store.append("..", "graph", {"n": "dots"}, expected_head=None)
    assert (await store.head("team/a", "graph")).document == {"n": "slash"}
    assert (await store.head("team_a", "graph")).document == {"n": "underscore"}
    assert (await store.head("..", "graph")).document == {"n": "dots"}
    written = {p.relative_to(tmp_path).parts[0] for p in (tmp_path).rglob("HEAD")}
    assert written == {"root"}  # nothing landed outside the configured root


# -- 3: blob reads follow the committed chain -----------------------------------------------------------------------------------
async def test_blob_list_and_get_ignore_a_losing_writers_orphan():
    blobs = MemoryBlobStore()
    store = BlobRevisionStore(blobs, prefix="p")
    first = await store.append("ws", "checkpoints", {"n": 0}, expected_head=None)
    winner = BlobRevisionStore(blobs, prefix="p")
    original = blobs.put_if_match

    async def race(key, data, *, tag):  # another writer advances HEAD between our read and our write
        blobs.put_if_match = original
        await winner.append("ws", "checkpoints", {"winner": True}, expected_head=first.digest)
        return await original(key, data, tag=tag)

    blobs.put_if_match = race
    with pytest.raises(HeadMoved):
        await store.append("ws", "checkpoints", {"loser": True}, expected_head=first.digest)
    assert len(await blobs.list("p/ws/checkpoints/revisions/")) == 3  # the orphan object exists ...
    assert [r.document for r in await store.list("ws", "checkpoints")] == [{"winner": True}, {"n": 0}]  # ... but is never listed
    head_digest = (await store.head("ws", "checkpoints")).digest
    keys = await blobs.list("p/ws/checkpoints/revisions/")
    loser_digest = next(k for k in keys if "000001" in k and head_digest not in k).rsplit("-", 1)[1][:-5]
    assert await store.get("ws", "checkpoints", loser_digest) is None and await store.get("ws", "checkpoints", head_digest) is not None


async def test_file_list_follows_the_chain_not_the_directory(tmp_path):
    store = FileRevisionStore(tmp_path)
    first = await store.append("ws", "graph", {"n": 0}, expected_head=None)
    second = await store.append("ws", "graph", {"n": 1}, expected_head=first.digest)
    orphan = tmp_path / "ws" / "graph" / f"000001-{'f' * 64}.json"
    orphan.write_text(json.dumps({**second.to_document(), "digest": "f" * 64, "document": {"orphan": True}}))
    assert [r.document for r in await store.list("ws", "graph")] == [{"n": 1}, {"n": 0}]
    assert await store.get("ws", "graph", "f" * 64) is None


# -- 4: a shared HookedModel means guidance calls count against the run budget --------------------------------------------------
async def test_guidance_calls_through_a_shared_hooked_model_consume_the_budget():
    revisions = MemoryRevisionStore()
    hooked = HookedModel(ScriptedChatModel(lambda r: "guidance" if r.role == "guidance" else BUILD), Hooks())

    class GuidedEvaluator(ScoringEvaluator):
        async def evaluate(self, ref, graph, *, iteration, purpose):
            guide = Guide(graph, hooked, GuidanceConfig(mode="generative_subgraph"))
            for _ in range(3):
                await guide.guidance("q", [Step("search")])
            return await super().evaluate(graph_ref_or(ref), graph, iteration=iteration, purpose=purpose)

    def graph_ref_or(ref):
        return ref

    report = await evolve(
        config=EvolveConfig(max_rounds=3, role_retries=0, budget=Budget(max_model_calls=5)),
        model=hooked,
        graph_store=RevisionGraphStore(revisions, "ws"),
        rejection_store=RevisionRejectionStore(revisions, "ws"),
        trace_source=StaticTraceSource(traces()),
        evaluator=GuidedEvaluator(),
    )
    # baseline: 3 guidance calls; round 1: refiner (4th), then validation starts and its 2nd guidance call is the 6th -> refused
    assert report.stopped_reason == "budget_exhausted" and report.model_calls <= 5 and report.budget["model_calls"] == report.model_calls
    assert hooked.meter.budget.max_model_calls == 5  # the loop adopted config.budget on the host's meter


# -- 5: the host's materialised candidate is validated -------------------------------------------------------------------------
async def test_host_candidate_failing_structural_checks_is_a_structural_failure():
    revisions = MemoryRevisionStore()
    inner = RevisionGraphStore(revisions, "ws")

    class Breaking:
        async def load(self):
            return await inner.load()

        async def seed(self, graph, *, meta):
            return await inner.seed(graph, meta=meta)

        async def propose(self, current_ref, current, edits, candidate, *, iteration, rejections_ref):
            broken = candidate.with_edges(list(candidate.edges) + [Edge("read", "LEADS_TO", "ghost", {"guidance": "x"})])
            return Candidate(ref="host", graph=broken, edits=edits, meta={})

        async def accept(self, candidate, *, expected_ref):
            return await inner.accept(candidate, expected_ref=expected_ref)

    evaluator = ScoringEvaluator()
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([BUILD]), graph_store=Breaking(),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=evaluator)
    assert report.outcomes() == ["structural_failure"]
    assert [c[0] for c in evaluator.calls] == ["baseline"]  # never evaluated
    assert "missing_endpoint" in {d["code"] for d in report.iterations[0].diagnostics}
    assert (await revisions.head("ws", GRAPH)).seq == 0 and report.graph.is_skeleton


# -- 6: graphs, nodes and edges are immutable ------------------------------------------------------------------------------------
def test_edges_nodes_and_graphs_are_read_only():
    g = two_hop()
    candidate = g.with_meta(x=1)
    before = g.digest
    with pytest.raises(TypeError):
        candidate.edges[0].attributes["guidance"] = "MUTATED"  # type: ignore[index]
    with pytest.raises(TypeError):
        candidate.nodes["search"].meta["k"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        candidate.nodes["zzz"] = Node("zzz", "ACTION")  # type: ignore[index]
    with pytest.raises(TypeError):
        candidate.meta["y"] = 2  # type: ignore[index]
    assert g.digest == before
    guide = Guide(g, None, GuidanceConfig(mode="raw_subgraph"))
    assert "read the best hit" in guide.guidance_sync("q", [Step("search")]).text
    assert Graph.from_document(g.to_document()).digest == before  # documents are still plain JSON


# -- 7: a pending checkpoint is reconciled before the perfect stop -----------------------------------------------------------
async def test_pending_checkpoint_is_recorded_even_when_the_head_is_already_perfect():
    class Perfect(ScoringEvaluator):
        async def evaluate(self, ref, graph, *, iteration, purpose):
            e = await super().evaluate(ref, graph, iteration=iteration, purpose=purpose)
            e.score = 1.0 if "search" in graph.nodes else 0.5
            return e

    revisions = MemoryRevisionStore()
    checkpoints = RevisionCheckpointStore(revisions, "ws")

    class DiesAfterAccept(RevisionRejectionStore):
        armed = True

        async def save(self, memory, *, expected_ref, iteration):
            if DiesAfterAccept.armed:
                DiesAfterAccept.armed = False
                raise RuntimeError("lost the lease")
            return await super().save(memory, expected_ref=expected_ref, iteration=iteration)

    common = dict(graph_store=RevisionGraphStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=Perfect(),
                  checkpoint_store=checkpoints, gate=StrictImprovementGate(perfect=1.0))
    with pytest.raises(RuntimeError):
        await evolve(config=EvolveConfig(max_rounds=2, role_retries=0), model=ScriptedChatModel([BUILD]), rejection_store=DiesAfterAccept(revisions, "ws"), **common)
    assert (await revisions.head("ws", GRAPH)).seq == 1 and await revisions.head("ws", REJECTIONS) is None
    report = await evolve(config=EvolveConfig(max_rounds=2, role_retries=0), model=ScriptedChatModel([]), rejection_store=RevisionRejectionStore(revisions, "ws"), **common)
    assert report.outcomes() == ["accepted"] and report.stopped_reason == "perfect"
    assert (await revisions.head("ws", REJECTIONS)) is not None and report.rejections.entries[0].iteration == 1


# -- 8: budget exhaustion inside the baseline evaluation is a clean stop --------------------------------------------------------
async def test_budget_exhausted_inside_baseline_returns_budget_exhausted():
    class Spending(ScoringEvaluator):
        async def evaluate(self, ref, graph, *, iteration, purpose):
            raise BudgetExceeded("max_model_calls", 3, 3)

    revisions = MemoryRevisionStore()
    report = await evolve(config=EvolveConfig(max_rounds=2), model=ScriptedChatModel([BUILD]), graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=Spending())
    assert report.stopped_reason == "budget_exhausted" and report.best is None and report.iterations == []


# -- 9: a reply without any of the four arrays is malformed, not no_action -------------------------------------------------------
def test_edit_set_requires_one_of_the_four_arrays():
    with pytest.raises(EditError) as info:
        EditSet.parse('{"nodes": [{"id": "x", "type": "ACTION"}], "edges": []}')
    assert "none of the four arrays" in info.value.diagnostics[0].message
    assert EditSet.parse('{"add_nodes": []}').is_empty  # any one of the four is enough


# -- 10: an explicitly empty vocabulary survives the document round trip ------------------------------------------------------
def test_empty_attribute_schema_round_trips():
    bare = Graph.skeleton(attribute_fields=())
    assert bare.attribute_fields == () and Graph.from_document(bare.to_document()).digest == bare.digest
    assert Graph.from_document({"nodes": [], "edges": []}).attribute_fields == ("condition", "guidance", "pitfalls")  # absent = default
    human, _ = Graph.from_human({"nodes": [], "edges": [], "relations": ["NEXT"], "attribute_fields": []})
    assert human.relations == ("NEXT",) and human.attribute_fields == () and Graph.from_document(human.to_document()).digest == human.digest


# -- 11: stale provenance reaches the refiner -----------------------------------------------------------------------------------
def test_attempts_block_marks_stale_traces():
    head = "a" * 64
    fresh = Trace("fresh", TaskOutcome("q1", 1.0, True), "ok", graph_ref=head)
    stale = Trace("stale", TaskOutcome("q2", 0.0, False), "boom", graph_ref="b" * 64)
    block = render_attempts_block([fresh, stale], cap=10_000, head_ref=head)
    assert "=== Trace fresh | task q1 | score 1.000 | PASSED | graph aaaaaaaaaaaa ===" in block
    assert "| graph bbbbbbbbbbbb STALE (ran under an earlier graph" in block and "1 ran under an earlier graph and are marked STALE" in block
    assert "STALE" not in render_attempts_block([fresh, stale], cap=10_000)  # no head given: no judgement


async def test_stale_marker_in_the_refiner_prompt_under_warn_policy():
    revisions = MemoryRevisionStore()
    model = ScriptedChatModel([BUILD])
    mixed = traces() + [Trace("old-1", TaskOutcome("q9", 0.0, False), "ran under an older graph", graph_ref="0" * 64)]
    await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=RevisionGraphStore(revisions, "ws"),
                 rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(mixed), evaluator=ScoringEvaluator())
    assert "=== Trace old-1" in model.calls[0].text and "STALE (ran under an earlier graph" in model.calls[0].text


# -- 12: a zero window shows no history -------------------------------------------------------------------------------------------
def test_zero_window_shows_no_steps():
    text = render_trajectory([Step("a"), Step("b")], window=0)
    assert "Action:" not in text and "2 step(s) taken" in text
    assert START in two_hop().nodes and END in two_hop().nodes
