# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Objective mode through the whole loop (REQ-004, REQ-007, REQ-015, REQ-016, REQ-018): the refiner sees the frozen
objective and every decision, decisions persist in rejection memory / reports / revision meta, deferred evaluations
cannot promote, and nothing about the objective costs a model call. TEST-007, TEST-011, part of TEST-009."""

import json

import pytest

from proceduralgraph import (
    Budget,
    EvolveConfig,
    MemoryRevisionStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    TaskOutcome,
    Trace,
    evolve,
)
from proceduralgraph.gates import Decision, Evaluation, ObjectiveGate
from proceduralgraph.rejections import RejectionMemory
from proceduralgraph.stores.revision import GRAPH, REJECTIONS

from .test_harness_end_to_end import BUILD, EMPTY, GUESS, VERIFY, edge, edits
from .test_objectives import context, objective

IDS = tuple(f"task-{i}" for i in range(2000))  # enough units for the equivalence margins to be reachable
OBJ = objective(mode="lexicographic")
CTX = context(OBJ, ids=IDS)
POBJ = objective(mode="pareto")
PCTX = context(POBJ, ids=IDS)
# adds a node without changing quality or cost: equivalent under the objective
NOTE = edits(add_nodes=[{"id": "note", "type": "REASONING", "description": "n"}], add_edges=[edge("read", "note"), edge("note", "End")])
# BUILD plus an "expensive" verification node: quality up, cost materially up (a trade-off under pareto)
_build = json.loads(BUILD)
BUILD_EXPENSIVE = json.dumps({**_build, "add_nodes": _build["add_nodes"] + [{"id": "expensive", "type": "REASONING", "description": "e"}],
                              "add_edges": _build["add_edges"] + [edge("answer", "expensive"), edge("expensive", "End")]})


class MetricEvaluator:
    """Deterministic paired metrics from the graph's shape. quality: 0.5 skeleton, 0.9 with search→read, 0.3 with 'guess'.
    cost (calls per task): 6 for the floundering skeleton, 3 once search→read guides the agent, 8 when it guesses, +5 for an
    'expensive' node. ``unknown_cost`` blanks the cost on one unit."""

    def __init__(self, ctx, *, unknown_cost=False, wrong_ref=False):
        self.ctx, self.unknown_cost, self.wrong_ref = ctx, unknown_cost, wrong_ref
        self.calls: list[tuple[str, int]] = []

    def observe(self, graph, task_id):
        q = 0.3 if "guess" in graph.nodes else 0.9 if graph.edges_between("search", "read") else 0.5
        c = 8.0 if "guess" in graph.nodes else 3.0 if graph.edges_between("search", "read") else 6.0
        if "expensive" in graph.nodes:
            c = min(c + 5.0, 10.0)
        if self.unknown_cost and task_id == IDS[7]:
            c = None
        return q, c

    async def evaluate(self, ref, graph, *, iteration, purpose):
        self.calls.append((purpose, iteration))
        per_task = []
        for task_id in self.ctx.expected_task_ids:
            q, c = self.observe(graph, task_id)
            per_task.append(TaskOutcome(task_id, q, q >= 0.5, metrics={"quality": q, "cost": c}))
        return Evaluation(ref="wrong" if self.wrong_ref else ref, score=None, per_task=per_task, objective_context=self.ctx)


def traces():
    return [Trace("t-pass", TaskOutcome("q1", 1.0, True, metrics={"quality": 1.0, "cost": 2.0}), "searched, read, answered", nodes_visited=["search", "read"]),
            Trace("t-fail", TaskOutcome("q2", 0.0, False, metrics={"quality": 0.0, "cost": 5.0}), "guessed", nodes_visited=["search"])]


def config(obj=OBJ, ctx=CTX, **over):
    return EvolveConfig(**{"max_rounds": 4, "role_retries": 0, **over}, objective=obj, objective_context=ctx)


async def run(model_replies, evaluator, *, obj=OBJ, ctx=CTX, revisions=None, gate=None, **over):
    revisions = revisions or MemoryRevisionStore()
    model = ScriptedChatModel(model_replies)
    report = await evolve(config=config(obj, ctx, **over), model=model, graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()),
                          evaluator=evaluator, gate=gate or ObjectiveGate(obj, ctx))
    return report, model, revisions


async def test_lexicographic_run_records_dispositions_and_renders_them_to_the_refiner():
    evaluator = MetricEvaluator(CTX)
    report, model, revisions = await run([BUILD, GUESS, NOTE, NOTE], evaluator, max_rounds=4)
    # BUILD improves quality -> accepted; GUESS drops quality -> rejected/metric_regression; NOTE is equivalent twice
    # (an equivalent comparison proved nothing, so the second NOTE is NOT a duplicate refusal: decision 1)
    assert report.outcomes() == ["accepted", "rejected", "rejected", "rejected"]
    assert [r.disposition for r in report.iterations] == ["accepted", "rejected", "equivalent", "equivalent"]
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate", "candidate", "candidate", "candidate"]
    assert report.model_calls == 4  # the gate cost no model call (TEST-011)
    entries = report.rejections.entries
    assert entries[1].decision["reasons"] == ["metric_regression"] and entries[1].decision["decisive_metric"] == "quality"
    assert entries[1].baseline_ref == report.iterations[0].graph_ref and entries[1].candidate_ref
    assert entries[2].disposition == "equivalent" and not entries[2].measured_rejection and entries[1].measured_rejection
    assert report.rejections.rejected_digests() == {entries[1].candidate_digest: 2}
    # the refiner saw the frozen objective, the measured observations and the decisive failure (REQ-015)
    prompt = model.calls[2].text  # round 3: after the rejection
    assert "Objective for this run" in prompt and "1. quality [fraction] maximize" in prompt and "mode lexicographic" in prompt
    assert "Decision: rejected (reasons: metric_regression)" in prompt and "Decisive metric: quality" in prompt
    assert "- quality [fraction] maximize: baseline 0.9 → candidate 0.3" in prompt and "positive = better" in prompt
    assert "measured: quality=1, cost=2" in prompt  # per-trace observations
    assert "iteration 2: REJECTED candidate" in prompt and "| rejected on quality (metric_regression)" in prompt
    # the accepted revision carries the decision and the two refs (REQ-016)
    head = await revisions.head("ws", GRAPH)
    assert head.meta["decision"]["disposition"] == "accepted" and head.meta["baseline_ref"] == head.meta["decision"]["baseline_ref"]
    assert head.meta["candidate_ref"] == head.meta["decision"]["candidate_ref"] and head.meta["validation_score"] is None
    # persistence round trip keeps the decision and refs
    stored = RejectionMemory.from_document((await revisions.head("ws", REJECTIONS)).document)
    assert stored.entries[1].decision == entries[1].decision and stored.entries[1].candidate_ref == entries[1].candidate_ref
    assert stored.to_document() == report.rejections.to_document()
    assert "decision" in report.iterations[1].to_dict() and report.iterations[1].to_dict()["decision"]["disposition"] == "rejected"
    # the operator rendering shows the same structured decision
    md = report.rejections.render_markdown()
    assert "Decision: rejected (reasons: metric_regression)" in md and "Decision: equivalent" in md


async def test_pareto_trade_off_and_unknown_cost():
    # BUILD_EXPENSIVE: quality up, cost materially up -> trade-off, retained incumbent; BUILD: both better -> accepted;
    # GUESS: quality down and cost up -> regression; VERIFY adds a REASONING node: nothing changes -> equivalent
    evaluator = MetricEvaluator(PCTX)
    report, model, _ = await run([BUILD_EXPENSIVE, BUILD, GUESS, VERIFY], evaluator, obj=POBJ, ctx=PCTX, max_rounds=4)
    assert [r.disposition for r in report.iterations] == ["rejected", "accepted", "rejected", "equivalent"]
    assert report.iterations[0].decision["reasons"] == ["trade_off"] and report.iterations[0].decision["metrics"]["quality"]["verdict"] == "improved"
    assert report.iterations[0].decision["metrics"]["cost"]["verdict"] == "regressed" and report.iterations[0].decision["decisive_metric"] == "cost"
    assert report.iterations[2].decision["reasons"] == ["metric_regression"]  # quality worse AND cost worse: no improvement, so not a trade-off
    assert "Decision: rejected (reasons: trade_off)" in model.calls[1].text  # the refiner is told why the trade-off lost
    # unknown cost on one unit: pareto cannot measure; lexicographic accepts on quality and lists the unknown
    unknown = MetricEvaluator(PCTX, unknown_cost=True)
    report, _, _ = await run([BUILD], unknown, obj=POBJ, ctx=PCTX, max_rounds=1)
    assert report.outcomes() == ["rejected"] and report.iterations[0].disposition == "unmeasured"
    assert report.iterations[0].decision["unknown_metrics"] == ["cost"] and report.graph.is_skeleton
    unknown = MetricEvaluator(CTX, unknown_cost=True)
    report, _, _ = await run([BUILD], unknown, max_rounds=1)
    assert report.iterations[0].disposition == "accepted" and report.iterations[0].decision["unknown_metrics"] == ["cost"]


async def test_deferred_evaluation_cannot_be_promoted_and_its_feedback_persists():
    class Deferred:
        def __init__(self):
            self.calls = 0

        async def evaluate(self, ref, graph, *, iteration, purpose):
            self.calls += 1
            return Evaluation(ref=ref, score=None, per_task=[], objective_context=CTX)  # scored later by the host

    report, _, revisions = await run([BUILD], Deferred(), max_rounds=1)
    assert report.outcomes() == ["rejected"] and report.iterations[0].disposition == "unmeasured"
    assert "deferred_evaluation" in report.iterations[0].decision["reasons"] and report.graph.is_skeleton
    assert (await revisions.head("ws", GRAPH)).seq == 0
    stored = RejectionMemory.from_document((await revisions.head("ws", REJECTIONS)).document)
    assert stored.entries[0].disposition == "unmeasured" and stored.entries[0].measured_rejection is False
    assert stored.rejected_digests() == {}  # a deferred result blocks nothing


async def test_custom_gate_feedback_survives_persistence():
    class HostGate:
        async def decide(self, best, candidate):
            if best is candidate:
                return Decision(accepted=False, stop=False)
            return Decision(accepted=False, feedback={"disposition": "unmeasured", "reasons": ["host_qualification_pending"], "ticket": "QA-17"})

    from .test_harness_end_to_end import ScoringEvaluator

    revisions = MemoryRevisionStore()
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([BUILD]), graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator(), gate=HostGate())
    assert report.outcomes() == ["rejected"] and report.iterations[0].decision["ticket"] == "QA-17"
    stored = RejectionMemory.from_document((await revisions.head("ws", REJECTIONS)).document)
    assert stored.entries[0].decision["reasons"] == ["host_qualification_pending"] and not stored.entries[0].measured_rejection
    assert "unmeasured (host_qualification_pending)" in stored.entries[0].headline()


async def test_ref_and_digest_checks():
    # a candidate evaluation whose ref does not identify the evaluated graph is refused (REQ-004)
    with pytest.raises(ValueError, match="REQ-004"):
        await run([BUILD], MetricEvaluator(CTX, wrong_ref=True), max_rounds=1)
    # a host-supplied baseline must identify the head
    revisions = MemoryRevisionStore()
    with pytest.raises(ValueError, match="baseline Evaluation.ref"):
        await evolve(config=config(), model=ScriptedChatModel([BUILD]), graph_store=RevisionGraphStore(revisions, "ws"),
                     rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=MetricEvaluator(CTX),
                     gate=ObjectiveGate(OBJ, CTX), baseline=Evaluation(ref="elsewhere", score=None, objective_context=CTX))
    # the gate and the config must agree
    with pytest.raises(ValueError, match="differ from EvolveConfig"):
        await run([BUILD], MetricEvaluator(CTX), gate=ObjectiveGate(POBJ, PCTX), max_rounds=1)
    with pytest.raises(ValueError, match="set together"):
        EvolveConfig(objective=OBJ)
    with pytest.raises(ValueError, match="does not match"):
        EvolveConfig(objective=POBJ, objective_context=CTX)
    # without an objective, scalar gates and lenient refs behave exactly as before (REQ-018)
    from .test_harness_end_to_end import ScoringEvaluator

    revisions = MemoryRevisionStore()
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([BUILD]), graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator(),
                          baseline=Evaluation(ref=None, score=0.5))
    assert report.outcomes() == ["accepted"] and report.iterations[0].disposition is None  # a scalar gate records its own small feedback
    assert report.iterations[0].decision == {"best_score": 0.5, "candidate_score": 0.8, "accepted": True}
    assert report.rejections.entries[0].measured_rejection is False and report.rejections.entries[0].kind == "accepted"


async def test_budget_still_stops_and_no_action_under_objective():
    evaluator = MetricEvaluator(CTX)
    report, _, _ = await run([BUILD, EMPTY], evaluator, max_rounds=2, budget=Budget(max_evaluations=2))
    assert report.outcomes() == ["accepted", "no_action"] and report.iterations[1].decision is None
    report, _, _ = await run([BUILD, VERIFY], MetricEvaluator(CTX), max_rounds=2, budget=Budget(max_evaluations=2))
    assert report.outcomes() == ["accepted"] and report.stopped_reason == "budget_exhausted"


def test_legacy_documents_keep_their_shape_and_digests():
    """TEST-009: a 0.2.0 run's documents round-trip byte for byte through the 0.3.0 readers and writers."""
    from pathlib import Path

    from proceduralgraph.documents import Revision, digest

    fixture = json.loads(Path(__file__).with_name("fixtures").joinpath("legacy_0_2_0", "run.json").read_text(encoding="utf-8"))
    memory = RejectionMemory.from_document(fixture["rejections"])
    assert memory.to_document() == fixture["rejections"] and digest(memory.to_document()) == digest(fixture["rejections"])
    assert all(e.decision is None and e.baseline_ref is None for e in memory.entries)
    assert memory.rejected_digests() == {memory.entries[2].candidate_digest: 3}
    head = Revision.from_document(fixture["graph_head"])
    head.verify()
    for t in fixture["traces"]["traces"]:
        assert Trace.from_document(t).to_document() == t
    for cp in fixture["checkpoints"]:
        payload = cp.get("payload", {})
        if payload.get("evaluation"):
            assert Evaluation.from_document(payload["evaluation"]).to_document() == payload["evaluation"]
    legacy_outcome = TaskOutcome("t", 1.0, True, "p", "truth", {"k": 1})  # positional 0.2.0 constructor
    assert legacy_outcome.metrics is None and "metrics" not in legacy_outcome.to_dict()
    assert Evaluation(ref="r", score=0.5).to_document() == {"ref": "r", "score": 0.5, "aggregate": {}, "per_task": []}


def test_cli_and_export_show_the_structured_decision(tmp_path, capsys):
    """REQ-016: the CLI rejection view and the exported rejections.md carry the same decision lines."""
    import asyncio

    from proceduralgraph import FileRevisionStore, export_workspace
    from proceduralgraph.cli import main

    revisions = FileRevisionStore(tmp_path / "store")
    evaluator = MetricEvaluator(CTX)

    async def go():
        return await evolve(config=config(max_rounds=2), model=ScriptedChatModel([BUILD, GUESS]), graph_store=RevisionGraphStore(revisions, "ws"),
                            rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=evaluator,
                            gate=ObjectiveGate(OBJ, CTX))

    report = asyncio.run(go())
    assert [i.disposition for i in report.iterations] == ["accepted", "rejected"]
    assert main(["--store", f"file:{tmp_path / 'store'}", "--workspace", "ws", "rejections"]) == 0
    out = capsys.readouterr().out
    assert "Decision: rejected (reasons: metric_regression)" in out and "Decisive metric: quality" in out and "Compared baseline" in out
    assert main(["--store", f"file:{tmp_path / 'store'}", "--workspace", "ws", "show"]) == 0
    assert "| rejected on quality (metric_regression)" in capsys.readouterr().out
    assert main(["--store", f"file:{tmp_path / 'store'}", "--workspace", "ws", "history", "--kind", "graph"]) == 0
    assert '"decision": "accepted"' in capsys.readouterr().out
    written = export_workspace(tmp_path / "out", graph=report.graph, rejections=report.rejections)
    text = (tmp_path / "out" / "rejections.md").read_text(encoding="utf-8")
    assert "Decision: rejected (reasons: metric_regression)" in text and "Decision: accepted" in text and len(written) == 4


async def test_scalar_mode_rendering_is_unchanged_and_custom_gate_is_warned():
    """REQ-018: without an objective the refiner's rejection lines and rejections.md read exactly as in 0.2.0."""
    from .test_harness_end_to_end import GUESS as G
    from .test_harness_end_to_end import ScoringEvaluator

    revisions = MemoryRevisionStore()
    model = ScriptedChatModel([BUILD, G])
    report = await evolve(config=EvolveConfig(max_rounds=2, role_retries=0), model=model, graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator())
    rejected = report.rejections.entries[1]
    assert rejected.headline().startswith("iteration 2: REJECTED candidate") and "| validation 0.3000 vs retained 0.8000 |" in rejected.headline()
    assert "None" not in rejected.headline() and "Validation outcome: score 0.3 against retained 0.8" in rejected.render_full()
    assert "Decision:" not in report.rejections.render_markdown() and "Decision:" not in report.rejections.render_for_refiner()
    assert report.rejections.entries[0].headline().endswith("| validation 0.8000 vs retained 0.5000 | edits +3 node(s) -0 node(s) +5 edge(s) -1 edge(s)")
    # an objective configured with a non-objective gate is allowed but warned
    class Lenient:
        async def decide(self, best, candidate):
            from proceduralgraph.gates import Decision

            return Decision(accepted=best is not candidate, feedback={"disposition": "accepted", "reasons": ["host_rule"]})

    report, _, _ = await run([BUILD], MetricEvaluator(CTX), gate=Lenient(), max_rounds=1)
    assert any("is not an ObjectiveGate" in w for w in report.warnings) and report.outcomes() == ["accepted"]


async def test_objective_header_survives_a_small_rejection_cap():
    """TEST-007 under context caps: the frozen objective heads the rejected-candidates slot even when entries are trimmed."""
    evaluator = MetricEvaluator(CTX)
    report, model, _ = await run([BUILD, GUESS, NOTE], evaluator, max_rounds=3, rejections_char_cap=1_200, rejections_full_entries=1)
    prompt = model.calls[2].text
    start = prompt.index("Previously rejected candidates:")
    block = prompt[start:prompt.index("Your job is to refine")]
    assert "Objective for this run" in block and "1. quality [fraction] maximize" in block and len(block) < 1_200 + 900
