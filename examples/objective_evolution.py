# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Objective mode, offline, with scripted model replies and synthetic paired task outcomes (REQ-020).

    uv run python examples/objective_evolution.py [./objective-workspace]

Shows, with no API key and no network:

1. **lexicographic** (quality first, then cost): an accepted improvement, a measured quality regression, an
   equivalent change that is refused without blocking a later re-proposal, and the refiner's view of every decision;
2. **pareto** (quality and cost both protected): a better-quality-but-costlier candidate refused as a trade-off;
3. **unknown cost** on one unit: pareto cannot measure, lexicographic accepts on quality and lists the unknown;
4. **restart recovery**: a worker dies after the graph was accepted but before rejection memory was saved; the next
   run recovers the recorded decision without re-evaluating the accepted graph as its own control;
5. **deferred host evaluation**: an evaluator that returns no per-task outcomes yet cannot promote a candidate.

The evaluator is synthetic: quality and cost are deterministic functions of the graph's shape, paired over 2,000
task ids. That demonstrates the library's behaviour, not a quality gain or a saving on any real task.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from proceduralgraph import (
    EvolveConfig,
    FileRevisionStore,
    Graph,
    MetricSpec,
    ObjectiveContext,
    ObjectiveGate,
    ObjectiveSpec,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    TaskOutcome,
    Trace,
    evolve,
)
from proceduralgraph.gates import Evaluation
from proceduralgraph.stores.revision import GRAPH

# --- the objective: quality first, then cost --------------------------------------------------------------------------------

QUALITY = MetricSpec("quality", "fraction of tasks solved", "maximize", 0.0, 1.0, min_improvement=0.02, equivalence_margin=0.1, non_regression_margin=0.1)
COST = MetricSpec("cost", "provider calls per task", "minimize", 0.0, 10.0, min_improvement=0.5, equivalence_margin=1.0, non_regression_margin=1.0)
TASK_IDS = tuple(f"task-{i:04d}" for i in range(2000))


def objective(mode: str) -> tuple[ObjectiveSpec, ObjectiveContext]:
    spec = ObjectiveSpec(metrics=(QUALITY, COST), mode=mode, alpha=0.05, min_units=30)
    ctx = ObjectiveContext(
        objective_digest=spec.digest,
        evaluation_design="synthetic-shape-v1",
        evaluator_version="example-1",
        cohort="validation-2000",
        expected_task_ids=TASK_IDS,
        budget_profile="unlimited-demo",
        evidence_partition="example",
        measurement_basis={"cost": "scripted provider calls"},
    )
    return spec, ctx


# --- a synthetic paired evaluator ----------------------------------------------------------------------------------------------


class ShapeEvaluator:
    """quality: 0.5 for the skeleton, 0.9 once search→read guides the agent, 0.3 when it guesses.
    cost: 6 calls floundering, 3 with the read path, 8 when guessing, +5 with an 'expensive' verification node."""

    def __init__(self, ctx: ObjectiveContext, *, unknown_cost_on: str | None = None, deferred: bool = False):
        self.ctx, self.unknown_cost_on, self.deferred = ctx, unknown_cost_on, deferred
        self.calls: list[str] = []

    def observe(self, graph: Graph, task_id: str) -> tuple[float, float | None]:
        q = 0.3 if "guess" in graph.nodes else 0.9 if graph.edges_between("search", "read") else 0.5
        c = 8.0 if "guess" in graph.nodes else 3.0 if graph.edges_between("search", "read") else 6.0
        if "expensive" in graph.nodes:
            c = min(c + 5.0, 10.0)
        return q, (None if task_id == self.unknown_cost_on else c)

    async def evaluate(self, ref, graph, *, iteration, purpose):
        self.calls.append(purpose)
        if self.deferred:
            return Evaluation(ref=ref, score=None, per_task=[], objective_context=self.ctx)  # the host will score it later
        per_task = []
        for task_id in self.ctx.expected_task_ids:
            q, c = self.observe(graph, task_id)
            per_task.append(TaskOutcome(task_id, q, q >= 0.5, metrics={"quality": q, "cost": c}))
        return Evaluation(ref=ref, score=None, per_task=per_task, objective_context=self.ctx)


TRACES = [
    Trace("t-pass", TaskOutcome("q1", 1.0, True, metrics={"quality": 1.0, "cost": 2.0}), "searched, read, answered", nodes_visited=["search", "read", "answer"]),
    Trace("t-fail", TaskOutcome("q2", 0.0, False, metrics={"quality": 0.0, "cost": 5.0}), "searched then guessed", nodes_visited=["search", "answer"]),
]

# --- scripted refiner replies ----------------------------------------------------------------------------------------------------


def edge(source, target, guidance="do it", relation="LEADS_TO"):
    return {"source": source, "target": target, "relation": relation, "condition": None, "guidance": guidance, "pitfalls": None}


def edits(**arrays):
    return json.dumps({"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": [], **arrays})


BUILD = edits(
    add_nodes=[{"id": "search", "type": "ACTION", "description": "s"}, {"id": "read", "type": "ACTION", "description": "r"}, {"id": "answer", "type": "ACTION", "description": "a"}],
    delete_edges=[{"source": "Start", "target": "End"}],
    add_edges=[edge("Start", "search"), edge("search", "read"), edge("read", "answer"), edge("answer", "End")],
)
GUESS = edits(add_nodes=[{"id": "guess", "type": "REASONING", "description": "g"}], add_edges=[edge("search", "guess"), edge("guess", "End")])
NOTE = edits(add_nodes=[{"id": "note", "type": "REASONING", "description": "n"}], add_edges=[edge("read", "note"), edge("note", "End")])
_b = json.loads(BUILD)
BUILD_EXPENSIVE = json.dumps({**_b, "add_nodes": _b["add_nodes"] + [{"id": "expensive", "type": "REASONING", "description": "e"}],
                              "add_edges": _b["add_edges"] + [edge("answer", "expensive"), edge("expensive", "End")]})


class Stores:
    def __init__(self, revisions, workspace):
        self.graph = RevisionGraphStore(revisions, workspace)
        self.rejections = RevisionRejectionStore(revisions, workspace)
        self.checkpoints = RevisionCheckpointStore(revisions, workspace)


async def run(title, revisions, workspace, mode, replies, evaluator, *, rounds, rejection_store=None):
    spec, ctx = objective(mode)
    stores = Stores(revisions, workspace)
    model = ScriptedChatModel(replies)
    print(f"\n=== {title} [{mode}] ===")
    report = await evolve(
        config=EvolveConfig(max_rounds=rounds, role_retries=0, objective=spec, objective_context=ctx, workspace=workspace),
        model=model,
        graph_store=stores.graph,
        rejection_store=rejection_store or stores.rejections,
        checkpoint_store=stores.checkpoints,
        trace_source=StaticTraceSource(TRACES),
        evaluator=evaluator,
        gate=ObjectiveGate(spec, ctx),
    )
    for it in report.iterations:
        d = it.decision or {}
        print(f"round {it.iteration}: outcome={it.outcome} disposition={it.disposition} reasons={d.get('reasons')} decisive={d.get('decisive_metric')}")
        for name, m in (d.get("metrics") or {}).items():
            if m.get("known"):
                iv = m["oriented_interval"]
                print(f"    {name}: {m['baseline_mean']:.3g} -> {m['candidate_mean']:.3g}  oriented [{iv['low']:+.3f}, {iv['high']:+.3f}]  {m.get('verdict', '')}")
            else:
                print(f"    {name}: unknown on at least one unit")
    for w in report.warnings:
        print(f"warning: {w}")
    print(f"head: {report.graph.summary()}  stopped: {report.stopped_reason}  model calls: {report.model_calls}")
    return report, model


async def main(root: Path) -> int:
    revisions = FileRevisionStore(root / "revisions")

    # 1. lexicographic: accept, measured regression, equivalent twice (the second is not a duplicate refusal)
    lex_eval = ShapeEvaluator(objective("lexicographic")[1])
    report, model = await run("1. quality first, then cost", revisions, "lex", "lexicographic", [BUILD, GUESS, NOTE, NOTE], lex_eval, rounds=4)
    assert [i.disposition for i in report.iterations] == ["accepted", "rejected", "equivalent", "equivalent"]
    print("\n--- what the refiner was shown after the regression (rejected-candidates slot, excerpt) ---")
    prompt = model.calls[2].text
    start = prompt.index("Previously rejected candidates:")
    print("\n".join(prompt[start:].splitlines()[:16]))

    # 2. pareto: a better-but-costlier candidate is a trade-off and the incumbent is retained
    par_eval = ShapeEvaluator(objective("pareto")[1])
    report, _ = await run("2. quality and cost both protected", revisions, "pareto", "pareto", [BUILD_EXPENSIVE, BUILD], par_eval, rounds=2)
    assert [i.disposition for i in report.iterations] == ["rejected", "accepted"] and report.iterations[0].decision["reasons"] == ["trade_off"]

    # 3. unknown cost on one unit
    report, _ = await run("3. unknown cost (pareto)", revisions, "unknown-pareto", "pareto", [BUILD], ShapeEvaluator(objective("pareto")[1], unknown_cost_on=TASK_IDS[7]), rounds=1)
    assert report.iterations[0].disposition == "unmeasured" and report.graph.is_skeleton
    report, _ = await run("3. unknown cost (lexicographic)", revisions, "unknown-lex", "lexicographic", [BUILD], ShapeEvaluator(objective("lexicographic")[1], unknown_cost_on=TASK_IDS[7]), rounds=1)
    assert report.iterations[0].disposition == "accepted" and report.iterations[0].decision["unknown_metrics"] == ["cost"]

    # 4. restart recovery: the rejection-memory save fails once, after the graph was accepted
    class DiesOnce(RevisionRejectionStore):
        armed = True

        async def save(self, memory, *, expected_ref, iteration):
            if DiesOnce.armed:
                DiesOnce.armed = False
                raise RuntimeError("worker lost its lease before saving rejection memory")
            return await super().save(memory, expected_ref=expected_ref, iteration=iteration)

    crash_eval = ShapeEvaluator(objective("lexicographic")[1])
    print("\n=== 4. restart recovery [lexicographic] ===")
    try:
        await run("4a. first attempt (will crash after acceptance)", revisions, "recover", "lexicographic", [BUILD], crash_eval, rounds=1,
                  rejection_store=DiesOnce(revisions, "recover"))
    except RuntimeError as exc:
        print(f"first attempt died: {exc}")
    head = await revisions.head("recover", GRAPH)
    print(f"graph head after the crash: seq {head.seq} (the acceptance landed); rejection memory: {(await revisions.head('recover', 'rejections')) is None and 'missing'}")
    report, _ = await run("4b. resumed run", revisions, "recover", "lexicographic", [], crash_eval, rounds=1)
    assert report.outcomes() == ["accepted"] and crash_eval.calls == ["baseline", "candidate"], crash_eval.calls
    print(f"evaluator calls across both attempts: {crash_eval.calls} (no evaluation of the accepted graph as its own control)")

    # 5. deferred host evaluation cannot promote
    report, _ = await run("5. deferred host evaluation", revisions, "deferred", "lexicographic", [BUILD], ShapeEvaluator(objective("lexicographic")[1], deferred=True), rounds=1)
    assert report.outcomes() == ["rejected"] and report.iterations[0].disposition == "unmeasured" and report.graph.is_skeleton

    print("\nall five scenarios behaved as documented")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "./objective-workspace")
    sys.exit(asyncio.run(main(target)))
