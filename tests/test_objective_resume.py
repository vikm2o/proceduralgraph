# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Resume binding under objective mode (REQ-017): interruptions before evaluation, after evaluation and after graph
acceptance resume with the recorded work and decision; any objective or context mutation refuses before paid work;
legacy unbound checkpoints are refused, never relabelled. TEST-008, part of TEST-009."""

import pytest

from proceduralgraph import (
    EvolveConfig,
    MemoryRevisionStore,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    evolve,
)
from proceduralgraph.gates import ObjectiveGate
from proceduralgraph.stores.revision import CHECKPOINTS, GRAPH, REJECTIONS

from .test_harness_end_to_end import BUILD, ScoringEvaluator
from .test_objective_evolution import CTX, IDS, OBJ, MetricEvaluator, traces
from .test_objectives import context, cost, objective, quality


class Stores:
    def __init__(self):
        self.revisions = MemoryRevisionStore()
        self.graph = RevisionGraphStore(self.revisions, "ws")
        self.rejections = RevisionRejectionStore(self.revisions, "ws")
        self.checkpoints = RevisionCheckpointStore(self.revisions, "ws")
        self.source = StaticTraceSource(traces())
        self.collects = 0
        original = self.source.collect

        async def counting(graph, *, graph_ref, iteration):
            self.collects += 1
            return await original(graph, graph_ref=graph_ref, iteration=iteration)

        self.source.collect = counting  # type: ignore[method-assign]

    def kwargs(self, *, obj=OBJ, ctx=CTX, evaluator, model, gate=None, rejection_store=None, **over):
        return dict(config=EvolveConfig(max_rounds=1, role_retries=0, objective=obj, objective_context=ctx, **over), model=model,
                    graph_store=self.graph, rejection_store=rejection_store or self.rejections, trace_source=self.source, evaluator=evaluator,
                    checkpoint_store=self.checkpoints, gate=gate or ObjectiveGate(obj, ctx))


class DiesBeforeEvaluation(MetricEvaluator):
    armed = True

    async def evaluate(self, ref, graph, *, iteration, purpose):
        if purpose == "candidate" and DiesBeforeEvaluation.armed:
            DiesBeforeEvaluation.armed = False
            raise RuntimeError("worker lost its lease during validation")
        return await super().evaluate(ref, graph, iteration=iteration, purpose=purpose)


class DiesInGate(ObjectiveGate):
    armed = True

    async def decide(self, best, candidate):
        if best is not candidate and DiesInGate.armed:
            DiesInGate.armed = False
            raise RuntimeError("worker lost its lease inside the gate")
        return await super().decide(best, candidate)


class DiesAfterAccept(RevisionRejectionStore):
    armed = True

    async def save(self, memory, *, expected_ref, iteration):
        if DiesAfterAccept.armed:
            DiesAfterAccept.armed = False
            raise RuntimeError("worker lost its lease before saving rejection memory")
        return await super().save(memory, expected_ref=expected_ref, iteration=iteration)


async def test_interrupt_before_evaluation_resumes_with_the_recorded_candidate():
    DiesBeforeEvaluation.armed = True
    s = Stores()
    evaluator = DiesBeforeEvaluation(CTX)
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(**s.kwargs(evaluator=evaluator, model=model))
    pending = await s.checkpoints.load(1)
    assert pending["objective"]["objective_digest"] == OBJ.digest and pending["objective"]["context_digest"] == CTX.digest and "evaluation" not in pending
    assert pending["objective"]["objective"] == OBJ.to_document() and pending["objective"]["context"] == CTX.to_document()  # REQ-017: full documents
    report = await evolve(**s.kwargs(evaluator=evaluator, model=model))
    assert report.outcomes() == ["accepted"] and report.iterations[0].disposition == "accepted" and report.iterations[0].resumed
    assert s.collects == 1 and len(model.calls) == 1  # no second rollout, no second refiner call
    assert [c[0] for c in evaluator.calls] == ["baseline", "baseline", "candidate"]  # the interrupted candidate evaluation never recorded; it runs once


async def test_interrupt_after_evaluation_reuses_it_and_records_the_same_decision():
    DiesInGate.armed = True
    s = Stores()
    evaluator = MetricEvaluator(CTX)
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(**s.kwargs(evaluator=evaluator, model=model, gate=DiesInGate(OBJ, CTX)))
    pending = await s.checkpoints.load(1)
    assert "evaluation" in pending and "baseline_evaluation" in pending and "decision" not in pending
    report = await evolve(**s.kwargs(evaluator=evaluator, model=model))
    assert report.outcomes() == ["accepted"] and any("reused the recorded validation" in w for w in report.iterations[0].warnings)
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate", "baseline"]  # no second candidate evaluation
    assert report.rejections.entries[0].decision["baseline_ref"] == pending["baseline_ref"]


async def test_cached_evaluation_must_identify_the_reproposed_candidate():
    """REQ-004 for the cached path: a host whose propose() hands back a different ref on resume cannot reuse the evaluation."""
    DiesInGate.armed = True
    s = Stores()
    evaluator = MetricEvaluator(CTX)
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(**s.kwargs(evaluator=evaluator, model=model, gate=DiesInGate(OBJ, CTX)))

    class Renaming(RevisionGraphStore):
        async def propose(self, current_ref, current, edits, candidate, *, iteration, rejections_ref):
            hosted = await super().propose(current_ref, current, edits, candidate, iteration=iteration, rejections_ref=rejections_ref)
            hosted.ref = "bundle-" + hosted.ref[:8]  # a different identity than the first attempt recorded
            return hosted

    kwargs = s.kwargs(evaluator=evaluator, model=model)
    kwargs["graph_store"] = Renaming(s.revisions, "ws")
    with pytest.raises(ValueError, match="REQ-004"):
        await evolve(**kwargs)
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate", "baseline"]  # refused before any further paid work
    assert (await s.revisions.head("ws", GRAPH)).seq == 0


async def test_interrupt_after_acceptance_recovers_the_decision_without_a_control_evaluation():
    DiesAfterAccept.armed = True
    s = Stores()
    evaluator = MetricEvaluator(CTX)
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(**s.kwargs(evaluator=evaluator, model=model, rejection_store=DiesAfterAccept(s.revisions, "ws")))
    assert (await s.revisions.head("ws", GRAPH)).seq == 1 and await s.revisions.head("ws", REJECTIONS) is None
    pending = await s.checkpoints.load(1)
    assert pending["decision"]["accepted"] is True and pending["decision"]["feedback"]["disposition"] == "accepted"
    report = await evolve(**s.kwargs(evaluator=evaluator, model=model))
    assert report.outcomes() == ["accepted"] and report.iterations[0].disposition == "accepted"
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate"]  # the accepted graph was never evaluated as its own control
    assert any("baseline taken from the pending checkpoint" in w for w in report.warnings)
    entry = report.rejections.entries[0]
    assert entry.decision == pending["decision"]["feedback"] and entry.candidate_ref == pending["host_ref"] and entry.baseline_ref == pending["baseline_ref"]
    assert (await s.revisions.head("ws", GRAPH)).seq == 1  # no second acceptance


@pytest.mark.parametrize(
    "mutation",
    [
        lambda: (objective(metrics=(quality(min_improvement=0.03), cost())), None),
        lambda: (objective(mode="pareto"), None),
        lambda: (objective(metrics=(cost(), quality())), None),
        lambda: (objective(metrics=(quality(upper_bound=2.0), cost())), None),
        lambda: (objective(metrics=(quality(equivalence_margin=0.2), cost())), None),
        lambda: (objective(alpha=0.1), None),
        lambda: (None, context(OBJ, ids=IDS, cohort="cohort-b")),
        lambda: (None, context(OBJ, ids=IDS, evaluator_version="eval-4")),
        lambda: (None, context(OBJ, ids=IDS, budget_profile="budget-q")),
        lambda: (None, context(OBJ, ids=IDS, evidence_partition="validation-2026-10")),
        lambda: (None, context(OBJ, ids=IDS, measurement_basis={"cost": "billed dollars"})),
        lambda: (None, context(OBJ, ids=IDS[:-1])),
    ],
    ids=["margin", "mode", "order", "bound", "tolerance", "alpha", "cohort", "evaluator", "budget", "partition", "basis", "task-ids"],
)
async def test_every_mutation_refuses_the_pending_attempt_before_paid_work(mutation):
    DiesBeforeEvaluation.armed = True
    s = Stores()
    evaluator = DiesBeforeEvaluation(CTX)
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(**s.kwargs(evaluator=evaluator, model=model))
    before = [r.document for r in await s.revisions.list("ws", CHECKPOINTS)]
    new_obj, new_ctx = mutation()
    obj = new_obj or OBJ
    ctx = new_ctx or context(obj, ids=IDS)
    fresh = MetricEvaluator(ctx)
    with pytest.raises(ValueError, match="pending checkpoint"):
        await evolve(**s.kwargs(obj=obj, ctx=ctx, evaluator=fresh, model=ScriptedChatModel([BUILD])))
    assert fresh.calls == [] and s.collects == 1  # refused before the baseline, the rollout or the refiner
    assert [r.document for r in await s.revisions.list("ws", CHECKPOINTS)] == before  # the record is preserved, untouched
    assert await s.revisions.head("ws", REJECTIONS) is None and (await s.revisions.head("ws", GRAPH)).seq == 0


async def test_legacy_unbound_checkpoint_is_refused_in_objective_mode_and_vice_versa():
    # a scalar-mode run leaves a pending checkpoint (no objective binding)
    from .test_budget_and_resume import Interrupting

    s = Stores()
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=s.graph, rejection_store=s.rejections,
                     trace_source=s.source, evaluator=Interrupting(), checkpoint_store=s.checkpoints)
    assert "objective" not in await s.checkpoints.load(1)
    with pytest.raises(ValueError, match="bound to no objective"):
        await evolve(**s.kwargs(evaluator=MetricEvaluator(CTX), model=model))
    assert "objective" not in await s.checkpoints.load(1)  # never relabelled
    # and the other way round: an objective-bound pending checkpoint cannot be finished by a scalar run
    DiesBeforeEvaluation.armed = True
    s2 = Stores()
    with pytest.raises(RuntimeError):
        await evolve(**s2.kwargs(evaluator=DiesBeforeEvaluation(CTX), model=ScriptedChatModel([BUILD])))
    with pytest.raises(ValueError, match="configured with no objective"):
        await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([BUILD]), graph_store=s2.graph, rejection_store=s2.rejections,
                     trace_source=s2.source, evaluator=ScoringEvaluator(), checkpoint_store=s2.checkpoints)


async def test_crash_after_accept_recovers_when_the_host_mints_its_own_refs():
    """The checkpointed evaluation's ref is the host's propose() ref, not necessarily what load() reports for the head."""
    DiesAfterAccept.armed = True
    s = Stores()

    class BundleRefs(RevisionGraphStore):
        async def propose(self, current_ref, current, edits, candidate, *, iteration, rejections_ref):
            hosted = await super().propose(current_ref, current, edits, candidate, iteration=iteration, rejections_ref=rejections_ref)
            hosted.ref = "bundle-" + candidate.digest[:8]
            return hosted

    class BundleEvaluator(MetricEvaluator):
        async def evaluate(self, ref, graph, *, iteration, purpose):
            if purpose == "baseline":
                ref = ref  # the head ref from load(): a digest
            return await super().evaluate(ref, graph, iteration=iteration, purpose=purpose)

    evaluator = BundleEvaluator(CTX)
    model = ScriptedChatModel([BUILD])
    kwargs = s.kwargs(evaluator=evaluator, model=model, rejection_store=DiesAfterAccept(s.revisions, "ws"))
    kwargs["graph_store"] = BundleRefs(s.revisions, "ws")
    with pytest.raises(RuntimeError):
        await evolve(**kwargs)
    kwargs = s.kwargs(evaluator=evaluator, model=model)
    kwargs["graph_store"] = BundleRefs(s.revisions, "ws")
    report = await evolve(**kwargs)
    assert report.outcomes() == ["accepted"] and [c[0] for c in evaluator.calls] == ["baseline", "candidate"]
    assert report.rejections.entries[0].candidate_ref.startswith("bundle-")
