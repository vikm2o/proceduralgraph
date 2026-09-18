# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""H1 / G7: budgets and resume. Acceptance §8.10 (resume re-proposes, never re-collects or re-refines) and §8.11
(budget exhaustion before a paid stage leaves the head and rejection memory saved)."""

import pytest

from proceduralgraph import (
    Budget,
    EvolveConfig,
    MemoryRevisionStore,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    evolve,
)
from proceduralgraph.stores.revision import GRAPH, REJECTIONS

from .test_harness_end_to_end import BUILD, VERIFY, Normalising, ScoringEvaluator, traces


class CountingSource(StaticTraceSource):
    def __init__(self, traces):
        super().__init__(traces)
        self.calls = 0

    async def collect(self, graph, *, graph_ref, iteration):
        self.calls += 1
        return await super().collect(graph, graph_ref=graph_ref, iteration=iteration)


class CountingGraphStore(RevisionGraphStore):
    def __init__(self, revisions, workspace):
        super().__init__(revisions, workspace)
        self.proposals: list[int] = []

    async def propose(self, current_ref, current, edits, candidate, *, iteration, rejections_ref):
        self.proposals.append(iteration)
        return await super().propose(current_ref, current, edits, candidate, iteration=iteration, rejections_ref=rejections_ref)


class Interrupting(ScoringEvaluator):
    """Dies on the first candidate evaluation, like a worker that lost its lease mid-validation."""

    def __init__(self):
        super().__init__()
        self.armed = True

    async def evaluate(self, ref, graph, *, iteration, purpose):
        if purpose == "candidate" and self.armed:
            self.armed = False
            raise RuntimeError("worker interrupted during validation")
        return await super().evaluate(ref, graph, iteration=iteration, purpose=purpose)


async def test_resume_finishes_the_interrupted_round_without_repeating_paid_stages():
    revisions = MemoryRevisionStore()
    graph_store = CountingGraphStore(revisions, "ws")
    rejection_store = RevisionRejectionStore(revisions, "ws")
    checkpoints = RevisionCheckpointStore(revisions, "ws")
    source = CountingSource(traces())
    evaluator = Interrupting()
    model = ScriptedChatModel([BUILD, VERIFY])
    common = dict(graph_store=graph_store, rejection_store=rejection_store, trace_source=source, evaluator=evaluator, checkpoint_store=checkpoints)
    with pytest.raises(RuntimeError):
        await evolve(config=EvolveConfig(max_rounds=2, role_retries=0), model=model, **common)
    assert source.calls == 1 and graph_store.proposals == [1] and len(model.calls) == 1
    assert await revisions.head("ws", REJECTIONS) is None  # nothing of the round was recorded
    assert (await checkpoints.load(1))["candidate_digest"] is not None
    # restart on the same stores: iteration 1 is finished from its checkpoint, iteration 2 runs normally
    report = await evolve(config=EvolveConfig(max_rounds=2, role_retries=0), model=model, **common)
    assert report.resumed_iteration == 1
    assert report.outcomes() == ["accepted", "accepted"] and [r.iteration for r in report.iterations] == [1, 2]
    assert graph_store.proposals == [1, 1, 2]  # re-proposed through the host for the same iteration
    assert source.calls == 2 and len(model.calls) == 2  # collect and the refiner ran once more, for iteration 2 only
    assert report.iterations[0].resumed and any("resumed iteration 1" in w for w in report.iterations[0].warnings)
    assert (await checkpoints.load(1)) == {"completed": True}
    assert [c[1] for c in evaluator.calls] == [0, 0, 1, 2]  # baseline twice (no ``baseline`` passed), then each candidate once
    # a third run starts at iteration 3 with nothing to resume
    report = await evolve(config=EvolveConfig(max_rounds=0), model=model, **common)
    assert report.resumed_iteration is None and report.iterations == []


async def test_resume_reuses_a_recorded_validation():
    revisions = MemoryRevisionStore()
    graph_store = CountingGraphStore(revisions, "ws")
    rejection_store = RevisionRejectionStore(revisions, "ws")
    checkpoints = RevisionCheckpointStore(revisions, "ws")

    class DiesAfterValidation(RevisionRejectionStore):
        armed = True

        async def save(self, memory, *, expected_ref, iteration):
            if DiesAfterValidation.armed:
                DiesAfterValidation.armed = False
                raise RuntimeError("lost the lease before saving rejection memory")
            return await super().save(memory, expected_ref=expected_ref, iteration=iteration)

    failing_store = DiesAfterValidation(revisions, "ws")
    evaluator = ScoringEvaluator()
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=graph_store, rejection_store=failing_store,
                     trace_source=StaticTraceSource(traces()), evaluator=evaluator, checkpoint_store=checkpoints)
    assert "evaluation" in await checkpoints.load(1)
    accepted_head = await revisions.head("ws", GRAPH)
    assert accepted_head.seq == 1  # the acceptance landed before the crash
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=graph_store, rejection_store=rejection_store,
                          trace_source=StaticTraceSource(traces()), evaluator=evaluator, checkpoint_store=checkpoints)
    assert report.outcomes() == ["accepted"]
    assert any("already been accepted before the interruption" in w for w in report.iterations[0].warnings)
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate"]  # no second candidate evaluation, and no baseline on the accepted graph
    assert any("baseline taken from the pending checkpoint" in w for w in report.warnings)
    assert (await revisions.head("ws", GRAPH)).seq == 1  # the resumed acceptance did not append a second time


async def test_budget_exhausted_before_a_paid_stage_leaves_saved_state_intact():
    """§8.11."""
    revisions = MemoryRevisionStore()
    graph_store = RevisionGraphStore(revisions, "ws")
    rejection_store = RevisionRejectionStore(revisions, "ws")
    evaluator = ScoringEvaluator()
    # one accepted round first
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([BUILD]), graph_store=graph_store,
                          rejection_store=rejection_store, trace_source=StaticTraceSource(traces()), evaluator=evaluator)
    assert report.outcomes() == ["accepted"]
    head_before = await revisions.head("ws", GRAPH)
    rejections_before = await revisions.head("ws", REJECTIONS)
    # max_evaluations=1 is spent by the baseline; the candidate validation is refused before it spends
    model = ScriptedChatModel([VERIFY])
    report = await evolve(config=EvolveConfig(max_rounds=3, role_retries=0, budget=Budget(max_evaluations=1)), model=model, graph_store=graph_store,
                          rejection_store=rejection_store, trace_source=StaticTraceSource(traces()), evaluator=evaluator)
    assert report.stopped_reason == "budget_exhausted" and report.iterations == []
    assert any("max_evaluations" in w for w in report.warnings)
    assert (await revisions.head("ws", GRAPH)).digest == head_before.digest
    assert (await revisions.head("ws", REJECTIONS)).digest == rejections_before.digest
    assert report.budget["evaluations"] == 1 and report.budget["model_calls"] == 1
    # max_model_calls=0 refuses the refiner call itself; a supplied baseline means no evaluation was needed either
    report = await evolve(config=EvolveConfig(max_rounds=3, role_retries=0, budget=Budget(max_model_calls=0)), model=ScriptedChatModel([VERIFY]),
                          graph_store=graph_store, rejection_store=rejection_store, trace_source=StaticTraceSource(traces()), evaluator=evaluator,
                          baseline=report.best)
    assert report.stopped_reason == "budget_exhausted" and report.iterations == [] and report.model_calls == 0
    # a zero evaluation budget with no baseline stops before line 1
    report = await evolve(config=EvolveConfig(max_rounds=3, budget=Budget(max_evaluations=0)), model=ScriptedChatModel([VERIFY]),
                          graph_store=graph_store, rejection_store=rejection_store, trace_source=StaticTraceSource(traces()), evaluator=evaluator)
    assert report.stopped_reason == "budget_exhausted" and report.best is None


async def test_resume_after_accept_with_a_normalising_host_does_not_accept_twice():
    """The resume-after-accept check compares the HOST's candidate to the head, so a host that normalises is recognised."""
    revisions = MemoryRevisionStore()
    inner = RevisionGraphStore(revisions, "ws")
    host = Normalising(inner)
    checkpoints = RevisionCheckpointStore(revisions, "ws")

    class DiesAfterValidation(RevisionRejectionStore):
        armed = True

        async def save(self, memory, *, expected_ref, iteration):
            if DiesAfterValidation.armed:
                DiesAfterValidation.armed = False
                raise RuntimeError("lost the lease before saving rejection memory")
            return await super().save(memory, expected_ref=expected_ref, iteration=iteration)

    evaluator = ScoringEvaluator()
    model = ScriptedChatModel([BUILD])
    with pytest.raises(RuntimeError):
        await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=host, rejection_store=DiesAfterValidation(revisions, "ws"),
                     trace_source=StaticTraceSource(traces()), evaluator=evaluator, checkpoint_store=checkpoints)
    assert (await revisions.head("ws", GRAPH)).seq == 1
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=host, rejection_store=RevisionRejectionStore(revisions, "ws"),
                          trace_source=StaticTraceSource(traces()), evaluator=evaluator, checkpoint_store=checkpoints)
    assert report.outcomes() == ["accepted"]
    assert any("already been accepted before the interruption" in w for w in report.iterations[0].warnings)
    assert [r.seq for r in await revisions.list("ws", GRAPH)] == [1, 0]  # no second acceptance
    assert host.proposals == [1, 1]  # re-proposed once on resume, as G7 requires
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate"]  # the accepted graph is never evaluated as its own control
