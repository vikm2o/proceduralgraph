# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Algorithm 1 (paper App. B.6) with the production extensions catalogued in docs/paper-differences.md. REQUIREMENTS G.

    S_0 <- Evaluate(G_0, D_val)                               (line 1; skipped when the host passes ``baseline``)
    H_rejected <- stored rejection memory                     (line 2)
    for k in 1..K:                                            (line 3; K = config.max_rounds)
        G_k <- G_{k-1}; S_k <- S_{k-1}                         (line 4; retain unless accepted)
        E_k <- trace_source.collect(G_{k-1})                  (lines 5-6; rollouts, or scored production episodes)
        C_k <- Tail_Lmax(ConcatTrajectories(E_k))             (line 7)
        R_k <- SerializeRejections(H_rejected)                (line 8)
        ΔG_k <- Refiner(G_{k-1}, C_k, {S_i}, R_k)             (line 9)
        (G_cand, d_k) <- PrepareCandidate(G_{k-1}, ΔG_k, c)   (line 10)
        if d_k != ∅: record structural failure; continue      (lines 11-13; no validation rollout)
        S_cand <- Evaluate(G_cand, D_val)                     (line 15; skipped for a duplicate candidate, F3)
        accept iff gate.decide(S_{k-1}, S_cand)               (lines 16-20; default gate: >=, ties accepted)
        persist rejection memory once, after the outcome      (G4)
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Budget, EvolveConfig
from .documents import HeadMoved
from .edits import EditSet, unified_diff
from .gates import Decision, Evaluation, Evaluator, Gate, TieAcceptingGate
from .graph import Diagnostic, Graph, GraphError, errors
from .hooks import BudgetExceeded, BudgetMeter, HookedModel, Hooks, IterationReport
from .model import ChatModel, ImagePart
from .redaction import Redactor, redact_trace
from .rejections import RejectionEntry, RejectionMemory
from .roles.bootstrap import BootstrapResult
from .roles.refiner import Refiner
from .stores.base import CheckpointStore, GraphStore, NullCheckpointStore, NullTraceStore, RejectionStore, TraceStore
from .traces import Trace, TraceSource, render_attempts_block, stratified_sample

STOPPED_REASONS = ("max_rounds", "rejected_streak", "perfect", "budget_exhausted", "gate_stop")


@dataclass
class RunReport:
    iterations: list[IterationReport]
    graph_ref: str | None
    graph: Graph
    rejections_ref: str | None
    rejections: RejectionMemory
    best: Evaluation | None  # None only when the run stopped before the baseline evaluation (budget_exhausted)
    stopped_reason: str
    budget: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    model_calls: int = 0
    resumed_iteration: int | None = None

    @property
    def accepted(self) -> int:
        return sum(1 for r in self.iterations if r.outcome == "accepted")

    def outcomes(self) -> list[str]:
        return [r.outcome for r in self.iterations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [r.to_dict() for r in self.iterations],
            "outcomes": self.outcomes(),
            "graph_ref": self.graph_ref,
            "graph": {"nodes": len(self.graph.nodes), "edges": len(self.graph.edges), "digest": self.graph.digest},
            "rejections_ref": self.rejections_ref,
            "best": self.best.to_dict() if self.best is not None else None,
            "stopped_reason": self.stopped_reason,
            "budget": self.budget,
            "warnings": list(self.warnings),
            "model_calls": self.model_calls,
            "resumed_iteration": self.resumed_iteration,
        }


def resolve_mode(config_mode: str, graph: Graph) -> str:
    """E4: ``auto`` is ``scratch_incremental`` on the skeleton and ``static_incremental`` otherwise."""
    if config_mode != "auto":
        return config_mode
    return "scratch_incremental" if graph.is_skeleton else "static_incremental"


def coerce_initial_graph(value: Graph | BootstrapResult | dict[str, Any] | Path | None) -> tuple[Graph, dict[str, Any], list[Diagnostic]]:
    """G1a: ``None`` (or an empty document) is the skeleton; a ``Graph`` is validated as is; a ``BootstrapResult`` is
    its graph with ``origin: bootstrapped`` and the bootstrap's meta (§2.19); a dict or path goes through
    :meth:`Graph.from_human`. Returns (graph, seed meta, warnings)."""
    if value is None:
        return Graph.skeleton(), {"origin": "skeleton"}, []
    if isinstance(value, BootstrapResult):
        if value.graph is None:
            raise GraphError(value.diagnostics, value.refusal())
        return value.graph, value.seed_meta(), [d for d in value.diagnostics if not d.is_error]
    if isinstance(value, Graph):
        found = value.validate()
        if errors(found):
            raise GraphError(errors(found), "initial_graph fails the structural checks: " + "; ".join(str(d) for d in errors(found)))
        return value, {"origin": "skeleton" if value.is_skeleton else "onboarded"}, [d for d in found if not d.is_error]
    graph, notes = Graph.from_human(value)
    return graph, {"origin": "skeleton" if graph.is_skeleton else "onboarded"}, notes


async def evolve(
    *,
    config: EvolveConfig,
    model: ChatModel,
    graph_store: GraphStore,
    rejection_store: RejectionStore,
    trace_source: TraceSource,
    evaluator: Evaluator,
    gate: Gate | None = None,
    trace_store: TraceStore | None = None,
    checkpoint_store: CheckpointStore | None = None,
    hooks: Hooks | None = None,
    initial_graph: Graph | BootstrapResult | dict[str, Any] | Path | None = None,
    baseline: Evaluation | None = None,
    available_tools: Sequence[str] | None = None,
    redact: Redactor | None = None,
) -> RunReport:
    """Run up to ``config.max_rounds`` rounds of Algorithm 1 and persist the results.

    ``initial_graph`` seeds an empty workspace (a hand-written graph, a ``BootstrapResult``, or nothing for the
    skeleton); supplying one for a workspace that already has a graph is an error. ``baseline`` lets a host that already scored the head skip
    line 1. ``checkpoint_store`` enables resume: a round interrupted after its refiner stage is finished, not
    repeated. ``redact`` is applied to every collected trace before the refiner, the trace store or rejection memory
    see it. ``available_tools`` fills the prompt's tool list and produces ``unknown_action`` warnings.
    """
    hooks = hooks or Hooks()
    gate = gate or TieAcceptingGate()
    objective = config.objective
    # Objective mode (REQ-007, REQ-017): every checkpoint and decision this run writes is bound to these digests, and a
    # pending checkpoint bound to anything else (or to nothing) is refused rather than reused.
    binding: dict[str, Any] | None = None
    startup_warnings: list[str] = []
    if objective is not None:
        binding = {
            "objective_digest": objective.digest,
            "context_digest": config.objective_context.digest,
            "objective": objective.to_document(),
            "context": config.objective_context.to_document(),
        }
        gate_objective, gate_context = getattr(gate, "objective", None), getattr(gate, "context", None)
        if gate_objective is not None or gate_context is not None:
            if gate_objective is None or gate_context is None or gate_objective.digest != objective.digest or gate_context.digest != binding["context_digest"]:
                raise ValueError("the ObjectiveGate's objective/context digests differ from EvolveConfig.objective / objective_context")
        else:
            startup_warnings.append(
                f"an objective is configured but the gate ({type(gate).__name__}) is not an ObjectiveGate; the refiner is told decisions are "
                "measured against the objective, so make sure this gate applies it"
            )
    trace_store = trace_store or NullTraceStore()
    checkpoints = checkpoint_store or NullCheckpointStore()
    if isinstance(model, HookedModel):
        # G9: the host shares one HookedModel between its Guides and this loop so guidance calls are metered under the
        # same budget. Reuse its meter; adopt config.budget when the host's meter has no ceilings of its own.
        hooked = model
        if hooked.meter.budget == Budget():
            hooked.meter.budget = config.budget
        meter = hooked.meter
    else:
        meter = BudgetMeter(config.budget)
        hooked = HookedModel(model, hooks, meter)
    refiner = Refiner(hooked, config)
    run_warnings: list[str] = []

    async def warn(message: str) -> None:
        run_warnings.append(message)
        await hooks.on_warning(message)

    for message in startup_warnings:
        await warn(message)

    # -- state: the graph (G_0 or the stored head) and rejection memory ------------------------------------------------
    graph_ref, graph = await graph_store.load()
    if graph_ref is None:
        graph, seed_meta, notes = coerce_initial_graph(initial_graph)
        for note in notes:
            await warn(f"initial graph: {note}")
        graph_ref = await graph_store.seed(graph, meta=seed_meta)
    elif initial_graph is not None:
        raise ValueError(
            f"initial_graph was supplied but the workspace already has a graph (head {graph_ref[:12]}); "
            "use a new workspace or seed_workspace() instead of overwriting"
        )
    rejections_ref, memory = await rejection_store.load()

    # -- baseline (Alg. 1 line 1) -----------------------------------------------------------------------------------------
    # -- resume (G7): a round whose refiner output was recorded but never finished is completed, not repeated -------------
    start = memory.iteration + 1
    resume: dict[str, Any] | None = None
    recorded = await checkpoints.load(start)
    if recorded and not recorded.get("completed") and "edits" in recorded:
        recorded_binding = recorded.get("objective")
        same_binding = (recorded_binding is None and binding is None) or (
            isinstance(recorded_binding, dict) and binding is not None
            and recorded_binding.get("objective_digest") == binding["objective_digest"]
            and recorded_binding.get("context_digest") == binding["context_digest"]
        )
        if not same_binding:
            # REQ-017: an attempt made under another objective/context, or under none, cannot be reused; the record stays.
            raise ValueError(
                f"workspace {config.workspace!r} has a pending checkpoint for iteration {start} bound to "
                f"{'no objective' if recorded.get('objective') is None else 'objective ' + recorded['objective']['objective_digest'][:12]} "
                f"while this run is configured with {'no objective' if binding is None else 'objective ' + binding['objective_digest'][:12]}; "
                "start a new run in a fresh workspace (the pending record is kept, never relabelled or deleted)"
            )
        resume = recorded

    baseline_from_checkpoint = False
    if (
        baseline is None
        and resume is not None
        and resume.get("evaluation")
        and resume.get("host_candidate_digest") == graph.digest
        and resume.get("graph_ref") != graph_ref
    ):
        # The interrupted round's candidate IS the head (its acceptance landed before the crash). Its recorded evaluation
        # is the head's score; re-evaluating it here would use the accepted graph as its own control (REQ-017). Its ref is
        # the host's propose() ref for that candidate, which need not equal the ref load() reports for the head.
        baseline = Evaluation.from_document(resume["evaluation"])
        baseline_from_checkpoint = True
        await warn("baseline taken from the pending checkpoint: the head is the candidate accepted before the interruption")
    if baseline is None:
        try:
            meter.check(about_to="evaluation")
            async with hooks.stage("baseline"):
                baseline = await evaluator.evaluate(graph_ref, graph, iteration=0, purpose="baseline")
        except BudgetExceeded as exc:  # before the evaluation, or inside it when the host's Guides share the meter
            await warn(str(exc))
            return RunReport(iterations=[], graph_ref=graph_ref, graph=graph, rejections_ref=rejections_ref, rejections=memory, best=None,
                             stopped_reason="budget_exhausted", budget=meter.to_dict(), warnings=run_warnings, model_calls=hooked.calls)
        meter.evaluations += 1
    if objective is not None and not baseline_from_checkpoint:
        _check_ref(baseline, graph_ref, "baseline")
    best = baseline
    if resume is None and (await gate.decide(best, best)).stop:
        # A pending checkpoint takes precedence: the interrupted round is reconciled first (its acceptance may be what
        # made the head perfect), and the perfect stop then fires after it in the loop.
        return RunReport(iterations=[], graph_ref=graph_ref, graph=graph, rejections_ref=rejections_ref, rejections=memory, best=best,
                         stopped_reason="perfect", budget=meter.to_dict(), warnings=run_warnings, model_calls=hooked.calls)

    seen_trace_ids: set[str] = {t for e in memory.entries for t in e.trace_ids}
    reports: list[IterationReport] = []
    stopped = "max_rounds"
    rejected_streak = 0

    for k in range(start, start + config.max_rounds):
        try:
            meter.check(about_to="iteration")
        except BudgetExceeded as exc:
            await warn(str(exc))
            stopped = "budget_exhausted"
            break
        checkpoint = resume if (resume is not None and k == start) else None
        warnings: list[str] = []
        outcome = "structural_failure"
        edits: EditSet | None = None
        candidate: Graph | None = None
        candidate_digest: str | None = None
        diagnostics: list[Diagnostic] = []
        evaluation: Evaluation | None = None
        decision_stop = False
        decision: Decision | None = None
        hosted_ref: str | None = None
        duplicate_of: int | None = None
        previous_best_score = best.score
        trace_ids: list[str] = []
        trace_scores: dict[str, float] = {}
        stale_ids: list[str] = []
        mode = resolve_mode(config.mode, graph)
        checkpointed = False
        record: dict[str, Any] = {}
        try:
            if checkpoint is None:
                # -- rollout (lines 5-6) --
                async with hooks.stage(f"rollout:{k}"):
                    traces = await trace_source.collect(graph, graph_ref=graph_ref, iteration=k)
                if redact is not None:
                    traces = [redact_trace(t, redact) for t in traces]
                traces, stale_ids = _apply_stale_policy(traces, graph_ref, config.stale_trace_policy)
                if stale_ids and config.stale_trace_policy == "warn":
                    warnings.append(f"{len(stale_ids)} trace(s) ran under a graph other than the head {graph_ref[:12] if graph_ref else '?'}: {', '.join(stale_ids[:5])}")
                elif stale_ids and config.stale_trace_policy == "drop":
                    warnings.append(f"{len(stale_ids)} stale trace(s) dropped: {', '.join(stale_ids[:5])}")
                await trace_store.put(traces, iteration=k)
                sample, seen_trace_ids = stratified_sample(
                    traces, failing=config.failing_sample, passing=config.passing_sample, seen=seen_trace_ids, seed=config.seed, iteration=k
                )
                trace_ids = [t.id for t in sample]
                trace_scores = {t.id: t.outcome.score for t in sample}
                # -- C_k, R_k, the refiner (lines 7-9) --
                attempts_block = render_attempts_block(
                    sample, cap=config.refiner_context_cap, token_counter=config.token_counter, success_threshold=config.success_threshold,
                    head_ref=graph_ref,
                )
                rejected_block = memory.render_for_refiner(
                    full_entries=config.rejections_full_entries, char_cap=config.rejections_char_cap,
                    objective_summary=None if objective is None else "Objective for this run (every decision below is measured against it):\n" + objective.summary(),
                )
                images = _refiner_images(sample, config.refiner_images)
                async with hooks.stage(f"refiner:{k}"):
                    result = await refiner.propose(
                        graph, mode=mode, attempts_block=attempts_block, rejected_block=rejected_block, available_tools=available_tools, images=images
                    )
                edits, candidate, diagnostics = result.edits, result.candidate, list(result.diagnostics)
                if result.calls > 1:
                    warnings.append(f"refiner retried {result.calls - 1} time(s) after diagnostics were fed back")
                candidate_digest = candidate.digest if candidate is not None else None
                # The module's refiner output is recorded once and never regenerated: a resume re-proposes it through the
                # host exactly as the first attempt did.
                record = {
                    "edits": edits.to_dict(graph.attribute_fields) if edits is not None else None,
                    "candidate": candidate.to_document() if candidate is not None else None,
                    "candidate_digest": candidate_digest,
                    "diagnostics": [d.to_dict() for d in diagnostics],
                    "rejections_ref": rejections_ref,
                    "graph_ref": graph_ref,
                    "trace_ids": trace_ids,
                    "trace_scores": trace_scores,
                    "stale_trace_ids": stale_ids,
                    "mode": mode,
                }
                if binding is not None:
                    record["objective"] = binding
                await checkpoints.save(k, record)
                checkpointed = True
            else:
                record = dict(checkpoint)
                edits = EditSet.from_dict(checkpoint["edits"], graph.attribute_fields) if checkpoint.get("edits") is not None else None
                candidate = Graph.from_document(checkpoint["candidate"]) if checkpoint.get("candidate") else None
                candidate_digest = checkpoint.get("candidate_digest")
                diagnostics = [Diagnostic.from_dict(d) for d in checkpoint.get("diagnostics", [])]
                trace_ids = list(checkpoint.get("trace_ids", []))
                trace_scores = dict(checkpoint.get("trace_scores", {}))
                stale_ids = list(checkpoint.get("stale_trace_ids", []))
                mode = checkpoint.get("mode", mode)
                checkpointed = True
                warnings.append(f"resumed iteration {k} from its checkpoint; the rollout and refiner stages were not repeated")

            # -- PrepareCandidate outcome (lines 10-13) --
            if edits is None:
                outcome = "structural_failure"  # the reply could not be parsed at all (malformed_edit diagnostics)
            elif edits.is_empty:
                outcome = "no_action"
            elif candidate is None:
                outcome = "structural_failure"
            else:
                # -- the host's candidate is canonical (G3): materialise first, then every digest comparison below is
                # host-to-host (the duplicate check, the resume check, and what rejection memory records).
                hosted = await graph_store.propose(graph_ref, graph, edits, candidate, iteration=k, rejections_ref=rejections_ref)
                if hosted.graph.digest != candidate.digest:
                    warnings.append(
                        f"host materialised a candidate that differs from the module's ({len(hosted.graph.nodes)}/{len(hosted.graph.edges)} "
                        f"vs {len(candidate.nodes)}/{len(candidate.edges)} nodes/edges); continuing with the host's"
                    )
                    candidate = hosted.graph
                    candidate_digest = candidate.digest
                hosted_ref = hosted.ref
                host_errors = errors(hosted.graph.validate())  # the host's graph is what gets validated and served: check it too
                known = _known_candidate(candidate.digest, graph, memory)
                if host_errors:
                    diagnostics = diagnostics + host_errors
                    outcome = "structural_failure"
                    warnings.append(f"the host's materialised candidate fails the structural checks: {'; '.join(str(d) for d in host_errors)}")
                elif (
                    checkpoint is not None
                    and checkpoint.get("evaluation")
                    and candidate.digest == graph.digest
                    and checkpoint.get("graph_ref") != graph_ref
                ):
                    # The acceptance landed before the interruption (the head IS this candidate) but the rejection-memory
                    # save did not. Record the outcome from the checkpoint without re-accepting.
                    evaluation = Evaluation.from_document(checkpoint["evaluation"])
                    if objective is not None:
                        _check_cached_ref(checkpoint, evaluation, hosted.ref)
                    if checkpoint.get("decision"):
                        decision = _recorded_decision(checkpoint["decision"])
                    if checkpoint.get("baseline_evaluation"):
                        previous_best_score = Evaluation.from_document(checkpoint["baseline_evaluation"]).score
                    best, outcome = evaluation, "accepted"
                    warnings.append("the resumed candidate had already been accepted before the interruption; recorded from its checkpoint")
                elif known is not None:
                    outcome, duplicate_of = "duplicate_candidate", known[1]
                    warnings.append(f"candidate {candidate.digest[:12]} refused without validation: {known[0]}")
                else:
                    # -- validation (line 15) --
                    if checkpoint is not None and checkpoint.get("evaluation"):
                        evaluation = Evaluation.from_document(checkpoint["evaluation"])
                        if objective is not None:
                            _check_cached_ref(checkpoint, evaluation, hosted.ref)
                        warnings.append("reused the recorded validation of the resumed candidate")
                        if checkpoint.get("baseline_evaluation"):
                            best = Evaluation.from_document(checkpoint["baseline_evaluation"])  # the control the decision was (or will be) made against
                            previous_best_score = best.score
                    else:
                        meter.check(about_to="evaluation")
                        async with hooks.stage(f"validation:{k}"):
                            evaluation = await evaluator.evaluate(hosted.ref, candidate, iteration=k, purpose="candidate")
                        meter.evaluations += 1
                        if objective is not None:
                            _check_ref(evaluation, hosted.ref, "candidate")
                        record = {**record, "host_candidate_digest": candidate.digest, "host_ref": hosted.ref,
                                  "evaluation": evaluation.to_document(), "baseline_evaluation": best.to_document(), "baseline_ref": graph_ref}
                        await checkpoints.save(k, record)
                    # -- the gate (lines 16-20) --
                    if checkpoint is not None and checkpoint.get("decision"):
                        decision = _recorded_decision(checkpoint["decision"])
                        warnings.append("reused the recorded gate decision of the resumed candidate")
                    else:
                        decision = await gate.decide(best, evaluation)
                        # REQ-017: the decision is durable before the head moves, so a crash after accept() recovers the
                        # same decision without re-deciding against the accepted graph as its own control.
                        await checkpoints.save(k, {**record, "host_candidate_digest": candidate.digest,
                                                   "decision": {"accepted": decision.accepted, "stop": decision.stop, "feedback": decision.feedback}})
                    if decision.accepted:
                        hosted.meta["validation_score"] = evaluation.score
                        hosted.meta["iteration"] = k
                        hosted.meta["baseline_ref"] = graph_ref
                        hosted.meta["candidate_ref"] = hosted.ref
                        if decision.feedback:
                            hosted.meta["decision"] = copy.deepcopy(decision.feedback)
                        graph_ref = await graph_store.accept(hosted, expected_ref=graph_ref)
                        graph, best, outcome = candidate, evaluation, "accepted"
                    else:
                        outcome = "rejected"
                    decision_stop = decision.stop
        except BudgetExceeded as exc:
            # Abandon this round: nothing of it is persisted (the checkpoint stays so a resume with more budget finishes it).
            warnings.append(str(exc))
            for message in warnings:
                await warn(message)
            stopped = "budget_exhausted"
            break

        # -- rejection memory: one entry per round, written after the outcome is known (lines 12, 19; G4) --
        entry = RejectionEntry(
            iteration=k,
            kind=outcome,
            edits=edits or EditSet(),
            candidate=candidate,
            candidate_digest=candidate_digest,
            base_digest=record.get("graph_ref") or "",
            trace_ids=trace_ids,
            trace_scores=trace_scores,
            validation=evaluation,
            retained_score=previous_best_score if outcome in ("accepted", "rejected") else None,
            diagnostics=list(diagnostics),
            duplicate_of=duplicate_of,
            mode=mode,
            decision=copy.deepcopy(decision.feedback) if decision is not None and decision.feedback else None,
            baseline_ref=record.get("baseline_ref") if decision is not None else None,
            candidate_ref=hosted_ref if decision is not None else None,
        )
        memory.record(entry)
        rejections_ref = await rejection_store.save(memory, expected_ref=rejections_ref, iteration=k)
        if checkpointed:
            await checkpoints.save(k, {"completed": True})

        if outcome == "accepted":
            rejected_streak = 0
        else:
            rejected_streak += 1
        report = IterationReport(
            iteration=k,
            trace_ids=trace_ids,
            stale_trace_ids=stale_ids,
            edits=(edits or EditSet()).to_dict(graph.attribute_fields),
            candidate_digest=candidate_digest,
            outcome=outcome,
            evaluation=evaluation.to_dict() if evaluation is not None else None,
            diagnostics=[d.to_dict() for d in diagnostics],
            warnings=warnings,
            graph_ref=graph_ref,
            rejections_ref=rejections_ref,
            resumed=checkpoint is not None,
            decision=copy.deepcopy(decision.feedback) if decision is not None and decision.feedback else None,
        )
        reports.append(report)
        for message in warnings:
            await warn(message)
        await hooks.on_iteration(report)

        if outcome == "accepted" and (await gate.decide(best, best)).stop:
            stopped = "perfect"
            break
        if decision_stop:
            stopped = "gate_stop"
            break
        if config.max_rejected_streak is not None and rejected_streak >= config.max_rejected_streak:
            stopped = "rejected_streak"
            break

    return RunReport(
        iterations=reports,
        graph_ref=graph_ref,
        graph=graph,
        rejections_ref=rejections_ref,
        rejections=memory,
        best=best,
        stopped_reason=stopped,
        budget=meter.to_dict(),
        warnings=run_warnings,
        model_calls=hooked.calls,
        resumed_iteration=start if resume else None,
    )


def _apply_stale_policy(traces: list[Trace], graph_ref: str | None, policy: str) -> tuple[list[Trace], list[str]]:
    stale = [t.id for t in traces if t.graph_ref is not None and graph_ref is not None and t.graph_ref != graph_ref]
    if policy == "drop":
        stale_set = set(stale)
        return [t for t in traces if t.id not in stale_set], stale
    return traces, (stale if policy == "warn" else [])


def _refiner_images(traces: list[Trace], limit: int) -> list[ImagePart]:
    if limit <= 0:
        return []
    ordered = [t for t in reversed(traces) if not t.outcome.passed] + [t for t in reversed(traces) if t.outcome.passed]  # most recent failing first
    images: list[ImagePart] = []
    for trace in ordered:
        for image in trace.media:
            if len(images) >= limit:
                return images
            images.append(image)
    return images


def _check_ref(evaluation: Evaluation, expected_ref: str | None, what: str) -> None:
    """REQ-004: an evaluation must identify the graph the loop asked it to evaluate (objective mode only)."""
    if evaluation.ref != expected_ref:
        raise ValueError(f"the {what} Evaluation.ref {evaluation.ref!r} does not identify the graph evaluated ({expected_ref!r}) (REQ-004)")


def _check_cached_ref(checkpoint: dict[str, Any], evaluation: Evaluation, hosted_ref: str) -> None:
    """REQ-004 for the cached path: the recorded evaluation must be of the candidate the host just re-proposed."""
    if checkpoint.get("host_ref") != hosted_ref or evaluation.ref != hosted_ref:
        raise ValueError(
            f"the cached Evaluation.ref {evaluation.ref!r} (recorded for {checkpoint.get('host_ref')!r}) does not identify the re-proposed "
            f"candidate {hosted_ref!r}; the host's propose() is not idempotent for this round (REQ-004)"
        )


def _recorded_decision(value: dict[str, Any]) -> Decision:
    return Decision(accepted=bool(value.get("accepted")), feedback=copy.deepcopy(dict(value.get("feedback", {}))), stop=bool(value.get("stop")))


def _known_candidate(candidate_digest: str, current: Graph, memory: RejectionMemory) -> tuple[str, int | None] | None:
    """F3: a candidate the loop already has an answer for: identical to the head, measured as worse earlier, or
    structurally failed earlier with a recorded digest. Equivalent / unresolved / unmeasured comparisons do not count."""
    if candidate_digest == current.digest:
        return ("identical to the current head graph", None)
    earlier = memory.rejected_digests().get(candidate_digest)
    if earlier is not None:
        return (f"identical to the candidate rejected in iteration {earlier}", earlier)
    return None


def evolve_sync(**kwargs: Any) -> RunReport:
    """Convenience wrapper for scripts and notebooks without an event loop."""
    return asyncio.run(evolve(**kwargs))


async def refine_once(
    *,
    config: EvolveConfig,
    model: ChatModel,
    graph: Graph,
    traces: Sequence[Trace],
    available_tools: Sequence[str] | None = None,
    mode: str | None = None,
    hooks: Hooks | None = None,
    attempts_block: str | None = None,
) -> tuple[Graph | None, EditSet, list[Diagnostic]]:
    """The one-time modes (App. D.2 Modes 2 and 4, ``static_onetime`` / ``scratch_onetime``): one refiner pass over
    all traces, no store, no gate, no rejection memory (G6). Returns (candidate or None, the edit set, diagnostics).

    ``attempts_block`` replaces the rendered traces in the prompt's trajectories slot (used by ``bootstrap_graph`` when
    there are no traces at all). A ``HookedModel`` passed as ``model`` is reused, so the caller's meter counts the call.
    """
    hooked = model if isinstance(model, HookedModel) else HookedModel(model, hooks or Hooks(), BudgetMeter(config.budget))
    refiner = Refiner(hooked, config)
    chosen = mode or ("scratch_onetime" if graph.is_skeleton else "static_onetime")
    if chosen not in ("static_onetime", "scratch_onetime"):
        raise ValueError("refine_once runs the one-time modes only: static_onetime or scratch_onetime")
    if attempts_block is None:
        attempts_block = render_attempts_block(
            list(traces), cap=config.refiner_context_cap, token_counter=config.token_counter, success_threshold=config.success_threshold
        )
    result = await refiner.propose(
        graph, mode=chosen, attempts_block=attempts_block, rejected_block="(none: one-time mode)", available_tools=available_tools
    )
    return result.candidate, result.edits or EditSet(), list(result.diagnostics)


__all__ = ["STOPPED_REASONS", "HeadMoved", "IterationReport", "RunReport", "coerce_initial_graph", "evolve", "evolve_sync", "refine_once", "resolve_mode", "unified_diff"]
