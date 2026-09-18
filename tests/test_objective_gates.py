# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""ObjectiveGate (REQ-007 to REQ-014): the decision table in both modes, absolute constraints, evidence rules, missing
and invalid data. TEST-002, TEST-003, TEST-004, TEST-005, TEST-006."""

import math

import pytest

from proceduralgraph.gates import DISPOSITIONS, Evaluation, ObjectiveGate
from proceduralgraph.objectives import ObjectiveSpec, hoeffding_radius
from proceduralgraph.traces import TaskOutcome

from .test_objectives import context, cost, objective, quality

N = 2000
IDS = tuple(f"t{i}" for i in range(N))


def evaluation(ref, ctx, *, quality_values, cost_values, ids=IDS, drop=(), extra=(), passed=None):
    """One side of a comparison. ``quality_values`` / ``cost_values`` are a constant or a per-task sequence; ``None``
    entries are explicit unknowns."""
    per_task = []
    for i, task_id in enumerate(ids):
        if task_id in drop:
            continue
        q = quality_values if not isinstance(quality_values, list | tuple) else quality_values[i]
        c = cost_values if not isinstance(cost_values, list | tuple) else cost_values[i]
        ok = isinstance(q, int | float) and not isinstance(q, bool) and q >= 0.5
        per_task.append(TaskOutcome(task_id, 1.0 if ok else 0.0, ok if passed is None else passed, metrics={"quality": q, "cost": c}))
    for task_id in extra:
        per_task.append(TaskOutcome(task_id, 1.0, True, metrics={"quality": 0.9, "cost": 3.0}))
    return Evaluation(ref=ref, score=None, per_task=per_task, objective_context=ctx)


def gate_for(mode="lexicographic", metrics=None, **over):
    obj = objective(mode=mode, **({"metrics": metrics} if metrics else {}), **over)
    ctx = context(obj, ids=IDS)
    return ObjectiveGate(obj, ctx), obj, ctx


async def decide(gate, ctx, baseline, candidate):
    b = evaluation("base", ctx, **baseline)
    c = evaluation("cand", ctx, **candidate)
    return await gate.decide(b, c)


# Quality: bounds [0,1], g=0.02, e=0.1, r=0.1. Cost: bounds [0,10], g=0.5, e=1, r=1 (see test_objectives). With N=2000 and
# M=2 intervals, the paired radius is ~0.066 for quality and ~0.66 for cost, so these deltas are decisive:
BETTER_Q, WORSE_Q, MUCH_WORSE_Q = 0.8, 0.6, 0.4  # against a 0.7 baseline: +0.1 improves, -0.1 regresses (lex), -0.3 is material
FEWER_C, MORE_C = 3.5, 7.0  # against a 5.0 baseline: -1.5 improves, +2.0 is a material increase
BASE = dict(quality_values=0.7, cost_values=5.0)


@pytest.mark.parametrize(
    ("candidate", "lex", "pareto", "lex_reason", "pareto_reason"),
    [
        (dict(quality_values=BETTER_Q, cost_values=FEWER_C), "accepted", "accepted", "improvement", "improvement"),
        (dict(quality_values=0.7, cost_values=FEWER_C), "accepted", "accepted", "improvement", "improvement"),
        (dict(quality_values=BETTER_Q, cost_values=5.0), "accepted", "accepted", "improvement", "improvement"),
        (dict(quality_values=BETTER_Q, cost_values=MORE_C), "accepted", "rejected", "improvement", "trade_off"),
        (dict(quality_values=MUCH_WORSE_Q, cost_values=FEWER_C), "rejected", "rejected", "metric_regression", "trade_off"),
        (dict(quality_values=0.7, cost_values=5.0), "equivalent", "equivalent", "all_metrics_equivalent", "all_metrics_equivalent"),
        (dict(quality_values=0.7, cost_values=MORE_C), "rejected", "rejected", "metric_regression", "metric_regression"),
    ],
    ids=["better-q-fewer-c", "equiv-q-fewer-c", "better-q-equiv-c", "better-q-more-c", "worse-q-fewer-c", "all-equivalent", "equiv-q-more-c"],
)
async def test_decision_table(candidate, lex, pareto, lex_reason, pareto_reason):
    """TEST-002: every row of the REQ-014 table in both modes."""
    gate, _, ctx = gate_for("lexicographic")
    decision = await decide(gate, ctx, BASE, candidate)
    assert decision.feedback["disposition"] == lex and lex_reason in decision.feedback["reasons"]
    assert decision.accepted is (lex == "accepted") and decision.stop is False
    gate, _, ctx = gate_for("pareto")
    decision = await decide(gate, ctx, BASE, candidate)
    assert decision.feedback["disposition"] == pareto and pareto_reason in decision.feedback["reasons"]
    assert decision.accepted is (pareto == "accepted")
    assert decision.feedback["disposition"] in DISPOSITIONS


async def test_lexicographic_win_claims_nothing_about_later_metrics_but_shows_them():
    gate, _, ctx = gate_for("lexicographic")
    decision = await decide(gate, ctx, BASE, dict(quality_values=BETTER_Q, cost_values=MORE_C))
    fb = decision.feedback
    assert fb["disposition"] == "accepted" and fb["decisive_metric"] == "quality"
    assert fb["metrics"]["quality"]["verdict"] == "improved" and "verdict" not in fb["metrics"]["cost"] and "reached" not in fb["metrics"]["cost"]
    # the resource increase is still visible in original units and as an oriented interval
    assert fb["metrics"]["cost"]["mean_delta"] == pytest.approx(2.0) and fb["metrics"]["cost"]["oriented_interval"]["high"] < 0
    assert fb["metrics"]["cost"]["baseline_mean"] == pytest.approx(5.0) and fb["metrics"]["cost"]["candidate_mean"] == pytest.approx(7.0)


async def test_pareto_tolerated_loss_stays_visible_and_is_not_an_improvement():
    gate, _, ctx = gate_for("pareto")
    # cost slightly worse (+0.2, within the non-regression margin of 1.0) while quality improves
    decision = await decide(gate, ctx, BASE, dict(quality_values=BETTER_Q, cost_values=5.2))
    assert decision.accepted and decision.feedback["metrics"]["cost"]["verdict"] == "tolerated_loss"
    assert decision.feedback["metrics"]["cost"]["mean_delta"] == pytest.approx(0.2)


async def test_absolute_constraints_pass_violate_unresolved_and_perfect_quality_allows_cost_wins():
    """TEST-003."""
    floor_q = quality(floor=0.9)
    gate, obj, ctx = gate_for("lexicographic", metrics=(floor_q, cost()))
    assert obj.interval_count() == 3  # 2 paired + 1 candidate-mean interval for the floor
    passing = await decide(gate, ctx, dict(quality_values=0.9, cost_values=5.0), dict(quality_values=1.0, cost_values=5.0))
    assert passing.feedback["constraints"]["quality"]["status"] == "pass" and passing.feedback["disposition"] == "accepted"
    violated = await decide(gate, ctx, dict(quality_values=0.95, cost_values=5.0), dict(quality_values=0.8, cost_values=3.0))
    assert violated.feedback["disposition"] == "rejected" and "constraint_violation" in violated.feedback["reasons"]
    assert violated.feedback["constraints"]["quality"]["status"] == "violated" and violated.feedback["decisive_metric"] == "quality"
    borderline = await decide(gate, ctx, dict(quality_values=0.91, cost_values=5.0), dict(quality_values=0.91, cost_values=3.0))
    assert borderline.feedback["disposition"] == "unresolved" and borderline.feedback["constraints"]["quality"]["status"] == "unresolved"
    # quality already perfect on both sides: equivalent on quality, so the cost improvement decides
    perfect = await decide(gate, ctx, dict(quality_values=1.0, cost_values=5.0), dict(quality_values=1.0, cost_values=3.0))
    assert perfect.accepted and perfect.feedback["decisive_metric"] == "cost"
    ceiling_c = cost(ceiling=6.0)
    gate, _, ctx = gate_for("pareto", metrics=(quality(), ceiling_c))
    over = await decide(gate, ctx, dict(quality_values=0.7, cost_values=5.0), dict(quality_values=0.9, cost_values=7.0))
    assert over.feedback["disposition"] == "rejected" and over.feedback["constraints"]["cost"]["status"] == "violated"


async def test_intervals_allocation_and_exact_boundaries():
    """TEST-004: interval arithmetic, complete alpha allocation, strict thresholds, insufficient units, small samples."""
    gate, obj, ctx = gate_for("lexicographic")
    decision = await decide(gate, ctx, BASE, dict(quality_values=BETTER_Q, cost_values=5.0))
    unc = decision.feedback["uncertainty"]
    assert unc == {"method": "paired_hoeffding_v1", "alpha": 0.05, "intervals": 2, "alpha_interval": 0.025, "units": N, "min_units": 2}
    q = decision.feedback["metrics"]["quality"]["oriented_interval"]
    radius = hoeffding_radius(2.0, N, 0.025)
    assert q["mean"] == pytest.approx(0.1) and q["radius"] == pytest.approx(radius) and q["low"] == pytest.approx(0.1 - radius)
    # exact boundary: L == g is not an improvement (strict >), U == 0 is not a regression (strict <)
    exact_g = 0.7 + 0.02 + radius
    decision = await decide(gate, ctx, BASE, dict(quality_values=exact_g - 1e-9, cost_values=5.0))
    assert decision.feedback["disposition"] != "accepted"  # L just below g
    decision = await decide(gate, ctx, BASE, dict(quality_values=exact_g + 1e-6, cost_values=5.0))
    assert decision.feedback["disposition"] == "accepted"  # L just above g
    exact_zero = 0.7 - radius
    decision = await decide(gate, ctx, BASE, dict(quality_values=exact_zero + 1e-9, cost_values=5.0))
    assert decision.feedback["disposition"] != "rejected"  # U just above 0
    decision = await decide(gate, ctx, BASE, dict(quality_values=exact_zero - 1e-6, cost_values=5.0))
    assert decision.feedback["disposition"] == "rejected"  # U just below 0
    # the allocation covers the whole objective even when lexicographic returns on the first metric
    with_floor = ObjectiveSpec(metrics=(quality(floor=0.5), cost(ceiling=9.0)))
    ctx2 = context(with_floor, ids=IDS)
    decision = await ObjectiveGate(with_floor, ctx2).decide(evaluation("b", ctx2, **BASE), evaluation("c", ctx2, quality_values=BETTER_Q, cost_values=5.0))
    assert decision.feedback["uncertainty"]["intervals"] == 4 and decision.feedback["uncertainty"]["alpha_interval"] == pytest.approx(0.0125)
    # too few units
    small_ids = ("a", "b", "c")
    obj_small = objective(min_units=4)
    ctx_small = context(obj_small, ids=small_ids)
    decision = await ObjectiveGate(obj_small, ctx_small).decide(
        evaluation("b", ctx_small, ids=small_ids, quality_values=0.7, cost_values=5.0), evaluation("c", ctx_small, ids=small_ids, quality_values=0.9, cost_values=5.0)
    )
    assert decision.feedback["disposition"] == "unresolved" and "insufficient_units" in decision.feedback["reasons"]
    # identical observed values on a small sample are not proof of equivalence
    obj2 = objective()
    ctx2 = context(obj2, ids=small_ids)
    decision = await ObjectiveGate(obj2, ctx2).decide(
        evaluation("b", ctx2, ids=small_ids, quality_values=0.7, cost_values=5.0), evaluation("c", ctx2, ids=small_ids, quality_values=0.7, cost_values=5.0)
    )
    assert decision.feedback["disposition"] == "unresolved" and decision.feedback["decisive_metric"] == "quality"


async def test_probe_and_construction():
    gate, obj, ctx = gate_for()
    e = evaluation("x", ctx, **BASE)
    probe = await gate.decide(e, e)
    assert probe.accepted is False and probe.stop is False and probe.feedback["disposition"] == "probe"
    other = objective(mode="pareto")
    with pytest.raises(ValueError):
        ObjectiveGate(other, ctx)  # context bound to a different objective


async def test_refusals_context_pairing_and_invalid_values():
    """TEST-005."""
    gate, obj, ctx = gate_for()
    other_ctx = context(obj, ids=IDS, budget_profile="budget-q", evaluator_version="eval-4")
    decision = await gate.decide(evaluation("b", ctx, **BASE), evaluation("c", other_ctx, quality_values=BETTER_Q, cost_values=3.0))
    assert decision.feedback["disposition"] == "invalid" and "context_mismatch" in decision.feedback["reasons"]
    assert decision.feedback["context_mismatch"]["candidate"] == ["evaluator_version", "budget_profile"]
    decision = await gate.decide(evaluation("b", None, **BASE), evaluation("c", ctx, **BASE))
    assert decision.feedback["disposition"] == "invalid" and "missing_context" in decision.feedback["reasons"]
    basis_ctx = context(obj, ids=IDS, measurement_basis={"cost": "billed dollars"})
    decision = await gate.decide(evaluation("b", ctx, **BASE), evaluation("c", basis_ctx, **BASE))
    assert decision.feedback["context_mismatch"]["candidate"] == ["measurement_basis"]
    # pairing
    missing = await gate.decide(evaluation("b", ctx, **BASE), evaluation("c", ctx, quality_values=BETTER_Q, cost_values=3.0, drop={"t7"}))
    assert missing.feedback["disposition"] == "invalid" and "incomplete_pairing" in missing.feedback["reasons"]
    assert missing.feedback["pairing"]["candidate"]["missing"] == ["t7"]
    extra = await gate.decide(evaluation("b", ctx, **BASE), evaluation("c", ctx, quality_values=BETTER_Q, cost_values=3.0, extra=("t9999",)))
    assert extra.feedback["disposition"] == "invalid" and extra.feedback["pairing"]["candidate"]["unexpected"] == ["t9999"]
    dup = evaluation("c", ctx, quality_values=BETTER_Q, cost_values=3.0)
    dup.per_task.append(dup.per_task[0])
    duplicate = await gate.decide(evaluation("b", ctx, **BASE), dup)
    assert duplicate.feedback["disposition"] == "invalid" and "duplicate_task_ids" in duplicate.feedback["reasons"]
    # values
    for bad in (float("nan"), float("inf"), 1.5, -0.1, "0.9"):
        values = [0.9] * N
        values[3] = bad
        decision = await gate.decide(evaluation("b", ctx, **BASE), evaluation("c", ctx, quality_values=values, cost_values=3.0))
        assert decision.feedback["disposition"] == "invalid" and "invalid_value" in decision.feedback["reasons"], bad
        assert decision.feedback["invalid_values"][0]["task_id"] == "t3"
    # deferred evaluation: nothing measured yet
    deferred = Evaluation(ref="c", score=None, per_task=[], objective_context=ctx)
    decision = await gate.decide(evaluation("b", ctx, **BASE), deferred)
    assert decision.feedback["disposition"] == "unmeasured" and "deferred_evaluation" in decision.feedback["reasons"] and not decision.accepted


async def test_unknown_data_rules():
    """TEST-006 and REQ-011: unknown required cost cannot win a cost comparison; failed paid attempts count; unknown
    report-only metrics and unreached lexicographic metrics do not spoil a higher-priority win."""
    gate, obj, ctx = gate_for("lexicographic")
    costs = [3.0] * N
    costs[10] = None  # one unknown cost
    # quality equivalent, cost is the deciding metric but unknown on one unit -> unmeasured
    decision = await decide(gate, ctx, BASE, dict(quality_values=0.7, cost_values=costs))
    assert decision.feedback["disposition"] == "unmeasured" and decision.feedback["decisive_metric"] == "cost"
    assert decision.feedback["unknown_metrics"] == ["cost"] and "cost_mean" not in decision.feedback["metrics"]["cost"]
    # quality improves: accepted on quality; the unknown cost is listed and carries no savings claim
    decision = await decide(gate, ctx, BASE, dict(quality_values=BETTER_Q, cost_values=costs))
    assert decision.accepted and decision.feedback["unknown_metrics"] == ["cost"] and decision.feedback["metrics"]["cost"]["known"] is False
    # pareto: any unknown optimized metric is unmeasured
    pgate, _, pctx = gate_for("pareto")
    decision = await decide(pgate, pctx, BASE, dict(quality_values=BETTER_Q, cost_values=costs))
    assert decision.feedback["disposition"] == "unmeasured" and not decision.accepted
    # failed and no-change paid attempts are in the means: two expensive failures raise the candidate's cost mean
    expensive = [3.0] * N
    expensive[0] = expensive[1] = 10.0
    decision = await decide(gate, ctx, BASE, dict(quality_values=0.7, cost_values=expensive))
    assert decision.feedback["metrics"]["cost"]["candidate_mean"] == pytest.approx((3.0 * (N - 2) + 20.0) / N)
    # a report-only metric may be unknown without blocking
    latency = cost(name="latency", unit="seconds", role="report_only", upper_bound=100.0)
    rgate, robj, rctx = gate_for("lexicographic", metrics=(quality(), latency))
    base = evaluation("b", rctx, **BASE)
    cand = evaluation("c", rctx, quality_values=BETTER_Q, cost_values=5.0)
    for t in base.per_task + cand.per_task:
        t.metrics["latency"] = None
    decision = await rgate.decide(base, cand)
    assert decision.accepted and decision.feedback["unknown_metrics"] == ["latency"]
    assert robj.interval_count() == 1 and decision.feedback["metrics"]["latency"]["role"] == "report_only"
    # a known report-only metric is reported as informational, never decisive
    for t in cand.per_task:
        t.metrics["latency"] = 50.0
    for t in base.per_task:
        t.metrics["latency"] = 20.0
    decision = await rgate.decide(base, cand)
    assert decision.accepted and decision.feedback["metrics"]["latency"]["informational"] is True
    assert decision.feedback["metrics"]["latency"]["mean_delta"] == pytest.approx(30.0) and math.isfinite(decision.feedback["metrics"]["latency"]["oriented_interval"]["low"])


async def test_zero_equivalence_margin_and_unknown_on_a_constrained_metric():
    """REQ-012 documented behaviour: a zero equivalence margin makes `equivalent` unreachable on finite samples, so a
    metric can only be passed by demonstrated improvement. REQ-011: an unknown value on a constrained metric is unmeasured."""
    strict_q = quality(equivalence_margin=0.0)
    gate, _, ctx = gate_for("lexicographic", metrics=(strict_q, cost()))
    same = await decide(gate, ctx, BASE, BASE)
    assert same.feedback["disposition"] == "unresolved" and same.feedback["decisive_metric"] == "quality"
    better = await decide(gate, ctx, BASE, dict(quality_values=BETTER_Q, cost_values=5.0))
    assert better.accepted
    floor_q = quality(floor=0.5)
    gate, _, ctx = gate_for("lexicographic", metrics=(floor_q, cost()))
    values = [0.9] * N
    values[0] = None
    decision = await decide(gate, ctx, BASE, dict(quality_values=values, cost_values=3.0))
    assert decision.feedback["disposition"] == "unmeasured" and "constraint_unmeasured" in decision.feedback["reasons"]
    assert decision.feedback["constraints"]["quality"] == {"status": "unmeasured"}


def test_context_is_hashable_picklable_and_copyable():
    import copy
    import pickle

    obj = objective()
    ctx = context(obj)
    assert hash(ctx) == hash(context(obj)) and pickle.loads(pickle.dumps(ctx)) == ctx and copy.deepcopy(ctx) == ctx
    assert ctx.basis_for("cost") == "provider calls" and ctx.basis_for("quality") is None
    assert hash(obj) == hash(objective()) and pickle.loads(pickle.dumps(obj)) == obj


def test_render_decision_tolerates_partial_host_feedback():
    from proceduralgraph.rejections import render_decision

    lines = render_decision({"disposition": "unmeasured", "reasons": ["pending"], "metrics": {"x": {"known": True}, "y": {"known": False, "role": "report_only"}}})
    assert lines[0] == "Decision: unmeasured (reasons: pending)"
    assert lines[1].startswith("- x [] : baseline ? → candidate ? (oriented ? .. ?") and "report only" in lines[2]
