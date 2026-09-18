# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Objective contracts (REQ-001 to REQ-004) and the uncertainty method (REQ-012). TEST-001, part of TEST-004."""

import math

import pytest

from proceduralgraph.objectives import (
    Interval,
    MetricSpec,
    ObjectiveContext,
    ObjectiveError,
    ObjectiveSpec,
    hoeffding_radius,
    paired_hoeffding_v1,
)


def quality(**over):
    base = dict(name="quality", unit="fraction", direction="maximize", lower_bound=0.0, upper_bound=1.0,
                min_improvement=0.02, equivalence_margin=0.1, non_regression_margin=0.1)
    return MetricSpec(**{**base, **over})


def cost(**over):
    base = dict(name="cost", unit="calls", direction="minimize", lower_bound=0.0, upper_bound=10.0,
                min_improvement=0.5, equivalence_margin=1.0, non_regression_margin=1.0)
    return MetricSpec(**{**base, **over})


def objective(**over):
    return ObjectiveSpec(**{"metrics": (quality(), cost()), **over})


def context(obj, ids=("t1", "t2", "t3"), **over):
    base = dict(objective_digest=obj.digest, evaluation_design="design-v1", evaluator_version="eval-3", cohort="cohort-a",
                expected_task_ids=tuple(ids), budget_profile="budget-p", evidence_partition="validation-2026-09",
                measurement_basis={"cost": "provider calls"})
    return ObjectiveContext(**{**base, **over})


def test_metric_validation():
    m = quality(floor=0.9)
    assert m.width == 1.0 and m.optimized and m.has_constraint and m.oriented(0.8, 0.5) == pytest.approx(0.3)
    assert cost().oriented(3.0, 5.0) == pytest.approx(2.0)  # minimize: baseline - candidate, positive is better
    for bad in (
        dict(name=""),
        dict(unit=""),
        dict(direction="up"),
        dict(role="decorative"),
        dict(lower_bound=1.0, upper_bound=1.0),
        dict(lower_bound=float("nan")),
        dict(upper_bound=float("inf")),
        dict(min_improvement=-0.1),
        dict(equivalence_margin=float("inf")),
        dict(floor=1.5),
        dict(floor=0.9, ceiling=0.8),
        dict(role="report_only", floor=0.5),
    ):
        with pytest.raises(ObjectiveError):
            quality(**bad)


def test_objective_validation_and_interval_count():
    obj = objective()
    assert obj.interval_count() == 2 and obj.alpha_interval == pytest.approx(0.025)
    with_floor = objective(metrics=(quality(floor=0.9), cost(ceiling=8.0)))
    assert with_floor.interval_count() == 4
    with_report = objective(metrics=(quality(), cost(role="report_only")))
    assert with_report.interval_count() == 1 and [m.name for m in with_report.optimized] == ["quality"]
    for bad in (
        dict(metrics=()),
        dict(metrics=(quality(), quality())),
        dict(metrics=(cost(role="report_only"),)),
        dict(mode="weighted"),
        dict(uncertainty="bootstrap"),
        dict(alpha=0.0),
        dict(alpha=1.0),
        dict(min_units=1),
        dict(min_units=2.5),
        dict(schema=2),
    ):
        with pytest.raises(ObjectiveError):
            objective(**bad)


def test_documents_round_trip_and_digests_are_semantic():
    obj = objective(mode="pareto")
    again = ObjectiveSpec.from_document(obj.to_document())
    assert again == obj and again.digest == obj.digest
    assert objective(mode="lexicographic").digest != obj.digest
    assert objective(metrics=(cost(), quality()), mode="pareto").digest != obj.digest  # order is semantic
    assert objective(metrics=(quality(min_improvement=0.03), cost()), mode="pareto").digest != obj.digest
    assert objective(mode="pareto", alpha=0.05).digest == obj.digest  # same value, same digest
    ctx = context(obj)
    ctx_again = ObjectiveContext.from_document(ctx.to_document())
    assert ctx_again == ctx and ctx_again.digest == ctx.digest and ctx.units == 3 and ctx.bound_to(obj)
    assert context(obj, cohort="cohort-b").digest != ctx.digest
    assert context(obj, ids=("t1", "t2", "t3", "t4")).digest != ctx.digest
    assert context(obj, measurement_basis={"cost": "billed dollars"}).digest != ctx.digest
    assert ctx.mismatches(context(obj, cohort="cohort-b", evaluator_version="eval-4")) == ["evaluator_version", "cohort"]
    assert ctx.mismatches(None) == ["missing_context"] and ctx.mismatches(ctx_again) == []
    for bad in (dict(cohort=""), dict(ids=()), dict(ids=("t1", "t1")), dict(measurement_basis={"cost": ""})):
        with pytest.raises(ObjectiveError):
            context(obj, **bad)
    with pytest.raises(ObjectiveError):
        MetricSpec.from_document({"name": "x"})
    assert "objective " in obj.summary() and "1. quality [fraction] maximize" in obj.summary()


def test_hoeffding_radius_and_paired_interval():
    # radius = width * sqrt(log(2 / alpha) / (2 n))
    assert hoeffding_radius(1.0, 200, 0.05) == pytest.approx(math.sqrt(math.log(40) / 400))
    assert hoeffding_radius(2.0, 200, 0.05) == pytest.approx(2 * hoeffding_radius(1.0, 200, 0.05))
    assert hoeffding_radius(1.0, 800, 0.05) == pytest.approx(hoeffding_radius(1.0, 200, 0.05) / 2)
    interval = paired_hoeffding_v1([0.1, 0.3, 0.2, 0.2], width=2.0, alpha_interval=0.05)
    assert interval.mean == pytest.approx(0.2) and interval.radius == pytest.approx(hoeffding_radius(2.0, 4, 0.05))
    assert interval.low == pytest.approx(0.2 - interval.radius) and interval.high == pytest.approx(0.2 + interval.radius)
    assert Interval(0.0, 0.1, 0.05, 0.05).within(-0.1, 0.1) and not Interval(0.0, 0.2, 0.1, 0.1).within(-0.1, 0.1)
    for bad in (dict(width=1.0, alpha_interval=0.0), dict(width=1.0, alpha_interval=1.0)):
        with pytest.raises(ObjectiveError):
            paired_hoeffding_v1([0.1], **bad)
    with pytest.raises(ObjectiveError):
        paired_hoeffding_v1([], width=1.0, alpha_interval=0.05)
    with pytest.raises(ObjectiveError):
        hoeffding_radius(1.0, 0, 0.05)
