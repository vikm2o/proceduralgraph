# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Quality and cost objectives: what a candidate graph must improve, by how much, and on what evidence.

The paper gates on one scalar validation score (eq. 5). A production host usually wants more: keep quality at least
as good while lowering measured resource use, or trade them off under explicit rules. This module holds the
**contracts** a host declares once per evolution run, immutable and content-addressed:

- :class:`MetricSpec`: one measured quantity, its unit, direction, finite observation bounds, the thresholds that
  decide (minimum meaningful improvement, equivalence margin, non-regression margin), optional absolute floor and
  ceiling on the candidate mean, and whether it decides acceptance (``optimize``) or is only reported.
- :class:`ObjectiveSpec`: the ordered metrics, the acceptance mode (``lexicographic`` or ``pareto``), the uncertainty
  method, the comparison error budget ``alpha`` and the minimum number of independent paired units.
- :class:`ObjectiveContext`: what the evidence was measured on, so a baseline and a candidate are only ever compared
  when they share it: the objective digest, the host's evaluation-design identity, evaluator version, cohort,
  the immutable expected task ids, the resource-budget profile, the evidence partition and each metric's
  measurement basis.

Nothing here names a vendor, a currency or a conversion weight: names, units and bases are the host's. The gate that
applies these is :class:`proceduralgraph.gates.ObjectiveGate`; the interval arithmetic is :func:`paired_hoeffding_v1`.
This module imports only ``documents`` so ``config`` and ``gates`` can import it without a cycle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .documents import digest

OBJECTIVE_SCHEMA_VERSION = 1
DIRECTIONS = ("maximize", "minimize")
ROLES = ("optimize", "report_only")
MODES = ("lexicographic", "pareto")
UNCERTAINTY_METHODS = ("paired_hoeffding_v1",)


class ObjectiveError(ValueError):
    """An objective, metric or context that fails its own contract."""


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ObjectiveError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _nonnegative(value: Any, what: str) -> float:
    number = _finite(value, what)
    if number < 0:
        raise ObjectiveError(f"{what} must be >= 0, got {number}")
    return number


@dataclass(frozen=True)
class MetricSpec:
    """One measured quantity. Thresholds are in the metric's own unit. ``lower_bound`` / ``upper_bound`` bound every
    observation (a value outside them is invalid evidence, never clipped); ``floor`` / ``ceiling`` bound the candidate
    *mean* and are acceptance constraints, so ``report_only`` metrics may not carry them (REQ-002)."""

    name: str
    unit: str
    direction: str
    lower_bound: float
    upper_bound: float
    min_improvement: float = 0.0
    equivalence_margin: float = 0.0
    non_regression_margin: float = 0.0
    floor: float | None = None
    ceiling: float | None = None
    role: str = "optimize"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ObjectiveError("metric name must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ObjectiveError(f"metric {self.name!r} needs a unit")
        if self.direction not in DIRECTIONS:
            raise ObjectiveError(f"metric {self.name!r}: direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.role not in ROLES:
            raise ObjectiveError(f"metric {self.name!r}: role must be one of {ROLES}, got {self.role!r}")
        lower = _finite(self.lower_bound, f"metric {self.name!r} lower_bound")
        upper = _finite(self.upper_bound, f"metric {self.name!r} upper_bound")
        if not upper > lower:
            raise ObjectiveError(f"metric {self.name!r}: upper_bound must exceed lower_bound ({lower} .. {upper})")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        for attr in ("min_improvement", "equivalence_margin", "non_regression_margin"):
            object.__setattr__(self, attr, _nonnegative(getattr(self, attr), f"metric {self.name!r} {attr}"))
        for attr in ("floor", "ceiling"):
            value = getattr(self, attr)
            if value is not None:
                number = _finite(value, f"metric {self.name!r} {attr}")
                if not lower <= number <= upper:
                    raise ObjectiveError(f"metric {self.name!r}: {attr} {number} lies outside the observation bounds [{lower}, {upper}]")
                object.__setattr__(self, attr, number)
        if self.floor is not None and self.ceiling is not None and self.floor > self.ceiling:
            raise ObjectiveError(f"metric {self.name!r}: floor {self.floor} exceeds ceiling {self.ceiling}")
        if self.role == "report_only" and (self.floor is not None or self.ceiling is not None):
            raise ObjectiveError(f"metric {self.name!r}: a report_only metric cannot carry a floor or ceiling")

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def optimized(self) -> bool:
        return self.role == "optimize"

    @property
    def has_constraint(self) -> bool:
        return self.floor is not None or self.ceiling is not None

    def oriented(self, candidate: float, baseline: float) -> float:
        """The paired difference with positive meaning improvement (REQ-007)."""
        return candidate - baseline if self.direction == "maximize" else baseline - candidate

    def in_bounds(self, value: float) -> bool:
        return self.lower_bound <= value <= self.upper_bound

    def to_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "min_improvement": self.min_improvement,
            "equivalence_margin": self.equivalence_margin,
            "non_regression_margin": self.non_regression_margin,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "role": self.role,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> MetricSpec:
        if not isinstance(value, Mapping):
            raise ObjectiveError("a metric document must be a mapping")
        try:
            return cls(
                name=value["name"],
                unit=value["unit"],
                direction=value["direction"],
                lower_bound=value["lower_bound"],
                upper_bound=value["upper_bound"],
                min_improvement=value.get("min_improvement", 0.0),
                equivalence_margin=value.get("equivalence_margin", 0.0),
                non_regression_margin=value.get("non_regression_margin", 0.0),
                floor=value.get("floor"),
                ceiling=value.get("ceiling"),
                role=value.get("role", "optimize"),
            )
        except KeyError as exc:
            raise ObjectiveError(f"metric document is missing {exc}") from exc


@dataclass(frozen=True)
class ObjectiveSpec:
    """The ordered metrics and the acceptance rule. Metric order is the priority in ``lexicographic`` mode and the
    reporting order in both modes (REQ-003). Immutable for a run; ``digest`` is its identity."""

    metrics: tuple[MetricSpec, ...]
    mode: str = "lexicographic"
    uncertainty: str = "paired_hoeffding_v1"
    alpha: float = 0.05
    min_units: int = 2
    schema: int = OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics or not all(isinstance(m, MetricSpec) for m in metrics):
            raise ObjectiveError("an objective needs at least one MetricSpec")
        names = [m.name for m in metrics]
        if len(set(names)) != len(names):
            raise ObjectiveError(f"metric names must be unique, got {names}")
        if not any(m.optimized for m in metrics):
            raise ObjectiveError("an objective needs at least one metric with role 'optimize'")
        if self.mode not in MODES:
            raise ObjectiveError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.uncertainty not in UNCERTAINTY_METHODS:
            raise ObjectiveError(f"uncertainty must be one of {UNCERTAINTY_METHODS}, got {self.uncertainty!r}")
        alpha = _finite(self.alpha, "alpha")
        if not 0 < alpha < 1:
            raise ObjectiveError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")
        if isinstance(self.min_units, bool) or not isinstance(self.min_units, int) or self.min_units < 2:
            raise ObjectiveError(f"min_units must be an integer >= 2, got {self.min_units!r}")
        if self.schema != OBJECTIVE_SCHEMA_VERSION:
            raise ObjectiveError(f"objective schema {self.schema} is not supported (this package writes {OBJECTIVE_SCHEMA_VERSION})")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "alpha", alpha)

    @property
    def optimized(self) -> tuple[MetricSpec, ...]:
        return tuple(m for m in self.metrics if m.optimized)

    @property
    def report_only(self) -> tuple[MetricSpec, ...]:
        return tuple(m for m in self.metrics if not m.optimized)

    def metric(self, name: str) -> MetricSpec:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)

    def interval_count(self) -> int:
        """``M`` (REQ-012): one paired interval per optimized metric plus one candidate-mean interval per optimized
        metric that carries a floor or ceiling. Fixed for the whole objective, however early a decision returns."""
        return sum(1 for m in self.optimized) + sum(1 for m in self.optimized if m.has_constraint)

    @property
    def alpha_interval(self) -> float:
        return self.alpha / self.interval_count()

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metrics": [m.to_document() for m in self.metrics],
            "mode": self.mode,
            "uncertainty": self.uncertainty,
            "alpha": self.alpha,
            "min_units": self.min_units,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> ObjectiveSpec:
        if not isinstance(value, Mapping):
            raise ObjectiveError("an objective document must be a mapping")
        return cls(
            metrics=tuple(MetricSpec.from_document(m) for m in value.get("metrics", [])),
            mode=value.get("mode", "lexicographic"),
            uncertainty=value.get("uncertainty", "paired_hoeffding_v1"),
            alpha=value.get("alpha", 0.05),
            min_units=value.get("min_units", 2),
            schema=value.get("schema", OBJECTIVE_SCHEMA_VERSION),
        )

    @property
    def digest(self) -> str:
        return digest(self.to_document())

    def summary(self) -> str:
        """One line per metric for prompts and reports; original units, no numbers invented."""
        lines = [f"objective {self.digest[:12]}: mode {self.mode}, uncertainty {self.uncertainty}, alpha {self.alpha:g}, "
                 f"min {self.min_units} paired units, {self.interval_count()} interval(s) share alpha"]
        for i, m in enumerate(self.metrics, start=1):
            parts = [f"{i}. {m.name} [{m.unit}] {m.direction}", f"role {m.role}", f"bounds [{m.lower_bound:g}, {m.upper_bound:g}]"]
            if m.optimized:
                parts.append(f"min improvement {m.min_improvement:g}, equivalence ±{m.equivalence_margin:g}, non-regression {m.non_regression_margin:g}")
            if m.floor is not None:
                parts.append(f"mean floor {m.floor:g}")
            if m.ceiling is not None:
                parts.append(f"mean ceiling {m.ceiling:g}")
            lines.append("  " + "; ".join(parts))
        return "\n".join(lines)


@dataclass(frozen=True)
class ObjectiveContext:
    """What the evidence was measured on (REQ-004). Two evaluations are comparable only when their contexts are
    identical; the gate refuses otherwise, before any acceptance."""

    objective_digest: str
    evaluation_design: str
    evaluator_version: str
    cohort: str
    expected_task_ids: tuple[str, ...]
    budget_profile: str
    evidence_partition: str
    measurement_basis: Mapping[str, str] | tuple[tuple[str, str], ...] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr in ("objective_digest", "evaluation_design", "evaluator_version", "cohort", "budget_profile", "evidence_partition"):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise ObjectiveError(f"ObjectiveContext.{attr} must be a non-empty string")
        ids = tuple(str(t) for t in self.expected_task_ids)
        if not ids:
            raise ObjectiveError("expected_task_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ObjectiveError("expected_task_ids must be unique: one id is one independent analysis unit")
        object.__setattr__(self, "expected_task_ids", ids)
        basis = dict(self.measurement_basis)
        for k, v in basis.items():
            if not isinstance(k, str) or not isinstance(v, str) or not v.strip():
                raise ObjectiveError("measurement_basis maps metric names to non-empty basis strings")
        # stored as sorted pairs so the frozen context stays hashable, picklable and deep-copyable
        object.__setattr__(self, "measurement_basis", tuple(sorted(basis.items())))

    def basis_for(self, metric: str) -> str | None:
        return dict(self.measurement_basis).get(metric)


    @property
    def units(self) -> int:
        return len(self.expected_task_ids)

    def to_document(self) -> dict[str, Any]:
        return {
            "objective_digest": self.objective_digest,
            "evaluation_design": self.evaluation_design,
            "evaluator_version": self.evaluator_version,
            "cohort": self.cohort,
            "expected_task_ids": list(self.expected_task_ids),
            "budget_profile": self.budget_profile,
            "evidence_partition": self.evidence_partition,
            "measurement_basis": dict(self.measurement_basis),
        }  # a dict from the stored pairs

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> ObjectiveContext:
        if not isinstance(value, Mapping):
            raise ObjectiveError("an objective context document must be a mapping")
        try:
            return cls(
                objective_digest=value["objective_digest"],
                evaluation_design=value["evaluation_design"],
                evaluator_version=value["evaluator_version"],
                cohort=value["cohort"],
                expected_task_ids=tuple(value["expected_task_ids"]),
                budget_profile=value["budget_profile"],
                evidence_partition=value["evidence_partition"],
                measurement_basis=dict(value.get("measurement_basis", {})),
            )
        except KeyError as exc:
            raise ObjectiveError(f"objective context document is missing {exc}") from exc

    @property
    def digest(self) -> str:
        return digest(self.to_document())

    def mismatches(self, other: ObjectiveContext | None) -> list[str]:
        """Field names that differ, for explicit refusal reasons. Empty when the contexts are identical."""
        if other is None:
            return ["missing_context"]
        mine, theirs = self.to_document(), other.to_document()
        return [key for key in mine if mine[key] != theirs[key]]

    def bound_to(self, objective: ObjectiveSpec) -> bool:
        return self.objective_digest == objective.digest


@dataclass(frozen=True)
class Interval:
    """A closed interval with the point estimate it surrounds and the radius that produced it."""

    low: float
    high: float
    mean: float
    radius: float

    def within(self, low: float, high: float) -> bool:
        return self.low >= low and self.high <= high

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high, "mean": self.mean, "radius": self.radius}


def hoeffding_radius(width: float, n: int, alpha_interval: float) -> float:
    """``width * sqrt(log(2 / alpha_interval) / (2 n))`` (REQ-012): the two-sided Hoeffding half-width for the mean of
    ``n`` independent observations each confined to an interval of length ``width``."""
    if n <= 0:
        raise ObjectiveError("n must be positive")
    if not 0 < alpha_interval < 1:
        raise ObjectiveError("alpha_interval must satisfy 0 < alpha < 1")
    return width * math.sqrt(math.log(2.0 / alpha_interval) / (2.0 * n))


def paired_hoeffding_v1(values: Sequence[float], *, width: float, alpha_interval: float) -> Interval:
    """The interval for the mean of ``values``, each confined to an interval of length ``width``. For paired
    differences of observations bounded in ``[a, b]`` pass ``width = 2 (b - a)``; for a plain mean pass ``b - a``."""
    if not values:
        raise ObjectiveError("paired_hoeffding_v1 needs at least one value")
    mean = math.fsum(values) / len(values)
    radius = hoeffding_radius(width, len(values), alpha_interval)
    return Interval(low=mean - radius, high=mean + radius, mean=mean, radius=radius)


__all__ = [
    "DIRECTIONS",
    "MODES",
    "OBJECTIVE_SCHEMA_VERSION",
    "ROLES",
    "UNCERTAINTY_METHODS",
    "Interval",
    "MetricSpec",
    "ObjectiveContext",
    "ObjectiveError",
    "ObjectiveSpec",
    "hoeffding_radius",
    "paired_hoeffding_v1",
]
