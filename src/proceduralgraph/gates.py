# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Validation evaluation and the gating decision (paper §3.3 Step 3, eq. 4 and 5). REQUIREMENTS G5.

The harness never scores anything itself. An :class:`Evaluator` runs a graph on the validation split and returns an
:class:`Evaluation`; a :class:`Gate` compares the candidate with the retained graph's cached score and decides. The
default gate is the paper's rule, ``>=`` with ties accepted (Alg. 1 line 16). :class:`StrictImprovementGate` is
skillwiki's rule; :class:`PairedGate` is the production option that needs per-task outcomes.

The perfect-score early stop lives on the gate (``perfect``), as in skillwiki: the harness probes ``decide(best, best)``
after each acceptance and stops with ``stopped_reason="perfect"`` when the gate says so. The default gate's
``perfect=None`` keeps Alg. 1's run-all-K behaviour.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .objectives import Interval, MetricSpec, ObjectiveContext, ObjectiveSpec, paired_hoeffding_v1
from .traces import TaskOutcome

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Graph


@dataclass
class Evaluation:
    """A validation result. ``objective_context`` (optional, REQ-005) binds the evidence to an
    :class:`~proceduralgraph.objectives.ObjectiveContext`; it is omitted from documents when absent (REQ-018)."""

    ref: str | None  # host-opaque identity of the evaluated graph (digest, bundle id, ...)
    score: float | None  # S_val(G); None when the host's gate does not use a scalar, or the evaluation is deferred
    aggregate: dict[str, Any] = field(default_factory=dict)
    per_task: list[TaskOutcome] = field(default_factory=list)
    objective_context: ObjectiveContext | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"ref": self.ref, "score": self.score, "aggregate": self.aggregate, "tasks": len(self.per_task)}
        if self.objective_context is not None:
            value["objective_context"] = self.objective_context.digest
        return value

    def to_document(self) -> dict[str, Any]:
        """Full form, including per-task outcomes, for resume checkpoints and rejection memory."""
        value: dict[str, Any] = {"ref": self.ref, "score": self.score, "aggregate": self.aggregate, "per_task": [t.to_dict() for t in self.per_task]}
        if self.objective_context is not None:
            value["objective_context"] = self.objective_context.to_document()
        return value

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Evaluation:
        context = value.get("objective_context")
        return cls(
            ref=value.get("ref"),
            score=value.get("score"),
            aggregate=dict(value.get("aggregate", {})),
            per_task=[TaskOutcome.from_dict(t) for t in value.get("per_task", [])],
            objective_context=ObjectiveContext.from_document(context) if context else None,
        )


@dataclass
class Decision:
    accepted: bool
    feedback: dict[str, Any] = field(default_factory=dict)
    stop: bool = False  # e.g. perfect validation score reached


class Evaluator(Protocol):
    async def evaluate(self, ref: str | None, graph: Graph, *, iteration: int, purpose: str) -> Evaluation:
        """``purpose`` is ``"baseline"`` for S_0 / the loaded head and ``"candidate"`` for G_k^cand."""
        ...


class Gate(Protocol):
    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        """Called with ``best is candidate`` as a perfect-score probe: answer ``accepted=False`` and set ``stop`` from
        the score alone."""
        ...


def _probe(best: Evaluation, candidate: Evaluation, perfect: float | None) -> Decision | None:
    if best is candidate:
        stop = perfect is not None and best.score is not None and best.score >= perfect
        return Decision(accepted=False, feedback={}, stop=stop)
    return None


class _ScalarGate:
    """Shared shape of the two scalar gates: probe handling, the scalar check, feedback and the perfect-score stop.
    Subclasses define :meth:`accepts`."""

    def __init__(self, *, perfect: float | None):
        self.perfect = perfect

    def accepts(self, candidate: float, best: float) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        probe = _probe(best, candidate, self.perfect)
        if probe is not None:
            return probe
        if best.score is None or candidate.score is None:
            raise ValueError(f"{type(self).__name__} needs scalar scores; supply a custom Gate otherwise")
        accepted = self.accepts(candidate.score, best.score)
        new_best = candidate.score if accepted else best.score
        stop = self.perfect is not None and new_best >= self.perfect
        return Decision(
            accepted=accepted,
            feedback={"best_score": round(best.score, 4), "candidate_score": round(candidate.score, 4), "accepted": accepted},
            stop=stop,
        )


class TieAcceptingGate(_ScalarGate):
    """The paper's gate (eq. 5, Alg. 1 line 16): accept iff S_val(cand) >= S_val(retained). Ties are accepted.
    ``perfect=None`` (default) never stops early, as Alg. 1 runs all K rounds."""

    def __init__(self, *, perfect: float | None = None):
        super().__init__(perfect=perfect)

    def accepts(self, candidate: float, best: float) -> bool:
        return candidate >= best


class StrictImprovementGate(_ScalarGate):
    """Accept iff the candidate strictly improves (skillwiki's rule); stop when the score reaches ``perfect``."""

    def __init__(self, *, perfect: float | None = 1.0, minimum_delta: float = 0.0):
        super().__init__(perfect=perfect)
        self.minimum_delta = minimum_delta

    def accepts(self, candidate: float, best: float) -> bool:
        return candidate > best + self.minimum_delta


class PairedGate:
    """Accept only when a paired bootstrap over per-task scores says the candidate is very unlikely to be worse.

    Pairs the two evaluations by ``task_id``, resamples the paired differences ``resamples`` times and accepts iff the
    mean difference is >= 0 and the fraction of resamples with a non-negative mean is at least ``min_win_probability``
    (``>=`` on both, matching the paper's tie rule). Refuses to decide on fewer than ``min_tasks`` shared tasks and
    raises when either side lacks ``per_task`` outcomes.
    """

    def __init__(
        self,
        *,
        min_win_probability: float = 0.9,
        min_tasks: int = 20,
        resamples: int = 2000,
        seed: int = 17,
        perfect: float | None = 1.0,
    ):
        if not 0.5 <= min_win_probability <= 1.0:
            raise ValueError("min_win_probability must be in [0.5, 1.0]")
        self.min_win_probability, self.min_tasks, self.resamples, self.seed, self.perfect = (
            min_win_probability, min_tasks, resamples, seed, perfect,
        )

    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        probe = _probe(best, candidate, self.perfect)
        if probe is not None:
            return probe
        if not best.per_task or not candidate.per_task:
            raise ValueError("PairedGate needs per_task outcomes on both evaluations, paired by task_id")
        before = {t.task_id: t.score for t in best.per_task}
        after = {t.task_id: t.score for t in candidate.per_task}
        shared = sorted(set(before) & set(after))
        if len(shared) < self.min_tasks:
            return Decision(
                accepted=False,
                feedback={
                    "accepted": False,
                    "reason": "insufficient_paired_tasks",
                    "paired_tasks": len(shared),
                    "min_tasks": self.min_tasks,
                },
            )
        deltas = [after[t] - before[t] for t in shared]
        mean_delta = statistics.fmean(deltas)
        rng = random.Random(f"{self.seed}:{best.ref}:{candidate.ref}")
        n = len(deltas)
        wins, means = 0, []
        for _ in range(self.resamples):
            total = sum(deltas[rng.randrange(n)] for _ in range(n))
            means.append(total / n)
            wins += total >= 0
        means.sort()
        win_probability = wins / self.resamples
        low, high = means[int(0.025 * (self.resamples - 1))], means[int(0.975 * (self.resamples - 1))]
        accepted = mean_delta >= 0 and win_probability >= self.min_win_probability
        new_best = candidate.score if accepted else best.score
        stop = self.perfect is not None and new_best is not None and new_best >= self.perfect
        return Decision(
            accepted=accepted,
            feedback={
                "accepted": accepted,
                "paired_tasks": n,
                "mean_delta": round(mean_delta, 4),
                "delta_95_ci": [round(low, 4), round(high, 4)],
                "win_probability": round(win_probability, 3),
                "min_win_probability": self.min_win_probability,
                "best_score": None if best.score is None else round(best.score, 4),
                "candidate_score": None if candidate.score is None else round(candidate.score, 4),
            },
            stop=stop,
        )


DISPOSITIONS = ("accepted", "rejected", "equivalent", "unresolved", "unmeasured", "invalid")
PROBE_DISPOSITION = "probe"  # the identity probe's answer; never persisted as a round's decision
DECISION_SCHEMA_VERSION = 1


class ObjectiveGate:
    """Multi-metric acceptance over paired, bounded evidence (REQ-007 to REQ-014). Opt-in; the paper's scalar
    ``TieAcceptingGate`` stays the default.

    Every decision is deterministic, needs no model call, and returns a structured comparison in
    ``Decision.feedback``: schema version, disposition, stable reason codes, objective and context digests, the two
    refs, coverage counts, per-metric original-unit means and oriented intervals with their thresholds, unknown
    metrics, the decisive metric and the absolute-constraint results. Every disposition other than ``accepted`` maps
    to ``Decision.accepted=False``; the outer loop records it as ``rejected`` with the disposition alongside.

    Check order: context (both evaluations carry the configured context and objective), pairing (unique ids on each
    side, exactly the expected set; empty ``per_task`` is a deferred evaluation and ``unmeasured``), value validity,
    the minimum unit count, absolute constraints on candidate means, then the mode rule. The identity probe
    ``decide(best, best)`` answers ``accepted=False, stop=False``: a perfect quality score alone never ends the run.

    The uncertainty guarantee is for one fixed candidate and a predeclared paired comparison on independent units
    (REQ-013). Reusing the same validation tasks across many proposals is a search, not a new guarantee; hosts own
    fresh qualification evidence and any sequential design. Acceptance here selects a development graph only.
    """

    def __init__(self, objective: ObjectiveSpec, context: ObjectiveContext):
        if not context.bound_to(objective):
            raise ValueError("the ObjectiveContext is bound to a different objective digest than the ObjectiveSpec given")
        self.objective, self.context = objective, context

    # -- the Gate protocol ----------------------------------------------------------------------------------------------

    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        if best is candidate:  # the harness's perfect-score probe: cost optimisation continues past perfect quality (REQ-014)
            return Decision(accepted=False, feedback={"schema": DECISION_SCHEMA_VERSION, "disposition": PROBE_DISPOSITION}, stop=False)
        feedback = self.compare(best, candidate)
        return Decision(accepted=feedback["disposition"] == "accepted", feedback=feedback, stop=False)

    # -- the comparison, usable without the protocol ----------------------------------------------------------------------

    def compare(self, baseline: Evaluation, candidate: Evaluation) -> dict[str, Any]:
        objective = self.objective
        report: dict[str, Any] = {
            "schema": DECISION_SCHEMA_VERSION,
            "disposition": "unresolved",
            "reasons": [],
            "objective_digest": objective.digest,
            "context_digest": self.context.digest,
            "mode": objective.mode,
            "baseline_ref": baseline.ref,
            "candidate_ref": candidate.ref,
            "coverage": {"expected": self.context.units, "baseline": len(baseline.per_task), "candidate": len(candidate.per_task)},
            "uncertainty": {
                "method": objective.uncertainty,
                "alpha": objective.alpha,
                "intervals": objective.interval_count(),
                "alpha_interval": objective.alpha_interval,
                "units": self.context.units,
                "min_units": objective.min_units,
            },
            "metrics": {},
            "unknown_metrics": [],
            "decisive_metric": None,
            "constraints": {},
        }

        def finish(disposition: str, *reasons: str, decisive: str | None = None) -> dict[str, Any]:
            report["disposition"] = disposition
            report["reasons"] = list(dict.fromkeys([*report["reasons"], *reasons]))
            if decisive is not None:
                report["decisive_metric"] = decisive
            return report

        # 1. context
        for side, evaluation in (("baseline", baseline), ("candidate", candidate)):
            mismatches = self.context.mismatches(evaluation.objective_context)
            if mismatches:
                report.setdefault("context_mismatch", {})[side] = mismatches
                return finish("invalid", "context_mismatch" if evaluation.objective_context is not None else "missing_context")
        # 2. pairing
        if not baseline.per_task or not candidate.per_task:
            return finish("unmeasured", "deferred_evaluation")
        expected = set(self.context.expected_task_ids)
        by_side: dict[str, dict[str, TaskOutcome]] = {}
        for side, evaluation in (("baseline", baseline), ("candidate", candidate)):
            ids = [t.task_id for t in evaluation.per_task]
            if len(set(ids)) != len(ids):
                return finish("invalid", "duplicate_task_ids")
            if set(ids) != expected:
                report.setdefault("pairing", {})[side] = {"missing": sorted(expected - set(ids)), "unexpected": sorted(set(ids) - expected)}
                return finish("invalid", "incomplete_pairing")
            by_side[side] = {t.task_id: t for t in evaluation.per_task}
        # 3. observations per metric: (baseline values, candidate values) in expected order, or None when any is unknown
        observed: dict[str, tuple[list[float], list[float]] | None] = {}
        for metric in objective.metrics:
            pair = self._observations(metric, by_side["baseline"], by_side["candidate"], report)
            if pair is False:
                return finish("invalid", "invalid_value", decisive=metric.name)
            observed[metric.name] = pair
            if pair is None:
                report["unknown_metrics"].append(metric.name)
        # 4. enough units
        n = self.context.units
        if n < objective.min_units:
            return finish("unresolved", "insufficient_units")
        alpha_interval = objective.alpha_interval
        for metric in objective.metrics:
            pair = observed[metric.name]
            entry: dict[str, Any] = {
                "role": metric.role,
                "direction": metric.direction,
                "unit": metric.unit,
                "thresholds": {"min_improvement": metric.min_improvement, "equivalence_margin": metric.equivalence_margin,
                               "non_regression_margin": metric.non_regression_margin},
                "known": pair is not None,
            }
            if pair is not None:
                base_values, cand_values = pair
                deltas = [metric.oriented(c, b) for b, c in zip(base_values, cand_values, strict=True)]
                interval = paired_hoeffding_v1(deltas, width=2 * metric.width, alpha_interval=alpha_interval)
                entry["baseline_mean"] = math.fsum(base_values) / n
                entry["candidate_mean"] = math.fsum(cand_values) / n
                entry["mean_delta"] = entry["candidate_mean"] - entry["baseline_mean"]
                entry["oriented_interval"] = interval.to_dict()
                entry["informational"] = not metric.optimized
            report["metrics"][metric.name] = entry
        # 5. absolute constraints on candidate means (REQ-008)
        violated: list[str] = []
        unresolved_constraints: list[str] = []
        unmeasured_constraints: list[str] = []
        for metric in objective.optimized:
            if not metric.has_constraint:
                continue
            pair = observed[metric.name]
            if pair is None:
                report["constraints"][metric.name] = {"status": "unmeasured"}
                unmeasured_constraints.append(metric.name)
                continue
            mean_interval = paired_hoeffding_v1(pair[1], width=metric.width, alpha_interval=alpha_interval)
            status = "pass"
            checks: dict[str, Any] = {"candidate_mean_interval": mean_interval.to_dict()}
            if metric.floor is not None:
                checks["floor"] = metric.floor
                if mean_interval.high < metric.floor:
                    status = "violated"
                elif mean_interval.low < metric.floor:
                    status = "unresolved"
            if metric.ceiling is not None:
                checks["ceiling"] = metric.ceiling
                if mean_interval.low > metric.ceiling:
                    status = "violated"
                elif status != "violated" and mean_interval.high > metric.ceiling:
                    status = "unresolved"
            report["constraints"][metric.name] = {"status": status, **checks}
            (violated if status == "violated" else unresolved_constraints if status == "unresolved" else []).append(metric.name)
        if violated:
            return finish("rejected", "constraint_violation", decisive=violated[0])
        if unmeasured_constraints:
            return finish("unmeasured", "constraint_unmeasured", decisive=unmeasured_constraints[0])
        if unresolved_constraints:
            return finish("unresolved", "constraint_unresolved", decisive=unresolved_constraints[0])
        # 6. the mode rule
        if objective.mode == "lexicographic":
            return self._lexicographic(observed, report, finish)
        return self._pareto(observed, report, finish)

    # -- pieces ---------------------------------------------------------------------------------------------------------------

    def _observations(self, metric: MetricSpec, baseline: dict[str, TaskOutcome], candidate: dict[str, TaskOutcome],
                      report: dict[str, Any]) -> tuple[list[float], list[float]] | None | bool:
        """Both sides' values in expected-id order; ``None`` when any value is unknown; ``False`` when any is invalid."""
        sides: list[list[float]] = []
        for label, outcomes in (("baseline", baseline), ("candidate", candidate)):
            values: list[float] = []
            for task_id in self.context.expected_task_ids:
                value = outcomes[task_id].metric(metric.name)
                if value is None:
                    return None
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                    report.setdefault("invalid_values", []).append({"side": label, "task_id": task_id, "metric": metric.name, "value": repr(value)})
                    return False
                if not metric.in_bounds(float(value)):
                    report.setdefault("invalid_values", []).append({"side": label, "task_id": task_id, "metric": metric.name, "value": value,
                                                                     "bounds": [metric.lower_bound, metric.upper_bound]})
                    return False
                values.append(float(value))
            sides.append(values)
        return sides[0], sides[1]

    def _lexicographic(self, observed, report, finish) -> dict[str, Any]:
        """REQ-009: in declared order, accept at the first demonstrated improvement, reject at the first demonstrated
        regression, move on only through equivalence, stop unresolved otherwise. An earlier win claims nothing later."""
        for metric in self.objective.optimized:
            if observed[metric.name] is None:
                return finish("unmeasured", "metric_unmeasured", decisive=metric.name)
            interval = Interval(**report["metrics"][metric.name]["oriented_interval"])
            report["metrics"][metric.name]["reached"] = True
            if interval.low > metric.min_improvement:
                report["metrics"][metric.name]["verdict"] = "improved"
                return finish("accepted", "improvement", decisive=metric.name)
            if interval.high < 0:
                report["metrics"][metric.name]["verdict"] = "regressed"
                return finish("rejected", "metric_regression", decisive=metric.name)
            if interval.within(-metric.equivalence_margin, metric.equivalence_margin):
                report["metrics"][metric.name]["verdict"] = "equivalent"
                continue
            report["metrics"][metric.name]["verdict"] = "unresolved"
            return finish("unresolved", "interval_too_wide", decisive=metric.name)
        return finish("equivalent", "all_metrics_equivalent")

    def _pareto(self, observed, report, finish) -> dict[str, Any]:
        """REQ-010: accept only if every optimized metric is protected (L >= -r) and at least one improves (L > g).
        Opposing demonstrated changes are a trade-off; a material regression alone is a regression."""
        improved: list[str] = []
        regressed: list[str] = []
        protected = True
        all_equivalent = True
        for metric in self.objective.optimized:
            if observed[metric.name] is None:
                return finish("unmeasured", "metric_unmeasured", decisive=metric.name)
            interval = Interval(**report["metrics"][metric.name]["oriented_interval"])
            entry = report["metrics"][metric.name]
            if interval.low > metric.min_improvement:
                improved.append(metric.name)
                entry["verdict"] = "improved"
            elif interval.high < -metric.non_regression_margin:
                regressed.append(metric.name)
                entry["verdict"] = "regressed"
            elif interval.low >= -metric.non_regression_margin:
                entry["verdict"] = "tolerated_loss" if interval.low < 0 else "held"
            else:
                entry["verdict"] = "unresolved"
            if interval.low < -metric.non_regression_margin:
                protected = False
            if not interval.within(-metric.equivalence_margin, metric.equivalence_margin):
                all_equivalent = False
        if improved and protected:
            return finish("accepted", "improvement", decisive=improved[0])
        if regressed and improved:
            return finish("rejected", "trade_off", decisive=regressed[0])
        if regressed:
            return finish("rejected", "metric_regression", decisive=regressed[0])
        if all_equivalent:
            return finish("equivalent", "all_metrics_equivalent")
        return finish("unresolved", "interval_too_wide")


__all__ = [
    "DECISION_SCHEMA_VERSION",
    "DISPOSITIONS",
    "PROBE_DISPOSITION",
    "Decision",
    "Evaluation",
    "Evaluator",
    "Gate",
    "ObjectiveGate",
    "PairedGate",
    "StrictImprovementGate",
    "TieAcceptingGate",
]
