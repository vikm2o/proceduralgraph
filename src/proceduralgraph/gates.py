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

import random
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .traces import TaskOutcome

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Graph


@dataclass
class Evaluation:
    ref: str | None  # host-opaque identity of the evaluated graph (digest, bundle id, ...)
    score: float | None  # S_val(G); None when the host's gate does not use a scalar
    aggregate: dict[str, Any] = field(default_factory=dict)
    per_task: list[TaskOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "score": self.score, "aggregate": self.aggregate, "tasks": len(self.per_task)}

    def to_document(self) -> dict[str, Any]:
        """Full form, including per-task outcomes, for resume checkpoints and rejection memory."""
        return {"ref": self.ref, "score": self.score, "aggregate": self.aggregate, "per_task": [t.to_dict() for t in self.per_task]}

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Evaluation:
        return cls(
            ref=value.get("ref"),
            score=value.get("score"),
            aggregate=dict(value.get("aggregate", {})),
            per_task=[TaskOutcome(**t) for t in value.get("per_task", [])],
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


__all__ = ["Decision", "Evaluation", "Evaluator", "Gate", "PairedGate", "StrictImprovementGate", "TieAcceptingGate"]
