# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Execution traces (Alg. 1 line 6) and the refiner context C_k (line 7). REQUIREMENTS D.

A :class:`Trace` is what the host recorded when the agent ran a task under a graph: a text rendering of the
trajectory, the structured steps if the host has them, optional media and the scored outcome. The
:class:`TraceSource` protocol lets the host decide whether traces come from a fresh rollout (the paper's setup) or
from production episodes that later received a score (the same thing at zero extra cost).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .model import ImagePart

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Graph


@dataclass
class Step:
    """One trajectory element: an action (a tool call or a host status event such as ``Month_Start``), its arguments
    and the observation that followed. ``action`` is what :class:`proceduralgraph.guidance.Localizer` matches."""

    action: str
    args: Any = None
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "args": self.args, "observation": self.observation}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Step:
        return cls(action=value["action"], args=value.get("args"), observation=value.get("observation", ""))

    def rendered(self) -> str:
        head = f"Action: {self.action}" + (f"({_short(self.args, 200)})" if self.args is not None else "")
        return head + (f"\nObservation: {self.observation}" if self.observation else "")


Trajectory = Sequence[Step]


@dataclass
class TaskOutcome:
    task_id: str
    score: float  # in [0, 1]; the paper's S_i
    passed: bool
    prediction: Any = None
    truth: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "passed": self.passed,
            "prediction": self.prediction,
            "truth": self.truth,
            "meta": self.meta,
        }


@dataclass
class Trace:
    id: str
    outcome: TaskOutcome
    text: str
    steps: list[Step] = field(default_factory=list)
    media: list[ImagePart] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # opaque host provenance
    graph_ref: str | None = None  # the graph revision the episode ran under (D3)
    nodes_visited: list[str] = field(default_factory=list)  # Guide.visited, copied by the host (C7)
    guidance: list[str] = field(default_factory=list)  # Guide.guidance_log texts, copied by the host (C7)

    @property
    def task_id(self) -> str:
        return self.outcome.task_id

    def to_document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outcome": self.outcome.to_dict(),
            "text": self.text,
            "steps": [s.to_dict() for s in self.steps],
            "media": [m.to_dict() for m in self.media],
            "meta": self.meta,
            "graph_ref": self.graph_ref,
            "nodes_visited": list(self.nodes_visited),
            "guidance": list(self.guidance),
        }

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Trace:
        return cls(
            id=value["id"],
            outcome=TaskOutcome(**value["outcome"]),
            text=value.get("text", ""),
            steps=[Step.from_dict(s) for s in value.get("steps", [])],
            media=[ImagePart.from_dict(m) for m in value.get("media", [])],
            meta=dict(value.get("meta", {})),
            graph_ref=value.get("graph_ref"),
            nodes_visited=list(value.get("nodes_visited", [])),
            guidance=list(value.get("guidance", [])),
        )

    def rendered(self, *, head_ref: str | None = None) -> str:
        """One block of the refiner's ``{attempts_block}`` (D4). With ``head_ref``, a trace that ran under a different
        graph revision is marked ``STALE`` so the refiner knows the behaviour predates the current graph (D3)."""
        status = "PASSED" if self.outcome.passed else "FAILED"
        head = f"=== Trace {self.id} | task {self.task_id} | score {self.outcome.score:.3f} | {status}"
        if self.nodes_visited:
            head += " | nodes: " + " → ".join(self.nodes_visited)
        if self.graph_ref:
            head += f" | graph {self.graph_ref[:12]}"
            if head_ref is not None and self.graph_ref != head_ref:
                head += " STALE (ran under an earlier graph, not the one being refined)"
        head += " ==="
        body = self.text
        if not body and self.steps:
            body = "\n".join(s.rendered() for s in self.steps)
        if self.outcome.prediction is not None or self.outcome.truth is not None:
            body += f"\nprediction: {_short(self.outcome.prediction)}\nground truth: {_short(self.outcome.truth)}"
        return f"{head}\n{body}"


class TraceSource(Protocol):
    async def collect(self, graph: Graph, *, graph_ref: str | None, iteration: int) -> list[Trace]:
        """Rollout(G_{k-1}, B_k): run the training batch under ``graph`` (with a ``Guide``) or return production
        episodes that already ran under ``graph_ref`` and later received a score."""
        ...


class StaticTraceSource:
    """A fixed list of traces (tests, archived evidence)."""

    def __init__(self, traces: Sequence[Trace]):
        self.traces = list(traces)

    async def collect(self, graph: Graph, *, graph_ref: str | None, iteration: int) -> list[Trace]:
        return list(self.traces)


class StridedTraceSource:
    """Sequential strides of ``stride`` traces over a fixed list (App. B.1: S=100 / S=20), wrapping around when the
    list is exhausted so a long run never returns an empty batch. Iteration ``k`` (1-based) gets stride ``k-1``."""

    def __init__(self, traces: Sequence[Trace], stride: int):
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.traces, self.stride = list(traces), stride

    async def collect(self, graph: Graph, *, graph_ref: str | None, iteration: int) -> list[Trace]:
        if not self.traces:
            return []
        n = len(self.traces)
        start = ((max(iteration, 1) - 1) * self.stride) % n
        return [self.traces[(start + i) % n] for i in range(min(self.stride, n))]


def stratified_sample(
    traces: Sequence[Trace],
    *,
    failing: int | None,
    passing: int | None,
    seen: set[str] | None = None,
    seed: int = 17,
    iteration: int = 0,
) -> tuple[list[Trace], set[str]]:
    """D5: ``None`` for both budgets means the paper (every trace). Otherwise up to ``failing`` failing and
    ``passing`` passing traces, rotating through unseen traces first; an exhausted stratum resets its ``seen`` set
    rather than sampling fewer than the budget. Returns the sample and the updated ``seen`` set."""
    seen = set(seen or ())
    if failing is None and passing is None:
        return list(traces), seen | {t.id for t in traces}
    rng = random.Random(f"{seed}:{iteration}")
    sample: list[Trace] = []
    for wanted, predicate in ((failing, lambda t: not t.outcome.passed), (passing, lambda t: t.outcome.passed)):
        pool = [t for t in traces if predicate(t)]
        if wanted is None:
            sample.extend(pool)
            seen.update(t.id for t in pool)
            continue
        if not pool or wanted <= 0:
            continue
        unseen = [t for t in pool if t.id not in seen]
        if len(unseen) < min(wanted, len(pool)):
            for t in pool:
                seen.discard(t.id)
            unseen = pool
        chosen = rng.sample(unseen, min(wanted, len(unseen)))
        sample.extend(chosen)
        seen.update(t.id for t in chosen)
    return sample, seen


def tail(text: str, cap: int, token_counter: Callable[[str], int] | None = None) -> str:
    """Tail_Lmax (App. B.6): keep the END of ``text``. Shorter inputs are returned unchanged. Characters by default;
    with ``token_counter`` the cap is in tokens and the cut is found by bisection on the counter."""
    if cap <= 0:
        return ""
    if token_counter is None:
        if len(text) <= cap:
            return text
        return f"[... {len(text) - cap} earlier characters discarded ...]\n" + text[-cap:]
    if token_counter(text) <= cap:
        return text
    lo, hi = 0, len(text)  # find the smallest start offset whose tail fits in ``cap`` tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if token_counter(text[mid:]) <= cap:
            hi = mid
        else:
            lo = mid + 1
    return f"[... earlier tokens discarded ({lo} characters) ...]\n" + text[lo:]


def render_attempts_block(
    traces: Sequence[Trace],
    *,
    cap: int,
    token_counter: Callable[[str], int] | None = None,
    success_threshold: float = 0.5,
    head_ref: str | None = None,
) -> str:
    """C_k = Tail_Lmax(ConcatTrajectories(E_k)) (Alg. 1 line 7), plus the D4 decision: high-scoring traces are
    concatenated first and low-scoring traces last, so the tail cut keeps the failures the refiner is asked to
    contrast. The partition uses ``success_threshold`` (non-binary scores); the PASSED/FAILED header word is the
    host's ``passed`` flag."""
    high = [t for t in traces if t.outcome.score >= success_threshold]
    low = [t for t in traces if t.outcome.score < success_threshold]
    stale = [t.id for t in traces if head_ref is not None and t.graph_ref and t.graph_ref != head_ref]
    ordered = sorted(high, key=lambda t: (-t.outcome.score, t.id)) + sorted(low, key=lambda t: (-t.outcome.score, t.id))
    summary = (
        f"{len(traces)} trajectories: {len(high)} high-scoring (score >= {success_threshold:g}), "
        f"{len(low)} low-scoring. Low-scoring trajectories are listed last."
        + (f" {len(stale)} ran under an earlier graph and are marked STALE." if stale else "")
    )
    body = "\n\n".join(t.rendered(head_ref=head_ref) for t in ordered)
    usage = summarize_node_usage(traces)
    text = summary + "\n\n" + body + (f"\n\n{usage}" if usage else "")
    return tail(text, cap, token_counter)


def summarize_node_usage(traces: Sequence[Trace]) -> str:
    """Per-node in-play counts on failing versus passing traces (D5). Empty when no trace has ``nodes_visited``."""
    usage: dict[str, list[int]] = {}
    for trace in traces:
        for name in dict.fromkeys(trace.nodes_visited):
            counts = usage.setdefault(name, [0, 0])
            counts[0 if not trace.outcome.passed else 1] += 1
    if not usage:
        return ""
    lines = ["Node usage across these trajectories:", "| node | visited in failing | visited in passing |", "| --- | --- | --- |"]
    for name, (failing, passing) in sorted(usage.items(), key=lambda kv: (-kv[1][0], kv[0])):
        lines.append(f"| {name} | {failing} | {passing} |")
    return "\n".join(lines)


def _short(value: Any, limit: int = 120) -> str:
    text = repr(value) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "StaticTraceSource",
    "Step",
    "StridedTraceSource",
    "TaskOutcome",
    "Trace",
    "TraceSource",
    "Trajectory",
    "render_attempts_block",
    "stratified_sample",
    "summarize_node_usage",
    "tail",
]
