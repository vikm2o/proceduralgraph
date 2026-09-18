# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Bootstrap a first graph from a problem statement, and optionally worked solutions, before any trace exists.

The paper builds graphs two ways: an expert writes one, or the refiner synthesises one from execution traces in the
``scratch`` modes (App. D.2). A team that has a task description and a few worked solutions, but no agent traces yet,
has neither. This role fills the gap with ``refine_once`` in ``scratch_onetime`` mode (G6, the paper's Mode 4): the
problem statement is the task context and each worked solution is presented as one successful trajectory. When there
are no solutions, the trajectories slot holds :data:`NO_SOLUTIONS_BLOCK` instead, one added sentence that asks the
refiner to design the procedure from the task context and the tool list; the rejected-candidates slot carries the same
placeholder ``refine_once`` always uses. Both placeholders are recorded in docs/paper-differences.md §2.19.

A bootstrapped graph is a *starting point*, not a validated one: like the paper's one-time modes it is committed
without a gate. The first ``evolve`` run scores it as the baseline, and every later change is gated as usual. It still
passes the same structural checks as a refiner candidate or a hand-written graph; a bootstrap that fails them is
refused with diagnostics, never seeded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..config import Budget, EvolveConfig
from ..edits import EditSet
from ..graph import Diagnostic, Graph
from ..hooks import BudgetMeter, HookedModel, Hooks
from ..model import ChatModel
from ..traces import TaskOutcome, Trace

NO_SOLUTIONS_BLOCK = (
    "(none yet: no agent has run this task. Design the procedure from the task context and the Available Tool Actions "
    "list above: the order of actions, the decisions between them, and the checks that prevent common mistakes.)"
)
"""What fills ``{attempts_block}`` when there are no worked solutions. Not in the paper; recorded as a departure."""

ORIGIN = "bootstrapped"


@dataclass
class BootstrapResult:
    """What :func:`bootstrap_graph` produced. ``graph`` is ``None`` when the bootstrap was refused; ``diagnostics`` say
    why. Pass the result itself to ``evolve(initial_graph=result)`` to seed with ``origin: bootstrapped``."""

    graph: Graph | None
    edits: EditSet
    problem_statement: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    mode: str = "scratch_onetime"
    calls: int = 0
    solutions: int = 0

    @property
    def ok(self) -> bool:
        return self.graph is not None

    def refusal(self) -> str:
        """One plain sentence for logs and the CLI when ``ok`` is false."""
        return "bootstrap refused: " + ("; ".join(str(d) for d in self.diagnostics if d.is_error) or "no graph produced")

    def seed_meta(self) -> dict[str, Any]:
        """The ``meta`` to record on the seed revision: origin, how many solutions, the mode and the refiner calls."""
        return {"origin": ORIGIN, "solutions": self.solutions, "mode": self.mode, "refiner_calls": self.calls}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "graph": None if self.graph is None else {"nodes": len(self.graph.nodes), "edges": len(self.graph.edges), "digest": self.graph.digest},
            "edits": self.edits.counts(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "mode": self.mode,
            "calls": self.calls,
            "solutions": self.solutions,
        }


def solutions_as_traces(solutions: Sequence[str]) -> list[Trace]:
    """Each non-empty worked solution becomes one passing trace (score 1.0) with the solution as its text."""
    kept = [text.strip() for text in solutions if text and text.strip()]
    return [Trace(id=f"solution-{i + 1}", outcome=TaskOutcome(task_id=f"solution-{i + 1}", score=1.0, passed=True), text=text) for i, text in enumerate(kept)]


async def bootstrap_graph(
    *,
    problem_statement: str,
    model: ChatModel,
    solutions: Sequence[str] = (),
    tools: Sequence[str] | None = None,
    config: EvolveConfig | None = None,
    hooks: Hooks | None = None,
) -> BootstrapResult:
    """One ``refine_once`` call in ``scratch_onetime`` mode that turns a problem statement and optional worked
    solutions into a graph. No store, no gate, no rejection memory.

    Either ``solutions`` or ``tools`` must be given: with neither, the refiner has nothing to build ACTION nodes from
    (the prompt's rule 1 needs a tool list, and without trajectories there is nothing to infer one from). ``config``
    supplies ``role_retries`` (one retry with diagnostics fed back by default), ``cycle_policy``,
    ``refiner_max_tokens`` and the budget; its ``task_description`` is replaced by ``problem_statement``. A
    ``HookedModel`` passed as ``model`` is reused as is (its hooks apply; its meter adopts ``config.budget`` when it
    has no ceilings of its own), so bootstrap spend can share a host's budget.
    """
    from ..harness import refine_once  # local: harness imports this module for the initial_graph coercion

    if not problem_statement or not problem_statement.strip():
        raise ValueError("problem_statement must not be empty")
    traces = solutions_as_traces(solutions)
    if not traces and not tools:
        raise ValueError("bootstrapping without worked solutions needs the tool list (tools=...): the refiner has nothing to build ACTION nodes from")
    settings = replace(config or EvolveConfig(), task_description=problem_statement.strip())
    if isinstance(model, HookedModel):
        hooked = model
        if hooked.meter.budget == Budget():
            hooked.meter.budget = settings.budget
    else:
        hooked = HookedModel(model, hooks or Hooks(), BudgetMeter(settings.budget))
    calls_before = hooked.meter.model_calls
    graph, edits, diagnostics = await refine_once(
        config=settings,
        model=hooked,
        graph=Graph.skeleton(),
        traces=traces,
        available_tools=tools,
        mode="scratch_onetime",
        attempts_block=None if traces else NO_SOLUTIONS_BLOCK,
    )
    diagnostics = list(diagnostics)
    if edits.is_empty or (graph is not None and graph.is_skeleton):
        graph = None
        message = "the refiner proposed no edits" if edits.is_empty else "the refiner's edits cancelled out"
        diagnostics.append(Diagnostic("no_action", f"{message}; the graph is still the Start → End skeleton", "reply"))
    return BootstrapResult(
        graph=graph,
        edits=edits,
        problem_statement=problem_statement.strip(),
        diagnostics=diagnostics,
        mode="scratch_onetime",
        calls=hooked.meter.model_calls - calls_before,
        solutions=len(traces),
    )


__all__ = ["NO_SOLUTIONS_BLOCK", "ORIGIN", "BootstrapResult", "bootstrap_graph", "solutions_as_traces"]
