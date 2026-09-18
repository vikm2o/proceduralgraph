# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Host hooks: make paid stages idempotent, meter model calls, observe iterations (REQUIREMENTS G9, H1).

Every method is optional; subclass and override what you need.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Budget
from .model import ChatModel, ModelRequest, ModelResponse

OUTCOMES = ("accepted", "rejected", "structural_failure", "no_action", "duplicate_candidate")


class BudgetExceeded(Exception):
    """A :class:`proceduralgraph.config.Budget` ceiling was reached. The harness catches this and stops cleanly."""

    def __init__(self, limit: str, used: float, ceiling: float):
        super().__init__(f"budget exhausted: {limit} used {used} of {ceiling}")
        self.limit, self.used, self.ceiling = limit, used, ceiling


@dataclass
class IterationReport:
    """One Alg. 1 round as seen by the host (G8). ``outcome`` is one of :data:`OUTCOMES`."""

    iteration: int
    trace_ids: list[str]
    stale_trace_ids: list[str]
    edits: dict[str, Any]
    candidate_digest: str | None
    outcome: str
    evaluation: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    graph_ref: str | None = None
    rejections_ref: str | None = None
    resumed: bool = False
    decision: dict[str, Any] | None = None  # the gate's Decision.feedback (REQ-016); omitted from to_dict when absent

    @property
    def accepted(self) -> bool:
        return self.outcome == "accepted"

    @property
    def disposition(self) -> str | None:
        """The gate's disposition when a structured decision was recorded (``accepted`` / ``rejected`` / ``equivalent`` /
        ``unresolved`` / ``unmeasured`` / ``invalid``); ``None`` for scalar gates."""
        return None if self.decision is None else self.decision.get("disposition")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "iteration": self.iteration,
            "trace_ids": list(self.trace_ids),
            "stale_trace_ids": list(self.stale_trace_ids),
            "edits": self.edits,
            "candidate_digest": self.candidate_digest,
            "outcome": self.outcome,
            "evaluation": self.evaluation,
            "diagnostics": list(self.diagnostics),
            "warnings": list(self.warnings),
            "graph_ref": self.graph_ref,
            "rejections_ref": self.rejections_ref,
            "resumed": self.resumed,
        }
        if self.decision is not None:
            value["decision"] = self.decision
        return value


class Hooks:
    @contextlib.asynccontextmanager
    async def stage(self, name: str) -> AsyncIterator[None]:
        """Wraps one paid stage (``baseline``, ``rollout:k``, ``refiner:k``, ``validation:k``)."""
        yield

    async def on_model_call(self, role: str, request: ModelRequest, response: ModelResponse) -> None:
        return None

    async def on_iteration(self, report: IterationReport) -> None:
        return None

    async def on_warning(self, message: str) -> None:
        return None


@dataclass
class BudgetMeter:
    """Running totals checked against a :class:`Budget`. Shared by the hooked model and the harness."""

    budget: Budget
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    output_tokens: int = 0
    evaluations: int = 0

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started

    def check(self, *, about_to: str = "") -> None:
        """Raise :class:`BudgetExceeded` if any ceiling is already reached. ``about_to`` names the next paid step, so a
        stage is refused before it spends rather than after."""
        b = self.budget
        if b.max_model_calls is not None and self.model_calls >= b.max_model_calls:
            raise BudgetExceeded("max_model_calls", self.model_calls, b.max_model_calls)
        if b.max_output_tokens is not None and self.output_tokens >= b.max_output_tokens:
            raise BudgetExceeded("max_output_tokens", self.output_tokens, b.max_output_tokens)
        if b.max_evaluations is not None and about_to == "evaluation" and self.evaluations >= b.max_evaluations:
            raise BudgetExceeded("max_evaluations", self.evaluations, b.max_evaluations)
        if b.max_seconds is not None and self.seconds >= b.max_seconds:
            raise BudgetExceeded("max_seconds", round(self.seconds, 1), b.max_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "output_tokens": self.output_tokens,
            "evaluations": self.evaluations,
            "seconds": round(self.seconds, 1),
            "ceilings": {
                "max_model_calls": self.budget.max_model_calls,
                "max_output_tokens": self.budget.max_output_tokens,
                "max_evaluations": self.budget.max_evaluations,
                "max_seconds": self.budget.max_seconds,
            },
        }


class HookedModel:
    """Wraps a ChatModel so every call is reported to ``hooks.on_model_call`` and counted against the budget.

    The budget is checked BEFORE each call; the check raises :class:`BudgetExceeded`, which the harness turns into a
    clean ``budget_exhausted`` stop. A host that runs the rollout in-process hands the same ``HookedModel`` to its
    :class:`proceduralgraph.guidance.Guide` so guidance calls are metered under the same budget (G9).
    """

    def __init__(self, model: ChatModel, hooks: Hooks, meter: BudgetMeter | None = None):
        self.model, self.hooks = model, hooks
        self.meter = meter or BudgetMeter(Budget())

    @property
    def calls(self) -> int:
        return self.meter.model_calls

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.meter.check(about_to="model_call")
        response = await self.model.complete(request)
        self.meter.model_calls += 1
        usage = response.usage or {}
        tokens = usage.get("output_tokens", usage.get("completion_tokens"))  # Anthropic / OpenAI names
        if isinstance(tokens, int | float):
            self.meter.output_tokens += int(tokens)
        await self.hooks.on_model_call(request.role, request, response)
        return response


__all__ = ["OUTCOMES", "BudgetExceeded", "BudgetMeter", "HookedModel", "Hooks", "IterationReport"]
