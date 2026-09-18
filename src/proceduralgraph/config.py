# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Loop configuration. Defaults follow the paper (Alg. 1, §4, App. B, App. D.3) where it states a number; every other
default is a recorded decision (docs/paper-differences.md §0). REQUIREMENTS G2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .objectives import ObjectiveContext, ObjectiveSpec

MODES = ("auto", "static_incremental", "scratch_incremental")
ONETIME_MODES = ("static_onetime", "scratch_onetime")
CYCLE_POLICIES = ("allow", "repair")
STALE_TRACE_POLICIES = ("warn", "drop", "allow")


@dataclass(frozen=True)
class Budget:
    """Hard ceilings for one ``evolve`` run. ``None`` means unlimited.

    ``max_output_tokens`` counts ``ModelResponse.usage["output_tokens"]`` when the model reports it; a model that
    reports nothing cannot exhaust it. ``max_evaluations`` counts validation runs (the baseline included). When a
    ceiling is hit the run stops with ``stopped_reason="budget_exhausted"``; everything already saved stays saved.
    """

    max_model_calls: int | None = None
    max_output_tokens: int | None = None
    max_evaluations: int | None = None
    max_seconds: float | None = None


@dataclass(frozen=True)
class EvolveConfig:
    task_description: str = "tasks in this workspace"
    # K rounds (§5.4, App. E run ten). Alg. 1 has no early stop: ``max_rejected_streak=None`` runs every round.
    max_rounds: int = 10
    max_rejected_streak: int | None = None
    # PrepareCandidate's cycle policy ``c`` (App. B.6). "repair" removes cycle-closing edges before validation.
    cycle_policy: str = "repair"
    # Refinement mode placeholder (App. D.2). "auto" = scratch_incremental on the skeleton, static_incremental otherwise.
    mode: str = "auto"
    # C_k = Tail_Lmax(ConcatTrajectories(E_k)). L_max is unspecified; characters unless a token counter is supplied.
    refiner_context_cap: int = 120_000
    token_counter: Callable[[str], int] | None = None
    # R_k = SerializeRejections(H_rejected): most recent entries in full, older ones as one line each.
    rejections_full_entries: int = 5
    rejections_char_cap: int = 30_000
    # App. D.3: 8,192 tokens for the refiner. Guidance output length lives in GuidanceConfig (the loop never calls it).
    refiner_max_tokens: int = 8192
    # One retry feeding diagnostics back when the refiner's output cannot be applied (departure; 0 = paper-exact).
    role_retries: int = 1
    # None = the paper: every trace in the batch reaches the refiner. Set both to sample stratified, rotating.
    failing_sample: int | None = None
    passing_sample: int | None = None
    # Traces whose graph_ref is not the current head: "warn" (count and report), "drop", or "allow" silently.
    stale_trace_policy: str = "warn"
    # Non-binary scores: at or above this is "high-scoring" for the attempts-block ordering (decision).
    success_threshold: float = 0.5
    # Images from Trace.media attached to the refiner request (most recent failing traces first). 0 = paper, text only.
    refiner_images: int = 0
    seed: int = 17
    workspace: str = "default"
    budget: Budget = field(default_factory=Budget)
    # Objective mode (REQ-007, opt-in): the frozen objective and the evidence context every evaluation must carry. Both or
    # neither. With an objective set, the loop enforces Evaluation.ref against the graph it asked to evaluate, binds
    # checkpoints to the objective and context digests, and renders both to the refiner.
    objective: ObjectiveSpec | None = None
    objective_context: ObjectiveContext | None = None

    def __post_init__(self) -> None:
        if (self.objective is None) != (self.objective_context is None):
            raise ValueError("objective and objective_context must be set together")
        if self.objective is not None and not self.objective_context.bound_to(self.objective):
            raise ValueError("objective_context.objective_digest does not match objective.digest")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}; the one-time modes are only reachable through refine_once()")
        if self.cycle_policy not in CYCLE_POLICIES:
            raise ValueError(f"cycle_policy must be one of {CYCLE_POLICIES}")
        if self.stale_trace_policy not in STALE_TRACE_POLICIES:
            raise ValueError(f"stale_trace_policy must be one of {STALE_TRACE_POLICIES}")
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be >= 0")


__all__ = ["CYCLE_POLICIES", "MODES", "ONETIME_MODES", "STALE_TRACE_POLICIES", "Budget", "EvolveConfig"]
