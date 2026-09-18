# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The online guidance runtime (paper §3.2, eq. 2 and 3). REQUIREMENTS C. Storage-free.

    u_t = Match(a_{t-1}, V),  G_t = N_h(u_t) if u_t ≠ ∅ else G,  g_t = Ψ(G_t, q, T_{t-w:t})   (2)
    a_t ∼ P_solver(· | q, T_t, g_t)                                                            (3)

A host constructs a :class:`Guide` over a frozen :class:`Graph` at the start of an episode and calls
``await guide.guidance(query, trajectory)`` before each solver step, appending the returned text to the solver's
prompt. The graph never changes during an episode; the offline loop only ever hands the host a new frozen graph.
This module imports nothing from ``stores`` (C6).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .graph import START, Graph
from .model import ChatModel, ModelRequest, TextPart
from .serialize import render_trajectory, serialize_context
from .traces import Step, Trajectory

MODES = ("generative_subgraph", "generative_full", "raw_subgraph", "raw_full", "none")

GUIDANCE_PROMPT = (
    "You are an expert cognitive architect and execution guide for an AI agent solving the task: {task_description}\n"
    "Here is {graph_context_desc}: {subgraph_summary}\n"
    "Here is the current active query / observation: {query}\n"
    "Here is the agent's recent execution trajectory: {recent_context}\n"
    "Analyze this {graph_source} in the context of the agent's current progress. Using the condition, guidance,\n"
    "and pitfalls attributes carried by the edges in the graph context, generate clear, detailed, and actionable guidance\n"
    "advising the agent on exactly what step or strategy to pursue next, what pitfalls to avoid, and how to recover from\n"
    "recent failures if any. You must include any specific command patterns, file paths, tools, or arguments defined in the\n"
    "graph context if they are relevant to the next steps."
)
"""App. B.5 "Guidance Generation Prompt (Local subgraph; default)", verbatim. The full-graph variant binds the three
graph slots differently and is otherwise identical."""

SUBGRAPH_CONTEXT_DESC = "the localized Procedural Graph context around the agent's current node"
SUBGRAPH_SOURCE = "local graph context"
FULL_CONTEXT_DESC = "the complete Procedural Graph governing the task structure and strategic guidance"
FULL_SOURCE = "complete Procedural Graph"


class Localizer(Protocol):
    def locate(self, graph: Graph, trajectory: Trajectory) -> str | None:
        """``Match(a_{t-1}, V)``: the node id the agent is at, or ``None`` when matching fails."""
        ...


class ExactActionLocalizer:
    """The paper's ``Match``: the last step's ``action`` exactly equals a node id. An empty trajectory is at ``Start``
    (a_0 = Start). ``normalize`` lets a host map its tool-call format onto node ids without writing a localizer."""

    def __init__(self, normalize: Callable[[str], str] | None = None):
        self.normalize = normalize

    def locate(self, graph: Graph, trajectory: Trajectory) -> str | None:
        if not trajectory:
            return START if START in graph.nodes else None
        action = trajectory[-1].action
        if self.normalize is not None:
            action = self.normalize(action)
        return action if action in graph.nodes else None


@dataclass(frozen=True)
class GuidanceConfig:
    """``hops`` = h, ``window`` = w (§4: h=2, w=3). ``mode``:

    - ``generative_subgraph`` (paper default): N_h(u_t) serialized, guidance model called; full graph when Match fails.
    - ``generative_full``: the complete graph serialized, guidance model called (Table 3 "Full graph, generative").
    - ``raw_subgraph``: the serialized N_h(u_t) returned as the guidance, no model call. Not in the paper's ablation
      (Table 3 only injects the raw *full* graph); provided as the cheapest localized option.
    - ``raw_full``: the serialized complete graph, no model call (Table 3 "Full graph, raw injection").
    - ``none``: empty guidance, the no-graph baseline, for A/B tests.
    """

    hops: int = 2
    window: int = 3
    mode: str = "generative_subgraph"
    include_relations: bool = True
    max_tokens: int = 1024
    task_description: str = "tasks in this workspace"
    max_guidance_calls: int | None = None  # C9: after this many model calls per Guide, fall back to raw_subgraph

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.hops < 1 or self.window < 0:
            raise ValueError("hops must be >= 1 and window >= 0")


@dataclass
class Guidance:
    """g_t plus what produced it, so the host can put it in the trace (C4, C7)."""

    text: str
    node_id: str | None
    subgraph_digest: str | None
    used_full_graph: bool
    context: str  # the serialized graph context the model saw (or that was returned raw)
    usage: dict[str, Any] = field(default_factory=dict)
    mode: str = "generative_subgraph"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "node_id": self.node_id,
            "subgraph_digest": self.subgraph_digest,
            "used_full_graph": self.used_full_graph,
            "usage": dict(self.usage),
            "mode": self.mode,
        }


class Guide:
    """Per-episode guidance over a frozen graph (C4). ``visited`` and ``guidance_log`` record what happened so the host
    can copy them into ``Trace.nodes_visited`` / ``Trace.guidance`` (C7)."""

    def __init__(
        self,
        graph: Graph,
        model: ChatModel | None = None,
        config: GuidanceConfig | None = None,
        localizer: Localizer | None = None,
    ):
        self.graph = graph
        self.model = model
        self.config = config or GuidanceConfig()
        self.localizer = localizer or ExactActionLocalizer()
        if self.config.mode.startswith("generative") and model is None:
            raise ValueError(f"mode {self.config.mode!r} needs a ChatModel; use a raw_* mode or 'none' without one")
        self.visited: list[str] = []
        self.guidance_log: list[Guidance] = []
        self.model_calls = 0

    # -- the per-step entry point ---------------------------------------------------------------------------------

    async def guidance(self, query: str, trajectory: Trajectory = ()) -> Guidance:
        """eq. 2: localize, pick N_h(u_t) or the full graph, serialize, and (in generative modes) call Ψ."""
        mode = self.config.mode
        node_id = self.localizer.locate(self.graph, trajectory)
        if node_id is not None:
            self.visited.append(node_id)
        if mode == "none":
            result = Guidance(text="", node_id=node_id, subgraph_digest=None, used_full_graph=False, context="", mode=mode)
            self.guidance_log.append(result)
            return result
        context, subgraph_digest, use_full = self._serialize(node_id, force_full=mode.endswith("_full"))
        generative = mode.startswith("generative")
        if generative and self.config.max_guidance_calls is not None and self.model_calls >= self.config.max_guidance_calls:
            generative = False  # C9: the episode ran away; serve the raw context instead of paying again
            mode = "raw_subgraph" if not use_full else "raw_full"
        if not generative:
            result = Guidance(text=context, node_id=node_id, subgraph_digest=subgraph_digest, used_full_graph=use_full, context=context, mode=mode)
            self.guidance_log.append(result)
            return result
        prompt = self.render_prompt(query, trajectory, context, full=use_full)
        request = ModelRequest(role="guidance", system="", parts=[TextPart(prompt)], max_tokens=self.config.max_tokens)
        assert self.model is not None
        response = await self.model.complete(request)
        self.model_calls += 1
        result = Guidance(
            text=response.text.strip(),
            node_id=node_id,
            subgraph_digest=subgraph_digest,
            used_full_graph=use_full,
            context=context,
            usage=dict(response.usage or {}),
            mode=mode,
        )
        self.guidance_log.append(result)
        return result

    def guidance_sync(self, query: str, trajectory: Trajectory = ()) -> Guidance:
        """C8: for synchronous hosts. Not for use inside a running event loop."""
        return asyncio.run(self.guidance(query, trajectory))

    # -- pieces a host may want on their own ----------------------------------------------------------------------

    def render_prompt(self, query: str, trajectory: Trajectory, context: str, *, full: bool) -> str:
        recent = render_trajectory(list(trajectory), window=self.config.window)
        return GUIDANCE_PROMPT.format(
            task_description=self.config.task_description,
            graph_context_desc=FULL_CONTEXT_DESC if full else SUBGRAPH_CONTEXT_DESC,
            subgraph_summary="\n" + context,
            query=query,
            recent_context="\n" + recent,
            graph_source=FULL_SOURCE if full else SUBGRAPH_SOURCE,
        )

    def _serialize(self, node_id: str | None, *, force_full: bool) -> tuple[str, str | None, bool]:
        """eq. 2's G_t as text: (context, subgraph digest or None, used_full_graph). The full graph when ``force_full``
        (the ``*_full`` modes) or when Match failed (``node_id is None``)."""
        if force_full or node_id is None:
            return serialize_context(self.graph, None, include_relations=self.config.include_relations), None, True
        context = serialize_context(self.graph, node_id, self.config.hops, include_relations=self.config.include_relations)
        return context, self.graph.neighborhood(node_id, self.config.hops).digest, False

    def context_for(self, trajectory: Trajectory) -> tuple[str | None, str, bool]:
        """(node id, serialized context, used_full_graph) without any model call; what the CLI ``guide`` command shows."""
        node_id = self.localizer.locate(self.graph, trajectory)
        context, _, use_full = self._serialize(node_id, force_full=self.config.mode.endswith("_full"))
        return node_id, context, use_full


def steps_from_actions(actions: Sequence[str]) -> list[Step]:
    """Convenience for tests and the CLI: bare action names become steps with no arguments or observations."""
    return [Step(action=a) for a in actions]


__all__ = [
    "FULL_CONTEXT_DESC",
    "FULL_SOURCE",
    "GUIDANCE_PROMPT",
    "MODES",
    "SUBGRAPH_CONTEXT_DESC",
    "SUBGRAPH_SOURCE",
    "ExactActionLocalizer",
    "Step",
    "Guidance",
    "GuidanceConfig",
    "Guide",
    "Localizer",
    "steps_from_actions",
]
