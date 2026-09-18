# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The refiner role (paper §3.3 Step 2, eq. 6; App. B.5 "Refiner Prompt"; Alg. 1 line 9). REQUIREMENTS E.

    ΔG_k ← Refiner(G_{k-1}, C_k, {S_i}, R_k)

One single-shot call: one system message (the prompt's first two sentences) and one user message (everything else,
verbatim and in the paper's order), plain text back, one JSON block with four arrays. The only text not in the paper
is the ``Allowed node types`` line (and ``Allowed relations`` when a host changed the vocabulary), inserted directly
after the "Available Tool Actions" line because the structural checks enforce those vocabularies and the paper's
prompt never states them (E1; departure recorded in docs/paper-differences.md).

E2: when the reply cannot be parsed or applied, the harness may retry once, feeding the diagnostics back verbatim as a
second user turn folded into the same transcript. With ``role_retries=0`` the paper's behaviour is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import EvolveConfig
from ..edits import EditError, EditSet, prepare_candidate
from ..graph import DEFAULT_RELATIONS, Diagnostic, Graph, errors
from ..model import ChatModel, ImagePart, ModelRequest, TextPart

REFINER_SYSTEM = (
    "You are an expert cognitive architect optimizing a Procedural Graph for an intelligent agent. The Procedural Graph\n"
    "encodes structured procedural guidance."
)

_USER_LINES = [
    "Task context: {task_description}",
    "Refinement mode: {mode}",
    "Available Tool Actions (the agent can only execute these actions): {available_tools_list}",
    # {vocabulary_lines} is the package's one insertion (E1); it renders to nothing when there is nothing to say.
    "{vocabulary_lines}",
    "Recent execution trajectories: {attempts_block}",
    "Current Procedural Graph representation: {current_graph_json}",
    "Previously rejected candidates: {rejected_block}",
    "Your job is to refine the Procedural Graph. Follow these guidelines based on the mode:",
    "• static_onetime / static_incremental: Prune edges/nodes that lead to loops, deadlocks, or failures.",
    "  Add missing nodes and edges that could fix the failures and improve performance for future tasks.",
    "• scratch_onetime / scratch_incremental: If starting from scratch (the graph contains only",
    "  Start → End), synthesize a brand new, complete Procedural Graph using the Available Tool Actions list, Status,",
    "  and successful patterns in the trajectories. Otherwise, prune edges/nodes that lead to loops, deadlocks, or",
    "  failures, and add missing nodes and edges based on the given graph.",
    "Rules for nodes and edges. Rules 2–4 describe the edge attributes in Φ(e): condition, guidance, and",
    "pitfalls. The remaining rules govern node compatibility, generality, and graph structure.",
    '1. Action Nodes. Any node of type ACTION must match one of the action/tool names in the "Available Tool Actions"',
    "   list above.",
    "2. Transition Conditions. If an edge has a condition, provide a natural-language semantic precondition under",
    '   which this transition should fire (e.g., "When dialogue history has been parsed but target constraints are',
    '   unknown"). Use null if the transition is unconditional.',
    "3. Execution Guidance. For every edge added in add_edges, you MUST provide a guidance string detailing",
    "   exactly what action to take next and the strategic rationale behind it.",
    "4. Pitfalls. Provide a pitfalls string warning about premature actions, forbidden words, or common formatting",
    "   pitfalls to avoid during this step.",
    "5. Generality & Leak Prevention. The updated Procedural Graph must guide the agent effectively without",
    "   overfitting to specific details of a single trajectory. Use high-level conceptual descriptions.",
    "6. Node ID Compatibility. If refining an existing graph (static modes), you MUST preserve the existing node IDs",
    "   (such as Month_Start, Decide_Capital, and the tool names) so they remain compatible with the",
    "   environment's state tracker. Do not rename them.",
    "7. Graph Structure. Follow the task's configured cycle policy. Every edge must reference existing nodes, and every",
    "   node must have a directed path to a terminal node. The environment loop handles repetition across simulation",
    "   cycles.",
    "Please propose the exact set of edits to perform. You must output your edits as a single valid JSON block containing",
    "four arrays: add_nodes, delete_nodes, add_edges, and delete_edges. Output format must be exactly:",
    '{{ "add_nodes": [{{"id":..., "type": "ACTION", "description":...}}],',
    '  "delete_nodes": ["node_id"],',
    '  "add_edges": [{{"source":..., "target":..., "relation":...,',
    '                 "condition":..., "guidance":..., "pitfalls":...}}],',
    '  "delete_edges": [{{"source":..., "target":...}}] }}',
    "Make sure to output ONLY the raw JSON block.",
]
REFINER_USER_TEMPLATE = "\n".join(_USER_LINES)

TOOLS_UNSPECIFIED = (
    "not specified: preserve existing ACTION node ids and add ACTION nodes only for actions that appear in the trajectories"
)

RETRY_TEMPLATE = (
    "{original}\n\n"
    "--- Your previous reply ---\n{reply}\n\n"
    "--- It could not be applied. Structural diagnostics ---\n{diagnostics}\n\n"
    "Fix these problems and reply again with ONLY the corrected raw JSON block (the same four arrays)."
)


def vocabulary_lines(graph: Graph) -> str:
    """E1's insertion: always the node types; the relations only when the host changed the paper's four."""
    lines = [f'Allowed node types (the "type" of every node must be one of these): {", ".join(graph.node_types)}']
    if set(graph.relations) != set(DEFAULT_RELATIONS):
        lines.append(f"Allowed relations: {', '.join(graph.relations)}")
    return "\n".join(lines)


def render_user_message(
    graph: Graph,
    *,
    task_description: str,
    mode: str,
    attempts_block: str,
    rejected_block: str,
    available_tools: Sequence[str] | None,
) -> str:
    import json

    tools = ", ".join(available_tools) if available_tools else TOOLS_UNSPECIFIED
    text = REFINER_USER_TEMPLATE.format(
        task_description=task_description,
        mode=mode,
        available_tools_list=tools,
        vocabulary_lines=vocabulary_lines(graph),
        attempts_block="\n" + attempts_block,
        current_graph_json="\n" + json.dumps(graph.to_refiner_json(), ensure_ascii=False, indent=1),
        rejected_block="\n" + rejected_block,
    )
    return text


@dataclass
class RefinerResult:
    edits: EditSet | None  # None when no reply parsed
    candidate: Graph | None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    calls: int = 0

    @property
    def outcome(self) -> str:
        """``candidate`` (structurally valid), ``no_action`` (empty edits) or ``structural_failure``."""
        if self.edits is not None and self.edits.is_empty:
            return "no_action"
        return "candidate" if self.candidate is not None else "structural_failure"


class Refiner:
    def __init__(self, model: ChatModel, config: EvolveConfig):
        self.model, self.config = model, config

    def build_request(
        self,
        graph: Graph,
        *,
        mode: str,
        attempts_block: str,
        rejected_block: str,
        available_tools: Sequence[str] | None = None,
        images: Sequence[ImagePart] = (),
        user_text: str | None = None,
    ) -> ModelRequest:
        text = user_text or render_user_message(
            graph,
            task_description=self.config.task_description,
            mode=mode,
            attempts_block=attempts_block,
            rejected_block=rejected_block,
            available_tools=available_tools,
        )
        parts: list[TextPart | ImagePart] = [TextPart(text), *images]
        return ModelRequest(role="refiner", system=REFINER_SYSTEM, parts=parts, max_tokens=self.config.refiner_max_tokens)

    async def propose(
        self,
        graph: Graph,
        *,
        mode: str,
        attempts_block: str,
        rejected_block: str,
        available_tools: Sequence[str] | None = None,
        images: Sequence[ImagePart] = (),
        retries: int | None = None,
    ) -> RefinerResult:
        """One refiner stage: call, parse, PrepareCandidate; on failure retry up to ``retries`` times (E2)."""
        retries = self.config.role_retries if retries is None else retries
        request = self.build_request(graph, mode=mode, attempts_block=attempts_block, rejected_block=rejected_block,
                                     available_tools=available_tools, images=images)
        original_text = request.text
        result = RefinerResult(edits=None, candidate=None)
        for attempt in range(retries + 1):
            response = await self.model.complete(request)
            result.calls += 1
            reply = response.text
            result.replies.append(reply)
            try:
                edits = EditSet.parse(reply, graph.attribute_fields)
            except EditError as exc:
                result.edits, result.candidate, result.diagnostics = None, None, list(exc.diagnostics)
            else:
                result.edits = edits
                if edits.is_empty:
                    result.candidate, result.diagnostics = None, []
                    return result  # no_action is not an error to retry (E3)
                candidate, diagnostics = prepare_candidate(graph, edits, cycle_policy=self.config.cycle_policy, available_tools=available_tools)
                result.candidate, result.diagnostics = candidate, diagnostics
                if candidate is not None:
                    return result
            if attempt < retries:
                folded = RETRY_TEMPLATE.format(
                    original=original_text,
                    reply=reply.strip() or "(empty reply)",
                    diagnostics="\n".join(f"- {d}" for d in errors(result.diagnostics)) or "- (no details)",
                )
                request = ModelRequest(role="refiner", system=REFINER_SYSTEM, parts=[TextPart(folded), *images], max_tokens=request.max_tokens)
        return result


__all__ = [
    "REFINER_SYSTEM",
    "REFINER_USER_TEMPLATE",
    "RETRY_TEMPLATE",
    "TOOLS_UNSPECIFIED",
    "Refiner",
    "RefinerResult",
    "render_user_message",
    "vocabulary_lines",
]
