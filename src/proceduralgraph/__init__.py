# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""proceduralgraph: self-evolving Procedural Graphs for LLM agents.

Independent reimplementation of *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents*
(Lu, Chen, Wu, Arık; arXiv:2609.09153): an online, per-step guidance runtime (``Guide``) a host calls inside its
agent loop, and an offline, gated self-evolution loop (``evolve``) with a persistent rejection memory. Storage-agnostic,
provider-agnostic, budgeted, resumable. See README.md.
"""

from .config import Budget, EvolveConfig
from .documents import SCHEMA_VERSION, HeadMoved, Revision, canonical_json, digest
from .edits import EditError, EditSet, prepare_candidate, repair_cycles, unified_diff
from .gates import Decision, Evaluation, Evaluator, Gate, PairedGate, StrictImprovementGate, TieAcceptingGate
from .graph import (
    DEFAULT_ATTRIBUTE_FIELDS,
    DEFAULT_NODE_TYPES,
    DEFAULT_RELATIONS,
    END,
    START,
    Diagnostic,
    Edge,
    Graph,
    GraphError,
    Neighborhood,
    Node,
)
from .guidance import ExactActionLocalizer, Guidance, GuidanceConfig, Guide, Localizer
from .harness import RunReport, evolve, evolve_sync, refine_once
from .hooks import BudgetExceeded, HookedModel, Hooks, IterationReport
from .model import (
    ChatModel,
    ImagePart,
    ModelRequest,
    ModelResponse,
    ScriptedChatModel,
    TextPart,
    ToolInvocation,
    ToolSpec,
)
from .redaction import EMAIL, Redactor, redact_trace, regex_redactor
from .rejections import RejectionEntry, RejectionMemory
from .roles.refiner import Refiner
from .serialize import render_markdown, render_mermaid, serialize_context
from .stores import (
    Candidate,
    CheckpointStore,
    FileRevisionStore,
    GraphStore,
    MemoryRevisionStore,
    NullCheckpointStore,
    NullTraceStore,
    RejectionStore,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    RevisionStore,
    RevisionTraceStore,
    TraceStore,
    export_workspace,
    seed_workspace,
)
from .traces import (
    StaticTraceSource,
    Step,
    StridedTraceSource,
    TaskOutcome,
    Trace,
    TraceSource,
    render_attempts_block,
    stratified_sample,
    summarize_node_usage,
    tail,
)

__version__ = "0.1.0"
__author__ = "Vikash Ranjan (CTO, styls.ai)"

__all__ = [
    "DEFAULT_ATTRIBUTE_FIELDS",
    "DEFAULT_NODE_TYPES",
    "DEFAULT_RELATIONS",
    "EMAIL",
    "END",
    "SCHEMA_VERSION",
    "START",
    "Budget",
    "BudgetExceeded",
    "Candidate",
    "ChatModel",
    "CheckpointStore",
    "Decision",
    "Diagnostic",
    "Edge",
    "EditError",
    "EditSet",
    "Evaluation",
    "Evaluator",
    "EvolveConfig",
    "ExactActionLocalizer",
    "FileRevisionStore",
    "Gate",
    "Graph",
    "GraphError",
    "GraphStore",
    "Guidance",
    "GuidanceConfig",
    "Guide",
    "HeadMoved",
    "HookedModel",
    "Hooks",
    "ImagePart",
    "IterationReport",
    "Localizer",
    "MemoryRevisionStore",
    "ModelRequest",
    "ModelResponse",
    "Neighborhood",
    "Node",
    "NullCheckpointStore",
    "NullTraceStore",
    "PairedGate",
    "Redactor",
    "Refiner",
    "RejectionEntry",
    "RejectionMemory",
    "RejectionStore",
    "Revision",
    "RevisionCheckpointStore",
    "RevisionGraphStore",
    "RevisionRejectionStore",
    "RevisionStore",
    "RevisionTraceStore",
    "RunReport",
    "ScriptedChatModel",
    "StaticTraceSource",
    "Step",
    "StridedTraceSource",
    "StrictImprovementGate",
    "TaskOutcome",
    "TextPart",
    "TieAcceptingGate",
    "ToolInvocation",
    "ToolSpec",
    "Trace",
    "TraceSource",
    "TraceStore",
    "__author__",
    "__version__",
    "canonical_json",
    "digest",
    "evolve",
    "evolve_sync",
    "export_workspace",
    "prepare_candidate",
    "redact_trace",
    "refine_once",
    "regex_redactor",
    "render_attempts_block",
    "render_markdown",
    "render_mermaid",
    "repair_cycles",
    "seed_workspace",
    "serialize_context",
    "stratified_sample",
    "summarize_node_usage",
    "tail",
    "unified_diff",
]
