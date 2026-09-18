# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""``proceduralgraph``: initialise, inspect and export a workspace held in a revision store (H8).

    proceduralgraph --store file:./workspace --workspace default init --graph expert.json
    proceduralgraph --store file:./workspace --workspace default init --from-text problem.md --solutions ./solved --tools search,read,answer --model anthropic:claude-sonnet-5
    proceduralgraph --store file:./workspace --workspace default show
    proceduralgraph --store postgres:postgresql+asyncpg://user:pw@host/db --workspace tenant-a export ./out
    proceduralgraph --store file:./workspace guide --query "who wrote X?" --step search --step read

Only ``file:`` and ``postgres:`` stores are addressable from the command line; object-store adapters need credentials
a host configures in code. ``guide`` runs the serializer in ``raw_subgraph`` mode with no model, so an operator can
see exactly what the agent would be shown at a given point of a trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import CYCLE_POLICIES, EvolveConfig
from .edits import unified_diff
from .graph import Graph, GraphError
from .guidance import GuidanceConfig, Guide, steps_from_actions
from .model import ChatModel, ScriptedChatModel
from .roles.bootstrap import bootstrap_graph
from .stores.base import RevisionStore
from .stores.file import FileRevisionStore, export_workspace
from .stores.revision import CHAIN_KINDS, GRAPH, RevisionGraphStore, RevisionRejectionStore
from .stores.transfer import seed_workspace

_HEX64 = re.compile(r"[0-9a-f]{64}")


class CliError(Exception):
    pass


def open_store(spec: str) -> RevisionStore:
    scheme, _, target = spec.partition(":")
    if scheme == "file" and target:
        return FileRevisionStore(target)
    if scheme == "postgres" and target:
        try:
            from .stores.postgres import PostgresRevisionStore
        except ImportError as exc:  # pragma: no cover - the module itself has no hard import
            raise CliError("the postgres store needs the extra: pip install 'proceduralgraph[postgres]'") from exc
        try:
            return PostgresRevisionStore.from_url(target)
        except ModuleNotFoundError as exc:  # sqlalchemy or the driver (asyncpg / psycopg) missing
            raise CliError(f"the postgres store needs the extra and a driver: pip install 'proceduralgraph[postgres]' ({exc})") from exc
        except Exception as exc:  # a malformed URL: sqlalchemy's ArgumentError and friends
            raise CliError(f"cannot open {target!r}: {exc}. Expected postgres:postgresql+asyncpg://user:pw@host/db") from exc
    raise CliError(f"unsupported store {spec!r}: use file:DIR or postgres:URL")


def open_model(spec: str) -> ChatModel:
    """``anthropic:MODEL_ID``, ``openai:MODEL_ID`` (each needs its extra), or ``scripted:FILE`` (a JSON list of replies,
    for offline demos and tests)."""
    scheme, _, target = spec.partition(":")
    if scheme in ("anthropic", "openai") and target:
        try:
            if scheme == "anthropic":
                from .adapters.anthropic import AnthropicChatModel as Adapter
            else:
                from .adapters.openai import OpenAIChatModel as Adapter
        except ModuleNotFoundError as exc:
            raise CliError(f"the {scheme} model needs the extra: pip install 'proceduralgraph[{scheme}]'") from exc
        try:
            return Adapter(target)
        except Exception as exc:  # the SDK refuses to construct: usually a missing API key in the environment
            raise CliError(f"cannot create the {scheme} client for {target!r}: {exc}") from exc
    if scheme == "scripted" and target:
        try:
            replies = json.loads(Path(target).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CliError(f"cannot read scripted replies from {target}: {exc}") from exc
        if not isinstance(replies, list) or not all(isinstance(r, str) for r in replies):
            raise CliError(f"{target} must hold a JSON list of reply strings")
        return ScriptedChatModel(replies)
    raise CliError(f"unsupported model {spec!r}: use anthropic:MODEL, openai:MODEL or scripted:FILE")


async def _heads(store: RevisionStore, workspace: str):
    graph_ref, graph = await RevisionGraphStore(store, workspace).load()
    rejections_ref, memory = await RevisionRejectionStore(store, workspace).load()
    return graph_ref, graph, rejections_ref, memory


async def cmd_init(store: RevisionStore, args: argparse.Namespace) -> str:
    graph_store = RevisionGraphStore(store, args.workspace)
    ref, _ = await graph_store.load()
    if ref is not None:
        raise CliError(f"workspace {args.workspace!r} already has a graph ({ref[:12]}); use a new workspace or `transfer`")
    notes = []
    extra_meta: dict = {}
    if args.graph and args.from_text:
        raise CliError("use either --graph FILE or --from-text FILE, not both")
    if args.graph:
        try:
            graph, notes = Graph.from_human(Path(args.graph))
        except (GraphError, OSError, ValueError) as exc:
            raise CliError(f"cannot load {args.graph}: {exc}") from exc
        origin = "skeleton" if graph.is_skeleton else "onboarded"
    elif args.from_text:
        problem, solutions, tools = _bootstrap_inputs(args)
        model = open_model(args.model)
        config = EvolveConfig(role_retries=args.retries, cycle_policy=args.cycle_policy)
        try:
            result = await bootstrap_graph(problem_statement=problem, model=model, solutions=solutions, tools=tools, config=config)
        except ValueError as exc:  # empty problem statement, or neither tools nor solutions
            raise CliError(str(exc)) from exc
        except Exception as exc:  # the provider failed (auth, network, quota)
            raise CliError(f"the model call failed: {type(exc).__name__}: {exc}") from exc
        if result.graph is None:
            raise CliError(result.refusal())
        graph, origin = result.graph, result.seed_meta()["origin"]
        notes = [d for d in result.diagnostics if not d.is_error]
        extra_meta = {k: v for k, v in result.seed_meta().items() if k != "origin"}
    else:
        if args.solutions or args.tools or args.model:
            raise CliError("--solutions, --tools and --model only apply with --from-text")
        graph, origin = Graph.skeleton(), "skeleton"
    digest = await graph_store.seed(graph, meta={"origin": origin, **extra_meta})
    lines = [f"{args.workspace}: seeded graph {digest[:12]} ({origin}) with {len(graph.nodes)} node(s), {len(graph.edges)} edge(s)"]
    if origin == "bootstrapped":
        lines.append(f"bootstrapped from {args.from_text} with {extra_meta['solutions']} worked solution(s); "
                     "this is a starting point, not a validated graph: the first evolve run scores it as the baseline")
    lines += [f"warning: {note}" for note in notes]
    return "\n".join(lines)


def _bootstrap_inputs(args: argparse.Namespace) -> tuple[str, list[str], list[str] | None]:
    """Read the problem statement, the solutions directory and the tool list for ``init --from-text``; every failure is a
    plain ``CliError``."""
    if not args.model:
        raise CliError("--from-text needs --model (anthropic:MODEL, openai:MODEL or scripted:FILE)")
    try:
        problem = Path(args.from_text).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"cannot read the problem statement {args.from_text}: {exc}") from exc
    if not problem.strip():
        raise CliError(f"the problem statement {args.from_text} is empty")
    solutions: list[str] = []
    if args.solutions:
        folder = Path(args.solutions)
        if not folder.is_dir():
            raise CliError(f"--solutions must be a directory of text files: {folder}")
        for path in sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")):
            try:
                solutions.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                raise CliError(f"cannot read the solution {path}: {exc} (solutions must be UTF-8 text files)") from exc
        if not solutions:
            raise CliError(f"--solutions directory {folder} holds no text files")
    tools = [t.strip() for chunk in (args.tools or []) for t in chunk.split(",") if t.strip()] or None
    if not solutions and not tools:
        raise CliError("--from-text without --solutions needs --tools: the refiner has nothing to build ACTION nodes from")
    return problem, solutions, tools


async def cmd_show(store: RevisionStore, args: argparse.Namespace) -> str:
    graph_ref, graph, rejections_ref, memory = await _heads(store, args.workspace)
    head = await store.head(args.workspace, GRAPH)
    score = head.meta.get("validation_score") if head else None
    lines = [
        f"workspace: {args.workspace}",
        f"graph: {graph_ref[:12] if graph_ref else '(empty)'}  {len(graph.nodes)} node(s)  {len(graph.edges)} edge(s)"
        + (f"  origin {head.meta.get('origin')}" if head else "")
        + (f"  last validation {score}" if score is not None else ""),
        f"rejections: {rejections_ref[:12] if rejections_ref else '(empty)'}  {len(memory.entries)} entr{'y' if len(memory.entries) == 1 else 'ies'}"
        f"  through iteration {memory.iteration}  " + ", ".join(f"{v} {k}" for k, v in memory.counts().items() if v),
        "",
        "## Nodes",
    ]
    for node in graph.nodes.values():
        lines.append(f"- {node.id} ({node.type}): {node.description or '(no description)'}")
    lines += ["", "## Edges"]
    for edge in graph.edges:
        lines.append(f"- {edge.source} -{edge.relation}-> {edge.target}" + (f"  [{edge.condition}]" if edge.condition else ""))
    lines += ["", f"## Rejection memory (last {args.tail})"]
    lines += [f"- {line}" for line in memory.last_lines(args.tail)] or ["(none)"]
    return "\n".join(lines)


async def cmd_export(store: RevisionStore, args: argparse.Namespace) -> str:
    _, graph, _, memory = await _heads(store, args.workspace)
    written = export_workspace(args.directory, graph=graph, rejections=memory)
    return "\n".join(str(path) for path in written)


async def cmd_history(store: RevisionStore, args: argparse.Namespace) -> str:
    kinds = [args.kind] if args.kind else list(CHAIN_KINDS)
    lines = []
    for kind in kinds:
        for revision in await store.list(args.workspace, kind, limit=args.limit):
            meta = {k: (v[:12] if isinstance(v, str) and _HEX64.fullmatch(v) else v) for k, v in revision.meta.items() if k not in ("edits", "decision")}
            if "edits" in revision.meta:
                meta["edits"] = {k: len(v) for k, v in revision.meta["edits"].items()}
            if isinstance(revision.meta.get("decision"), dict):
                meta["decision"] = revision.meta["decision"].get("disposition")
            rendered = json.dumps(meta, sort_keys=True, default=str) if meta else ""
            lines.append(f"{kind:<12} #{revision.seq:<4} {revision.digest[:12]}  {revision.created_at}  {rendered}"[:300])
    return "\n".join(lines) if lines else "(no revisions)"


async def cmd_diff(store: RevisionStore, args: argparse.Namespace) -> str:
    revisions = await store.list(args.workspace, GRAPH, limit=1_000_000)  # newest first
    if not revisions:
        raise CliError("no graph in this workspace")
    by_digest = {r.digest: r for r in revisions}

    def pick(ref: str | None, default_index: int):
        if ref is None:
            return revisions[default_index] if default_index < len(revisions) else None
        matches = [r for r in revisions if str(r.seq) == ref] if ref.isdigit() else []  # a bare number is a seq first
        matches = matches or [r for d, r in by_digest.items() if d.startswith(ref)]
        if len(matches) != 1:
            raise CliError(f"graph revision {ref!r} is {'ambiguous' if matches else 'unknown'}")
        return matches[0]

    after = pick(args.to, 0)
    before = pick(getattr(args, "from"), 1)
    before_graph = Graph.from_document(before.document) if before else Graph.skeleton()
    after_graph = Graph.from_document(after.document)
    head = (f"--- graph #{before.seq} {before.digest[:12]}\n+++ graph #{after.seq} {after.digest[:12]}\n" if before
            else f"+++ graph #{after.seq} {after.digest[:12]}\n")
    return head + (unified_diff(before_graph, after_graph) or "(no change)")


async def cmd_rejections(store: RevisionStore, args: argparse.Namespace) -> str:
    _, _, _, memory = await _heads(store, args.workspace)
    if args.refiner_view:
        return memory.render_for_refiner(full_entries=args.full)
    return memory.render_markdown()


async def cmd_transfer(store: RevisionStore, args: argparse.Namespace) -> str:
    digest = await seed_workspace(store, source=args.workspace, target=args.to)
    if digest is None:
        raise CliError(f"nothing transferred: {args.workspace!r} has no graph or {args.to!r} already has one")
    _, graph = await RevisionGraphStore(store, args.to).load()
    return f"{args.to}: graph {digest[:12]}  {len(graph.nodes)} node(s)  {len(graph.edges)} edge(s)  (from {args.workspace})"


async def cmd_guide(store: RevisionStore, args: argparse.Namespace) -> str:
    _, graph, _, _ = await _heads(store, args.workspace)
    mode = "raw_full" if args.full else "raw_subgraph"
    guide = Guide(graph, None, GuidanceConfig(mode=mode, hops=args.hops, include_relations=not args.paper_layout))
    result = await guide.guidance(args.query, steps_from_actions(args.step or []))
    head = f"localized at: {result.node_id or '(no match: full graph)'}" + ("  [full graph]" if result.used_full_graph else "")
    return head + "\n\n" + result.text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proceduralgraph", description="Initialise, inspect and export a proceduralgraph workspace.")
    parser.add_argument("--store", required=True, help="file:DIR or postgres:URL")
    parser.add_argument("--workspace", default="default")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="seed an empty workspace: from a hand-written graph JSON, from a problem statement via a model, or the Start → End skeleton")
    init.add_argument("--graph", help="path to a graph JSON in the refiner shape ({nodes, edges})")
    init.add_argument("--from-text", help="path to a problem statement; the refiner bootstraps a graph from it (needs --model)")
    init.add_argument("--solutions", help="directory of worked-solution text files, each presented to the refiner as a successful trajectory")
    init.add_argument("--tools", action="append", help="tool / action names the agent can execute; repeatable or comma-separated")
    init.add_argument("--model", help="anthropic:MODEL, openai:MODEL or scripted:FILE (a JSON list of replies)")
    init.add_argument("--retries", type=int, default=1, help="refiner retries with diagnostics fed back (0 = paper-exact)")
    init.add_argument("--cycle-policy", choices=CYCLE_POLICIES, default="repair", help="PrepareCandidate's cycle policy for the bootstrap")
    init.set_defaults(run=cmd_init)
    show = sub.add_parser("show", help="node/edge counts, head digest, last validation score, recent rejection lines")
    show.add_argument("--tail", type=int, default=10)
    show.set_defaults(run=cmd_show)
    export = sub.add_parser("export", help="write graph.json, graph.md, graph.mmd and rejections.md to a directory")
    export.add_argument("directory")
    export.set_defaults(run=cmd_export)
    history = sub.add_parser("history", help="revision chains, newest first")
    history.add_argument("--kind", choices=CHAIN_KINDS)
    history.add_argument("--limit", type=int, default=20)
    history.set_defaults(run=cmd_history)
    diff = sub.add_parser("diff", help="unified diff between two accepted graphs (default: previous head vs head)")
    diff.add_argument("--from", dest="from", help="digest prefix or seq")
    diff.add_argument("--to", help="digest prefix or seq")
    diff.set_defaults(run=cmd_diff)
    rejections = sub.add_parser("rejections", help="the rejection memory (operator markdown, or --refiner-view)")
    rejections.add_argument("--full", type=int, default=5, help="entries rendered in full in the refiner view")
    rejections.add_argument("--refiner-view", action="store_true", help="render exactly what the refiner is shown")
    rejections.set_defaults(run=cmd_rejections)
    transfer = sub.add_parser("transfer", help="copy this workspace's head graph into an empty workspace")
    transfer.add_argument("--to", required=True, help="target workspace")
    transfer.set_defaults(run=cmd_transfer)
    guide = sub.add_parser("guide", help="show the serialized graph context the agent would be given (no model call)")
    guide.add_argument("--query", required=True)
    guide.add_argument("--step", action="append", help="an action in the trajectory, in order; repeatable")
    guide.add_argument("--hops", type=int, default=2)
    guide.add_argument("--full", action="store_true", help="the complete graph instead of the localized subgraph")
    guide.add_argument("--paper-layout", action="store_true", help="omit relation labels, exactly as the paper's serializer")
    guide.set_defaults(run=cmd_guide)
    return parser


async def _run(store: RevisionStore, args: argparse.Namespace) -> str:
    try:
        return await args.run(store, args)
    finally:
        engine = getattr(store, "engine", None)  # the postgres store owns an async engine bound to this loop
        if engine is not None:
            await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = open_store(args.store)
        print(asyncio.run(_run(store, args)))
    except CliError as exc:
        print(f"proceduralgraph: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
