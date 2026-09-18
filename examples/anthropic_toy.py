# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The filesystem toy with a real model as refiner and guidance model (REQUIREMENTS I4).

    ANTHROPIC_API_KEY=... uv run python examples/anthropic_toy.py ./toy-workspace-anthropic [rounds]

Same environment as ``filesystem_toy.py`` (two-hop QA over a tiny corpus, three tools), but the refiner is a real
Claude call in the paper's ``scratch_incremental`` mode starting from the Start → End skeleton, and the online
guidance is the paper's default ``generative_subgraph`` (a guidance call per solver step). The scripted solver reads
its cues from the generated guidance text (does it mention reading? searching?), so whether the model's graph helps
depends on whether its guidance says so. Defaults: two rounds, tie-accepting gate, cycle repair off for the same
reason as the filesystem toy. Requires ``pip install proceduralgraph[anthropic]``. ``PROCEDURALGRAPH_MODEL`` overrides
the model id.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filesystem_toy import TASK, TOOLS, ToyEnvironment  # noqa: E402

from proceduralgraph import (  # noqa: E402
    Budget,
    EvolveConfig,
    FileRevisionStore,
    Graph,
    GuidanceConfig,
    Guide,
    Hooks,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    RevisionTraceStore,
    evolve,
    export_workspace,
)
from proceduralgraph.adapters.anthropic import AnthropicChatModel  # noqa: E402


class Printing(Hooks):
    async def on_warning(self, message: str) -> None:
        print(f"warning: {message}")

    async def on_iteration(self, report) -> None:
        print(f"iteration {report.iteration}: {report.outcome}  evaluation={report.evaluation}")


async def main(root: Path, rounds: int) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("set ANTHROPIC_API_KEY to run this example", file=sys.stderr)
        return 2
    model = AnthropicChatModel(os.environ.get("PROCEDURALGRAPH_MODEL", "claude-sonnet-5"))

    def generative_guide(graph: Graph) -> Guide:
        return Guide(graph, model, GuidanceConfig(mode="generative_subgraph", hops=2, window=3, task_description=TASK))

    environment = ToyEnvironment(
        generative_guide,
        read_cue=lambda g: "read" in g.lower(),
        search_cue=lambda g: "search" in g.lower(),
    )
    revisions = FileRevisionStore(root / "revisions")
    report = await evolve(
        config=EvolveConfig(
            task_description=TASK,
            max_rounds=rounds,
            cycle_policy="allow",
            workspace="toy",
            budget=Budget(max_model_calls=400, max_seconds=900),
        ),
        model=model,
        graph_store=RevisionGraphStore(revisions, "toy"),
        rejection_store=RevisionRejectionStore(revisions, "toy"),
        trace_store=RevisionTraceStore(revisions, "toy"),
        checkpoint_store=RevisionCheckpointStore(revisions, "toy"),
        trace_source=environment,
        evaluator=environment,
        hooks=Printing(),
        available_tools=TOOLS,
    )
    print(json.dumps(report.to_dict(), indent=1, default=str))
    for path in export_workspace(root / "export", graph=report.graph, rejections=report.rejections):
        print(f"exported {path}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "./toy-workspace-anthropic")
    sys.exit(asyncio.run(main(target, int(sys.argv[2]) if len(sys.argv) > 2 else 2)))
