# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The whole loop, offline, with no API key and no network (REQUIREMENTS I4).

    uv run python examples/filesystem_toy.py ./toy-workspace

A toy two-hop question-answering environment with three tools, ``search(q)``, ``read(doc)`` and ``answer(text)``.
The scripted solver answers straight after its first search unless the graph-derived guidance tells it to read and,
when a bridge entity turns up, to search again: two-hop questions fail without the graph and pass with it.

A scripted refiner then drives four rounds of Algorithm 1 against a ``FileRevisionStore``:

1. accepted: builds ``Start → search → read → search → answer → End`` (validation 0.5 → 1.0);
2. structural failure: an edge to a node that does not exist (never evaluated);
3. rejected: deletes the ``read → search`` transition, so two-hop questions fail again (1.0 → 0.5);
4. duplicate candidate: proposes the same deletion, refused without paying for validation.

The example uses ``role_retries=0`` (the paper's behaviour) so each scripted reply is consumed by exactly one round,
and ``cycle_policy="allow"`` because the toy's node ids are its tool names and the second search must revisit the
``search`` node (the paper's HotpotQA graph uses distinct per-hop node ids instead). It prints the ``RunReport`` and
exports ``graph.json``, ``graph.md``, ``graph.mmd`` and ``rejections.md`` for you to open.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

from proceduralgraph import (
    Evaluation,
    EvolveConfig,
    FileRevisionStore,
    Graph,
    GuidanceConfig,
    Guide,
    RevisionCheckpointStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    RevisionTraceStore,
    ScriptedChatModel,
    Step,
    TaskOutcome,
    Trace,
    evolve,
    export_workspace,
)

# --- the toy environment -----------------------------------------------------------------------------------------------

CORPUS = {
    # title: (text, fact the document answers with, bridge entity it links to)
    "Alice Martin": ("Alice Martin is a painter who was born in Lyon.", "Alice Martin is a painter", "Lyon"),
    "Lyon": ("Lyon is a city in France on the Rhône.", "France", None),
    "Bram Okafor": ("Bram Okafor founded Okafor Motors in Kumasi.", "Bram Okafor founded Okafor Motors", "Kumasi"),
    "Kumasi": ("Kumasi is the capital of the Ashanti Region of Ghana.", "Ghana", None),
    "Chen Wei": ("Chen Wei composed the opera Silver River.", "Chen Wei composed Silver River", None),
    "Dara Quinn": ("Dara Quinn is a marathon runner from Galway.", "Dara Quinn is a marathon runner", "Galway"),
    "Galway": ("Galway is a harbour city in Ireland.", "Ireland", None),
    "Eli Sato": ("Eli Sato wrote the novel Paper Lanterns.", "Eli Sato wrote Paper Lanterns", None),
}

TRAIN = [
    ("Who is Alice Martin?", "Alice Martin is a painter"),
    ("In which country was Alice Martin born?", "France"),
    ("What did Bram Okafor do?", "Bram Okafor founded Okafor Motors"),
    ("Which country is Bram Okafor's company based in?", "Ghana"),
]
VALIDATION = [
    ("What did Chen Wei compose?", "Chen Wei composed Silver River"),
    ("Which country is Dara Quinn from?", "Ireland"),
    ("What did Eli Sato write?", "Eli Sato wrote Paper Lanterns"),
    ("Where is Dara Quinn's home city?", "Ireland"),
]

TOOLS = ["search", "read", "answer"]
TASK = "two-hop question answering over a small corpus"


def search(query: str) -> list[str]:
    words = {w.strip("?.,'s").lower() for w in query.split()}
    return [title for title in CORPUS if any(part.lower() in words for part in title.split())]


def read(title: str) -> str:
    return CORPUS[title][0]


ReadCue = Callable[[str], bool]


class ToyEnvironment:
    """The scripted ReAct solver (eq. 3) plus the two host protocols built on it (TraceSource, Evaluator).

    ``guide_factory`` builds the per-episode :class:`Guide` (raw context here, a generative model in
    ``anthropic_toy.py``). ``read_cue`` / ``search_cue`` decide, from the guidance text g_t, whether the solver reads
    the top document and whether it searches the bridge entity: with raw context the cues are the serialized
    transitions; with generated guidance they are the words themselves.
    """

    def __init__(self, guide_factory: Callable[[Graph], Guide], *, read_cue: ReadCue, search_cue: ReadCue):
        self.guide_factory, self.read_cue, self.search_cue = guide_factory, read_cue, search_cue

    async def solve(self, question: str, guide: Guide) -> tuple[str, list[Step]]:
        steps: list[Step] = []
        await guide.guidance(question, steps)
        hits = search(question)
        steps.append(Step("search", {"q": question}, ", ".join(hits) or "(no results)"))
        if not hits:
            steps.append(Step("answer", "unknown", ""))
            return "unknown", steps
        current = hits[0]
        g = await guide.guidance(question, steps)
        if not self.read_cue(g.text):
            fact = CORPUS[current][1]
            steps.append(Step("answer", fact, ""))
            return fact, steps
        steps.append(Step("read", current, read(current)))
        g = await guide.guidance(question, steps)
        bridge = CORPUS[current][2]
        if bridge and self.search_cue(g.text):
            hits = search(bridge)
            steps.append(Step("search", {"q": bridge}, ", ".join(hits)))
            current = hits[0]
            await guide.guidance(question, steps)
            steps.append(Step("read", current, read(current)))
            await guide.guidance(question, steps)
        fact = CORPUS[current][1]
        steps.append(Step("answer", fact, ""))
        return fact, steps

    async def episode(self, question: str, expected: str, graph: Graph, graph_ref: str | None, trace_id: str) -> Trace:
        guide = self.guide_factory(graph)
        answer, steps = await self.solve(question, guide)
        passed = answer == expected
        return Trace(
            id=trace_id,
            outcome=TaskOutcome(task_id=question, score=1.0 if passed else 0.0, passed=passed, prediction=answer, truth=expected),
            text="\n".join(s.rendered() for s in steps),
            steps=steps,
            graph_ref=graph_ref,
            nodes_visited=guide.visited,
            guidance=[g.text for g in guide.guidance_log],
        )

    # TraceSource (Alg. 1 line 6)
    async def collect(self, graph: Graph, *, graph_ref: str | None, iteration: int) -> list[Trace]:
        return [await self.episode(q, a, graph, graph_ref, f"it{iteration}-train{i}") for i, (q, a) in enumerate(TRAIN)]

    # Evaluator (Alg. 1 lines 1 and 15)
    async def evaluate(self, ref: str | None, graph: Graph, *, iteration: int, purpose: str) -> Evaluation:
        traces = [await self.episode(q, a, graph, ref, f"val{i}") for i, (q, a) in enumerate(VALIDATION)]
        per_task = [t.outcome for t in traces]
        score = sum(o.score for o in per_task) / len(per_task)
        return Evaluation(ref=ref, score=score, aggregate={"purpose": purpose, "iteration": iteration}, per_task=per_task)


def raw_guide(graph: Graph) -> Guide:
    """raw_subgraph: the serialized N_h(u_t) is the guidance; no model call. ``anthropic_toy.py`` swaps in a ChatModel
    and generative_subgraph for the paper's default."""
    return Guide(graph, None, GuidanceConfig(mode="raw_subgraph", hops=2, task_description=TASK))


def raw_environment() -> ToyEnvironment:
    return ToyEnvironment(raw_guide, read_cue=lambda g: "→ [read]" in g, search_cue=lambda g: "→ [search]" in g)


# --- the scripted refiner --------------------------------------------------------------------------------------------------


def edge(source: str, target: str, guidance: str, *, relation: str = "LEADS_TO", condition: str | None = None, pitfalls: str | None = None) -> dict:
    return {"source": source, "target": target, "relation": relation, "condition": condition, "guidance": guidance, "pitfalls": pitfalls}


def edits(**arrays) -> str:
    return json.dumps({"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": [], **arrays})


ROUND_1_BUILD = edits(
    add_nodes=[
        {"id": "search", "type": "ACTION", "description": "Search the corpus for documents matching a query."},
        {"id": "read", "type": "ACTION", "description": "Read one document in full."},
        {"id": "answer", "type": "ACTION", "description": "Give the final answer."},
    ],
    delete_edges=[{"source": "Start", "target": "End"}],
    add_edges=[
        edge("Start", "search", "Search for the entity named in the question first."),
        edge("search", "read", "Read the top result before answering; the question may need a fact from a linked document.",
             condition="search returned at least one document", pitfalls="Do not answer from a title alone."),
        edge("read", "search", "If the document names a bridge entity (a city, a company), search for it to reach the second hop.",
             relation="PROVIDES_INPUT_FOR", condition="the document links to another entity the question asks about",
             pitfalls="Do not loop: one bridge search is enough."),
        edge("read", "answer", "Answer with the fact stated in the document you just read.", condition="the document answers the question"),
        edge("answer", "End", "Stop after answering."),
    ],
)
ROUND_2_BROKEN = edits(add_edges=[edge("answer", "verify_answer", "Verify before finishing.")])  # verify_answer does not exist
ROUND_3_DROP_SECOND_HOP = edits(delete_edges=[{"source": "read", "target": "search"}])
ROUND_4_DUPLICATE = ROUND_3_DROP_SECOND_HOP


async def main(root: Path) -> int:
    revisions = FileRevisionStore(root / "revisions")
    workspace = "toy"
    environment = raw_environment()
    report = await evolve(
        config=EvolveConfig(
            task_description=TASK,
            max_rounds=4,
            role_retries=0,  # paper-exact: a structural failure is recorded, not retried
            cycle_policy="allow",  # the toy revisits `search`; see the module docstring
            workspace=workspace,
        ),
        model=ScriptedChatModel([ROUND_1_BUILD, ROUND_2_BROKEN, ROUND_3_DROP_SECOND_HOP, ROUND_4_DUPLICATE]),
        graph_store=RevisionGraphStore(revisions, workspace),
        rejection_store=RevisionRejectionStore(revisions, workspace),
        trace_store=RevisionTraceStore(revisions, workspace),
        checkpoint_store=RevisionCheckpointStore(revisions, workspace),
        trace_source=environment,
        evaluator=environment,
        available_tools=TOOLS,
    )
    print(json.dumps(report.to_dict(), indent=1, default=str))
    written = export_workspace(root / "export", graph=report.graph, rejections=report.rejections)
    print("\nexported:")
    for path in written:
        print(f"  {path}")
    print(f"\noutcomes: {report.outcomes()}")
    print(f"validation: baseline {report.rejections.entries[0].retained_score} -> retained {report.best.score}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "./toy-workspace")
    sys.exit(asyncio.run(main(target)))
