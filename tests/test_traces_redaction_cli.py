# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""D / D6 / H8: the refiner context (Tail, ordering), stratified sampling, redaction and the command line.
Acceptance §8.7."""

import asyncio
import json

from proceduralgraph import (
    EMAIL,
    EvolveConfig,
    FileRevisionStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    Step,
    StridedTraceSource,
    TaskOutcome,
    Trace,
    evolve,
    redact_trace,
    regex_redactor,
    render_attempts_block,
    stratified_sample,
    summarize_node_usage,
    tail,
)
from proceduralgraph.cli import main

from .test_harness_end_to_end import BUILD, ScoringEvaluator, edge, edits, traces

# The CLI test initialises the workspace with a graph that already has ``search``; these edits extend it.
EXTEND = edits(
    add_nodes=[{"id": "read", "type": "ACTION", "description": "r"}, {"id": "answer", "type": "ACTION", "description": "a"}],
    add_edges=[edge("search", "read"), edge("read", "answer"), edge("answer", "End")],
)


def trace(id_, score, text, **extra):
    return Trace(id_, TaskOutcome(f"task-{id_}", score, score >= 0.5), text, **extra)


def test_tail_keeps_the_end_in_characters_and_tokens():
    text = "A" * 50 + "B" * 50
    assert tail(text, 200) == text
    cut = tail(text, 30)
    assert cut.endswith("B" * 30) and "70 earlier characters discarded" in cut
    words = " ".join(f"w{i}" for i in range(100))
    by_tokens = tail(words, 10, token_counter=lambda s: len(s.split()))
    assert by_tokens.endswith("w99") and len(by_tokens.split("\n", 1)[1].split()) <= 10
    assert tail(text, 0) == ""


def test_attempts_block_puts_failing_traces_last_and_the_tail_keeps_them():
    """§8.7: the tail keeps the end of the concatenation, and failing traces sit at the end."""
    passing = [trace("p1", 1.0, "P" * 400), trace("p2", 0.9, "Q" * 400)]
    failing = [trace("f1", 0.0, "F" * 400), trace("f2", 0.2, "G" * 400)]
    block = render_attempts_block(passing + failing, cap=10_000)
    order = [block.index(f"=== Trace {t} ") for t in ("p1", "p2", "f2", "f1")]
    assert order == sorted(order)  # high scores first (descending), low scores last (descending)
    assert block.startswith("4 trajectories: 2 high-scoring (score >= 0.5), 2 low-scoring.")
    cut = render_attempts_block(passing + failing, cap=1_000)
    assert "=== Trace f1 " in cut and "=== Trace p1 " not in cut and "FAILED" in cut
    assert cut.startswith("[... ") and len(cut) <= 1_000 + 60
    # non-binary scores partition on success_threshold
    block = render_attempts_block([trace("m", 0.6, "x"), trace("n", 0.4, "y")], cap=10_000, success_threshold=0.7)
    assert block.startswith("2 trajectories: 0 high-scoring (score >= 0.7), 2 low-scoring.")


def test_trace_rendering_steps_and_node_usage():
    t = Trace("s", TaskOutcome("q", 0.0, False), "", steps=[Step("search", {"q": "x"}, "3 hits"), Step("answer", "B", "wrong")],
              nodes_visited=["search", "answer"])
    text = t.rendered()
    assert text.startswith("=== Trace s | task q | score 0.000 | FAILED | nodes: search → answer ===")
    assert "Action: search({'q': 'x'})\nObservation: 3 hits" in text
    usage = summarize_node_usage([t, Trace("ok", TaskOutcome("q2", 1.0, True), "", nodes_visited=["search", "read"])])
    assert "| search | 1 | 1 |" in usage and "| answer | 1 | 0 |" in usage and "| read | 0 | 1 |" in usage
    again = Trace.from_document(t.to_document())
    assert again.steps[0].args == {"q": "x"} and again.nodes_visited == ["search", "answer"]


def test_stratified_sample_none_means_everything_and_rotation_otherwise():
    pool = [trace(f"f{i}", 0.0, "f") for i in range(6)] + [trace(f"p{i}", 1.0, "p") for i in range(4)]
    everything, seen = stratified_sample(pool, failing=None, passing=None)
    assert len(everything) == 10 and len(seen) == 10
    first, seen = stratified_sample(pool, failing=2, passing=1, seed=1, iteration=1)
    second, seen = stratified_sample(pool, failing=2, passing=1, seen=seen, seed=1, iteration=2)
    assert len(first) == 3 and len(second) == 3 and not ({t.id for t in first} & {t.id for t in second})


async def test_strided_source_wraps():
    pool = [trace(f"t{i}", 1.0, "x") for i in range(5)]
    source = StridedTraceSource(pool, stride=2)
    assert [t.id for t in await source.collect(None, graph_ref=None, iteration=1)] == ["t0", "t1"]
    assert [t.id for t in await source.collect(None, graph_ref=None, iteration=3)] == ["t4", "t0"]


def test_redaction_covers_text_and_observations():
    redactor = regex_redactor({"email": EMAIL, "order": r"ORD-\d+"})
    t = Trace("r", TaskOutcome("q", 1.0, True), "mail a@b.co about ORD-12", steps=[Step("lookup", "ORD-12", "customer c@d.org")],
              guidance=["next, email e@f.org"])
    clean = redact_trace(t, redactor)
    assert clean.text == "mail [REDACTED:email] about [REDACTED:order]" and clean.steps[0].observation == "customer [REDACTED:email]"
    assert clean.guidance == ["next, email [REDACTED:email]"]
    assert clean.steps[0].args == "ORD-12"  # structured fields are the host's job
    assert t.text.startswith("mail a@b.co")  # original untouched
    assert redact_trace(t, lambda s: s) is t


async def test_redaction_applies_before_the_refiner_and_the_store(tmp_path):
    revisions = FileRevisionStore(tmp_path / "rev")
    model = ScriptedChatModel([BUILD])
    dirty = [Trace("d", TaskOutcome("q", 0.0, False), "the reviewer x@y.io failed it", graph_ref=None)]
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=model, graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(dirty),
                          evaluator=ScoringEvaluator(), redact=regex_redactor({"email": EMAIL}))
    assert report.outcomes() == ["accepted"]
    assert "x@y.io" not in model.calls[0].text and "[REDACTED:email]" in model.calls[0].text


def test_cli_commands(tmp_path, capsys):
    root = tmp_path / "store"
    store = f"file:{root}"
    # init from a hand-written graph, then run one accepted round against the same store
    expert = {"nodes": [{"id": "search", "type": "ACTION", "description": "Search"}],
              "edges": [{"source": "Start", "target": "search", "relation": "LEADS_TO", "guidance": "go"}]}
    path = tmp_path / "expert.json"
    path.write_text(json.dumps(expert))
    assert main(["--store", store, "--workspace", "ws", "init", "--graph", str(path)]) == 0
    out = capsys.readouterr().out
    assert "seeded graph" in out and "skeleton_completed" in out
    assert main(["--store", store, "--workspace", "ws", "init"]) == 2
    assert "already has a graph" in capsys.readouterr().err
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"nodes": [{"id": "x", "type": "ACTION"}], "edges": [{"source": "x", "target": "nowhere", "relation": "LEADS_TO"}]}))
    assert main(["--store", store, "--workspace", "ws-broken", "init", "--graph", str(broken)]) == 2
    err = capsys.readouterr().err
    assert "cannot load" in err and "missing_endpoint" in err
    revisions = FileRevisionStore(root)
    report = asyncio.run(evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([EXTEND]),
                                graph_store=RevisionGraphStore(revisions, "ws"), rejection_store=RevisionRejectionStore(revisions, "ws"),
                                trace_source=StaticTraceSource(traces()), evaluator=ScoringEvaluator()))
    assert report.outcomes() == ["accepted"]
    # show
    assert main(["--store", store, "--workspace", "ws", "show"]) == 0
    out = capsys.readouterr().out
    assert "graph:" in out and "last validation 0.8" in out and "iteration 1: ACCEPTED" in out and "- read (ACTION)" in out
    # export
    assert main(["--store", store, "--workspace", "ws", "export", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "graph.mmd").exists() and (tmp_path / "out" / "rejections.md").read_text().startswith("# Rejection memory")
    # history and diff
    assert main(["--store", store, "--workspace", "ws", "history", "--kind", "graph"]) == 0
    out = capsys.readouterr().out
    assert out.count("graph ") == 2 and '"origin": "accepted"' in out
    assert main(["--store", store, "--workspace", "ws", "diff"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("--- graph #0") and "+- `read` (ACTION)" in out
    # rejections, both views
    assert main(["--store", store, "--workspace", "ws", "rejections"]) == 0
    assert "## Iteration 1: accepted" in capsys.readouterr().out
    assert main(["--store", store, "--workspace", "ws", "rejections", "--refiner-view"]) == 0
    assert "1 earlier round(s): 1 accepted" in capsys.readouterr().out
    # guide: what the agent would be shown, no model
    assert main(["--store", store, "--workspace", "ws", "guide", "--query", "who?", "--step", "search"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("localized at: search") and "Active Cognitive Node: [search] (Type: ACTION)" in out and "(LEADS_TO; Condition:" in out
    assert main(["--store", store, "--workspace", "ws", "guide", "--query", "who?", "--step", "nope", "--paper-layout"]) == 0
    out = capsys.readouterr().out
    assert "[full graph]" in out and "LEADS_TO;" not in out
    # transfer
    assert main(["--store", store, "--workspace", "ws", "transfer", "--to", "ws2"]) == 0
    assert "ws2: graph" in capsys.readouterr().out
    assert main(["--store", store, "--workspace", "ws", "transfer", "--to", "ws2"]) == 2
    # bad store specs
    assert main(["--store", "s3:bucket", "--workspace", "ws", "show"]) == 2
    assert "unsupported store" in capsys.readouterr().err
    assert main(["--store", "postgres:not-a-url", "--workspace", "ws", "show"]) == 2
    err = capsys.readouterr().err
    assert "postgres" in err


def test_readme_example_parses_and_its_imports_resolve():
    """§8.14: the sixty-second example is an integration sketch (the ``my_*`` names are the host's); everything else in
    it must parse and import."""
    import ast
    import re
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    block = re.search(r"## Sixty-second example.*?```python\n(.*?)```", text, re.S).group(1)
    tree = ast.parse(block)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("proceduralgraph"):
            if node.module.endswith("adapters.anthropic"):
                continue  # needs the optional extra
            module = __import__(node.module, fromlist=[a.name for a in node.names])
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module} has no {alias.name}"
    undefined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id.startswith("my_")}
    assert undefined == {"my_solver_step", "my_scorer", "my_render", "my_training_batch", "my_validation_questions"}
