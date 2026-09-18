# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Bootstrapping a graph from a problem statement and worked solutions (paper-differences §2.19)."""

import asyncio
import json

import pytest

from proceduralgraph import (
    Budget,
    EvolveConfig,
    GraphError,
    HookedModel,
    Hooks,
    MemoryRevisionStore,
    RevisionGraphStore,
    RevisionRejectionStore,
    ScriptedChatModel,
    StaticTraceSource,
    bootstrap_graph,
    evolve,
)
from proceduralgraph.cli import main
from proceduralgraph.roles.bootstrap import NO_SOLUTIONS_BLOCK
from proceduralgraph.stores.file import FileRevisionStore
from proceduralgraph.stores.revision import GRAPH

from .test_harness_end_to_end import BROKEN, BUILD, EMPTY, ScoringEvaluator, traces

PROBLEM = "Answer two-hop questions over a document corpus using search, read and answer."
TOOLS = ["search", "read", "answer"]


async def test_bootstrap_without_solutions_uses_the_one_added_sentence_and_needs_tools():
    model = ScriptedChatModel([BUILD])
    result = await bootstrap_graph(problem_statement=PROBLEM, model=model, tools=TOOLS)
    assert result.ok and result.mode == "scratch_onetime" and result.solutions == 0 and result.calls == 1
    assert {"search", "read", "answer"} <= set(result.graph.nodes)
    prompt = model.calls[0].text
    assert f"Task context: {PROBLEM}" in prompt and "Refinement mode: scratch_onetime" in prompt
    assert NO_SOLUTIONS_BLOCK in prompt and "Available Tool Actions (the agent can only execute these actions): search, read, answer" in prompt
    assert "Previously rejected candidates: \n(none: one-time mode)" in prompt
    assert result.seed_meta() == {"origin": "bootstrapped", "solutions": 0, "mode": "scratch_onetime", "refiner_calls": 1}
    with pytest.raises(ValueError, match="needs the tool list"):
        await bootstrap_graph(problem_statement=PROBLEM, model=ScriptedChatModel([BUILD]))
    with pytest.raises(ValueError, match="must not be empty"):
        await bootstrap_graph(problem_statement="   ", model=ScriptedChatModel([BUILD]), tools=TOOLS)


async def test_bootstrap_with_solutions_presents_them_as_passing_trajectories():
    model = ScriptedChatModel([BUILD])
    result = await bootstrap_graph(problem_statement=PROBLEM, model=model, solutions=["search Alice; read; search Lyon; answer France", "", "search Chen; answer"])
    assert result.ok and result.solutions == 2
    prompt = model.calls[0].text
    assert "=== Trace solution-1 | task solution-1 | score 1.000 | PASSED ===" in prompt and "search Lyon; answer France" in prompt
    assert "=== Trace solution-2 " in prompt and NO_SOLUTIONS_BLOCK not in prompt
    assert "not specified: preserve existing ACTION node ids" in prompt  # no tools given: the refiner infers them from the solutions


async def test_bootstrap_refuses_broken_or_empty_output():
    broken = await bootstrap_graph(problem_statement=PROBLEM, model=ScriptedChatModel([BROKEN, BROKEN]), tools=TOOLS, config=EvolveConfig(role_retries=1))
    assert not broken.ok and broken.calls == 2 and "missing_endpoint" in {d.code for d in broken.diagnostics}
    assert broken.refusal().startswith("bootstrap refused: error:missing_endpoint")
    empty = await bootstrap_graph(problem_statement=PROBLEM, model=ScriptedChatModel([EMPTY]), tools=TOOLS)
    assert not empty.ok and empty.diagnostics[-1].code == "no_action" and "proposed no edits" in empty.diagnostics[-1].message
    cancelled = json.dumps({"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": [{"source": "nowhere", "target": "End"}]})
    net_zero = await bootstrap_graph(problem_statement=PROBLEM, model=ScriptedChatModel([cancelled]), tools=TOOLS)
    assert not net_zero.ok and "cancelled out" in net_zero.diagnostics[-1].message


async def test_bootstrap_result_seeds_evolve_with_origin_bootstrapped():
    good = await bootstrap_graph(problem_statement=PROBLEM, model=ScriptedChatModel([BUILD]), tools=TOOLS)
    revisions = MemoryRevisionStore()
    evaluator = ScoringEvaluator()
    report = await evolve(config=EvolveConfig(max_rounds=1, role_retries=0), model=ScriptedChatModel([EMPTY]), graph_store=RevisionGraphStore(revisions, "ws"),
                          rejection_store=RevisionRejectionStore(revisions, "ws"), trace_source=StaticTraceSource(traces()), evaluator=evaluator,
                          initial_graph=good)
    assert evaluator.calls[0][0] == "baseline" and report.best.score == 0.8 and report.outcomes() == ["no_action"]
    head = await revisions.head("ws", GRAPH)
    assert head.meta["origin"] == "bootstrapped" and head.meta["solutions"] == 0 and head.meta["refiner_calls"] == 1
    refused = await bootstrap_graph(problem_statement=PROBLEM, model=ScriptedChatModel([EMPTY]), tools=TOOLS)
    with pytest.raises(GraphError, match="bootstrap refused"):
        await evolve(config=EvolveConfig(max_rounds=1), model=ScriptedChatModel([EMPTY]), graph_store=RevisionGraphStore(MemoryRevisionStore(), "ws"),
                     rejection_store=RevisionRejectionStore(MemoryRevisionStore(), "ws"), trace_source=StaticTraceSource(traces()), evaluator=evaluator,
                     initial_graph=refused)


async def test_bootstrap_reuses_a_shared_hooked_model_and_its_budget():
    hooked = HookedModel(ScriptedChatModel([BUILD]), Hooks())
    result = await bootstrap_graph(problem_statement=PROBLEM, model=hooked, tools=TOOLS, config=EvolveConfig(budget=Budget(max_model_calls=3)))
    assert result.ok and result.calls == 1 and hooked.meter.model_calls == 1 and hooked.meter.budget.max_model_calls == 3
    exhausted = HookedModel(ScriptedChatModel([BUILD]), Hooks())
    exhausted.meter.model_calls = 3
    with pytest.raises(Exception, match="budget exhausted"):
        await bootstrap_graph(problem_statement=PROBLEM, model=exhausted, tools=TOOLS, config=EvolveConfig(budget=Budget(max_model_calls=3)))


def test_cli_init_from_text_with_a_scripted_model(tmp_path, capsys):
    (tmp_path / "problem.md").write_text(PROBLEM)
    solved = tmp_path / "solved"
    solved.mkdir()
    (solved / "a.md").write_text("search Alice; read; search Lyon; answer France")
    (solved / ".gitkeep").write_text("")
    (tmp_path / "replies.json").write_text(json.dumps([BUILD]))
    store = f"file:{tmp_path / 'store'}"
    assert main(["--store", store, "--workspace", "qa", "init", "--from-text", str(tmp_path / "problem.md"), "--solutions", str(solved),
                 "--tools", "search,read", "--tools", "answer", "--model", f"scripted:{tmp_path / 'replies.json'}", "--retries", "0"]) == 0
    out = capsys.readouterr().out
    assert "seeded graph" in out and "(bootstrapped)" in out and "1 worked solution(s)" in out and "not a validated graph" in out
    head = asyncio.run(FileRevisionStore(tmp_path / "store").head("qa", GRAPH))
    assert head.meta["origin"] == "bootstrapped" and head.meta["solutions"] == 1 and head.meta["mode"] == "scratch_onetime"
    replies = json.loads((tmp_path / "replies.json").read_text())
    assert replies == [BUILD]


def test_cli_init_from_text_failures_are_plain_messages(tmp_path, capsys):
    store = f"file:{tmp_path / 'store'}"
    problem = tmp_path / "problem.md"
    problem.write_text(PROBLEM)
    good_model = tmp_path / "replies.json"
    good_model.write_text(json.dumps([BUILD]))

    def run(*extra):
        code = main(["--store", store, "--workspace", "ws", "init", *extra])
        return code, capsys.readouterr().err

    code, err = run("--from-text", str(problem))
    assert code == 2 and "needs --model" in err
    code, err = run("--from-text", str(problem), "--model", f"scripted:{good_model}")
    assert code == 2 and "needs --tools" in err
    empty = tmp_path / "empty.md"
    empty.write_text("   \n")
    code, err = run("--from-text", str(empty), "--tools", "a", "--model", f"scripted:{good_model}")
    assert code == 2 and "is empty" in err
    code, err = run("--from-text", str(tmp_path / "missing.md"), "--tools", "a", "--model", f"scripted:{good_model}")
    assert code == 2 and "cannot read the problem statement" in err
    bad_dir = tmp_path / "bin"
    bad_dir.mkdir()
    (bad_dir / "blob.bin").write_bytes(b"\xff\xfe\x00\x00binary")
    code, err = run("--from-text", str(problem), "--solutions", str(bad_dir), "--model", f"scripted:{good_model}")
    assert code == 2 and "cannot read the solution" in err and "UTF-8" in err
    nothing = tmp_path / "nothing"
    nothing.mkdir()
    code, err = run("--from-text", str(problem), "--solutions", str(nothing), "--model", f"scripted:{good_model}")
    assert code == 2 and "holds no text files" in err
    code, err = run("--from-text", str(problem), "--solutions", str(problem), "--model", f"scripted:{good_model}")
    assert code == 2 and "must be a directory" in err
    bad_model = tmp_path / "bad.json"
    bad_model.write_text(json.dumps([BROKEN, BROKEN]))
    code, err = run("--from-text", str(problem), "--tools", "search,read,answer", "--model", f"scripted:{bad_model}")
    assert code == 2 and "bootstrap refused" in err and "missing_endpoint" in err
    code, err = run("--from-text", str(problem), "--tools", "a", "--model", "carrier:pigeon")
    assert code == 2 and "unsupported model" in err
    code, err = run("--from-text", str(problem), "--tools", "a", "--model", f"scripted:{tmp_path / 'nope.json'}")
    assert code == 2 and "cannot read scripted replies" in err
    code, err = run("--graph", "x.json", "--from-text", str(problem))
    assert code == 2 and "not both" in err
    code, err = run("--tools", "a")
    assert code == 2 and "only apply with --from-text" in err
    # a provider that fails to construct is a plain message too (no API key in the environment)
    import os

    os.environ.pop("OPENAI_API_KEY", None)
    pytest.importorskip("openai")
    code, err = run("--from-text", str(problem), "--tools", "a", "--model", "openai:some-model")
    assert code == 2 and "cannot create the openai client" in err
