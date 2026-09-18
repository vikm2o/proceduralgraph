# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""E: the refiner prompt (paper order, one insertion), parsing, and the E2 retry fold."""

import json

from proceduralgraph.config import EvolveConfig
from proceduralgraph.graph import Graph
from proceduralgraph.model import ScriptedChatModel
from proceduralgraph.roles.refiner import REFINER_SYSTEM, TOOLS_UNSPECIFIED, Refiner, render_user_message

from .test_graph import two_hop

GOOD = json.dumps({
    "add_nodes": [{"id": "verify", "type": "reasoning", "description": "Check the answer"}],
    "delete_nodes": [],
    "add_edges": [
        {"source": "read", "target": "verify", "relation": "LEADS_TO", "condition": None, "guidance": "verify first", "pitfalls": "none"},
        {"source": "verify", "target": "answer", "relation": "LEADS_TO", "condition": None, "guidance": "then answer", "pitfalls": None},
    ],
    "delete_edges": [],
})
BAD = json.dumps({"add_nodes": [], "delete_nodes": [], "add_edges": [{"source": "read", "target": "ghost", "relation": "LEADS_TO", "guidance": "g"}], "delete_edges": []})
EMPTY = json.dumps({"add_nodes": [], "delete_nodes": [], "add_edges": [], "delete_edges": []})


def test_prompt_keeps_paper_order_with_one_insertion():
    g = two_hop()
    text = render_user_message(g, task_description="QA", mode="static_incremental", attempts_block="TRACES", rejected_block="NONE", available_tools=["search", "read", "answer"])
    order = [
        "Task context: QA",
        "Refinement mode: static_incremental",
        "Available Tool Actions (the agent can only execute these actions): search, read, answer",
        'Allowed node types (the "type" of every node must be one of these): ACTION, REASONING, STATUS',
        "Recent execution trajectories: \nTRACES",
        "Current Procedural Graph representation: ",
        "Previously rejected candidates: \nNONE",
        "Your job is to refine the Procedural Graph.",
        "1. Action Nodes.",
        "7. Graph Structure.",
        "Make sure to output ONLY the raw JSON block.",
    ]
    positions = [text.index(piece) for piece in order]
    assert positions == sorted(positions)
    assert "Allowed relations" not in text  # paper vocabulary in use
    assert '"id": "search"' in text and text.count("Allowed node types") == 1
    custom = Graph.skeleton(relations=("NEXT", "BACK"))
    assert "Allowed relations: NEXT, BACK" in render_user_message(custom, task_description="t", mode="scratch_incremental", attempts_block="", rejected_block="", available_tools=None)
    assert TOOLS_UNSPECIFIED in render_user_message(g, task_description="t", mode="static_incremental", attempts_block="", rejected_block="", available_tools=None)


async def test_propose_parses_and_prepares_candidate():
    model = ScriptedChatModel(["Here you go:\n```json\n" + GOOD + "\n```"])
    refiner = Refiner(model, EvolveConfig(task_description="QA"))
    result = await refiner.propose(two_hop(), mode="static_incremental", attempts_block="T", rejected_block="R")
    assert result.outcome == "candidate" and result.calls == 1
    assert "verify" in result.candidate.nodes and result.candidate.nodes["verify"].type == "REASONING"
    request = model.calls[0]
    assert request.role == "refiner" and request.system == REFINER_SYSTEM and request.max_tokens == 8192
    assert request.tools == ()


async def test_retry_folds_diagnostics_into_one_user_message():
    model = ScriptedChatModel([BAD, GOOD])
    refiner = Refiner(model, EvolveConfig(role_retries=1))
    result = await refiner.propose(two_hop(), mode="static_incremental", attempts_block="T", rejected_block="R")
    assert result.outcome == "candidate" and result.calls == 2 and len(result.replies) == 2
    retry = model.calls[1]
    assert retry.system == REFINER_SYSTEM and len([p for p in retry.parts]) == 1
    assert "--- Your previous reply ---" in retry.text and "missing_endpoint" in retry.text and "Task context:" in retry.text


async def test_no_retry_when_paper_exact_and_no_action_is_not_retried():
    model = ScriptedChatModel([BAD, GOOD])
    result = await Refiner(model, EvolveConfig(role_retries=0)).propose(two_hop(), mode="static_incremental", attempts_block="T", rejected_block="R")
    assert result.outcome == "structural_failure" and result.calls == 1 and result.edits is not None
    assert {d.code for d in result.diagnostics if d.is_error} >= {"missing_endpoint"}
    model = ScriptedChatModel([EMPTY, GOOD])
    result = await Refiner(model, EvolveConfig(role_retries=1)).propose(two_hop(), mode="static_incremental", attempts_block="T", rejected_block="R")
    assert result.outcome == "no_action" and result.calls == 1
    model = ScriptedChatModel(["I refuse.", "still no"])
    result = await Refiner(model, EvolveConfig(role_retries=1)).propose(two_hop(), mode="static_incremental", attempts_block="T", rejected_block="R")
    assert result.outcome == "structural_failure" and result.edits is None and result.calls == 2
    assert result.diagnostics[0].code == "malformed_edit"
