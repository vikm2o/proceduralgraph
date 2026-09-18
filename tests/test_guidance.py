# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""C: the online runtime. Acceptance §8.8 (eq. 2 fallback and hop-bounded context)."""

import pytest

from proceduralgraph.guidance import (
    FULL_CONTEXT_DESC,
    SUBGRAPH_CONTEXT_DESC,
    ExactActionLocalizer,
    GuidanceConfig,
    Guide,
    steps_from_actions,
)
from proceduralgraph.model import ScriptedChatModel
from proceduralgraph.traces import Step

from .test_graph import two_hop


async def test_fallback_to_full_graph_when_match_fails_and_hops_bound_the_context():
    g = two_hop()
    model = ScriptedChatModel(lambda request: "GUIDANCE:" + request.text[:40])
    guide = Guide(g, model, GuidanceConfig(hops=2, window=3, task_description="two-hop QA"))
    # Match succeeds on the last action
    result = await guide.guidance("who?", steps_from_actions(["search"]))
    assert result.node_id == "search" and result.used_full_graph is False and result.subgraph_digest
    assert "[search] → [read]" in result.context and "[read] → [answer]" in result.context
    assert "[Start]" not in result.context and "[answer] → [End]" not in result.context  # outside two hops of `search`
    assert model.calls[-1].role == "guidance" and SUBGRAPH_CONTEXT_DESC in model.calls[-1].text
    assert "two-hop QA" in model.calls[-1].text and result.text.startswith("GUIDANCE:")
    # Match fails: the guidance model receives the complete graph (eq. 2, "otherwise")
    result = await guide.guidance("who?", [Step("unknown_tool", {"x": 1}, "boom")])
    assert result.used_full_graph is True and result.node_id is None and result.subgraph_digest is None
    assert result.context.startswith("Complete Procedural Graph")
    assert FULL_CONTEXT_DESC in model.calls[-1].text
    # Empty trajectory is Start (a_0 = Start)
    result = await guide.guidance("who?", [])
    assert result.node_id == "Start" and not result.used_full_graph
    assert guide.visited == ["search", "Start"] and len(guide.guidance_log) == 3 and guide.model_calls == 3


async def test_window_limits_recent_context():
    g = two_hop()
    model = ScriptedChatModel(["ok"])
    guide = Guide(g, model, GuidanceConfig(window=2))
    steps = [Step("search", "a", "o1"), Step("read", "d", "o2"), Step("search", "b", "o3")]
    await guide.guidance("q", steps)
    prompt = model.calls[0].text
    assert "o1" not in prompt and "o2" in prompt and "o3" in prompt and "[... 1 earlier step(s) omitted ...]" in prompt


async def test_raw_modes_and_none_need_no_model():
    g = two_hop()
    raw = Guide(g, None, GuidanceConfig(mode="raw_subgraph"))
    result = await raw.guidance("q", steps_from_actions(["read"]))
    assert result.text == result.context and result.text.startswith("Active Cognitive Node: [read]")
    full = Guide(g, None, GuidanceConfig(mode="raw_full"))
    result = await full.guidance("q", steps_from_actions(["read"]))
    assert result.used_full_graph and result.text.startswith("Complete Procedural Graph")
    none = Guide(g, None, GuidanceConfig(mode="none"))
    result = await none.guidance("q", steps_from_actions(["read"]))
    assert result.text == "" and result.node_id == "read"
    with pytest.raises(ValueError):
        Guide(g, None, GuidanceConfig(mode="generative_subgraph"))


async def test_generative_full_and_custom_localizer_and_max_calls():
    g = two_hop()
    model = ScriptedChatModel(["one", "two"])
    guide = Guide(g, model, GuidanceConfig(mode="generative_full"))
    result = await guide.guidance("q", steps_from_actions(["read"]))
    assert result.used_full_graph and result.node_id == "read" and result.text == "one"
    normalizer = ExactActionLocalizer(normalize=lambda a: a.split("(")[0].lower())
    capped = Guide(g, ScriptedChatModel(["first"]), GuidanceConfig(max_guidance_calls=1), localizer=normalizer)
    first = await capped.guidance("q", steps_from_actions(["SEARCH(q=1)"]))
    assert first.node_id == "search" and first.text == "first"
    second = await capped.guidance("q", steps_from_actions(["READ(d)"]))
    assert second.mode == "raw_subgraph" and second.text.startswith("Active Cognitive Node: [read]") and capped.model_calls == 1


def test_sync_wrapper_and_context_for():
    g = two_hop()
    guide = Guide(g, None, GuidanceConfig(mode="raw_subgraph"))
    result = guide.guidance_sync("q", steps_from_actions(["answer"]))
    assert result.node_id == "answer"
    node, context, full = guide.context_for(steps_from_actions(["nope"]))
    assert node is None and full and context.startswith("Complete")
