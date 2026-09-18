# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""G5: the gates. Acceptance §8.4 (eq. 5 tie acceptance)."""

import pytest

from proceduralgraph.gates import Evaluation, PairedGate, StrictImprovementGate, TieAcceptingGate
from proceduralgraph.traces import TaskOutcome


async def test_tie_accepted_by_default_gate_and_rejected_by_strict():
    best = Evaluation("a", 0.75)
    tie = Evaluation("b", 0.75)
    assert (await TieAcceptingGate().decide(best, tie)).accepted is True
    assert (await StrictImprovementGate().decide(best, tie)).accepted is False
    assert (await TieAcceptingGate().decide(best, Evaluation("c", 0.7499))).accepted is False
    assert (await StrictImprovementGate().decide(best, Evaluation("c", 0.76))).accepted is True


async def test_perfect_probe_lives_on_the_gate():
    perfect = Evaluation("a", 1.0)
    assert (await TieAcceptingGate().decide(perfect, perfect)).stop is False  # paper default: run all K rounds
    assert (await TieAcceptingGate(perfect=1.0).decide(perfect, perfect)).stop is True
    assert (await StrictImprovementGate().decide(perfect, perfect)).stop is True
    assert (await PairedGate().decide(perfect, perfect)).stop is True
    decision = await TieAcceptingGate(perfect=1.0).decide(Evaluation("a", 0.9), Evaluation("b", 1.0))
    assert decision.accepted and decision.stop


def per_task(scores):
    return [TaskOutcome(f"t{i}", s, s >= 0.5) for i, s in enumerate(scores)]


async def test_paired_gate_accepts_ties_needs_per_task_and_enough_tasks():
    best = Evaluation("a", 0.5, per_task=per_task([0.5] * 25))
    same = Evaluation("b", 0.5, per_task=per_task([0.5] * 25))
    decision = await PairedGate().decide(best, same)
    assert decision.accepted is True and decision.feedback["mean_delta"] == 0.0 and decision.feedback["win_probability"] == 1.0
    worse = Evaluation("c", 0.4, per_task=per_task([0.5] * 20 + [0.0] * 5))
    assert (await PairedGate().decide(best, worse)).accepted is False
    better = Evaluation("d", 0.6, per_task=per_task([0.5] * 20 + [1.0] * 5))
    assert (await PairedGate().decide(best, better)).accepted is True
    few = Evaluation("e", 0.9, per_task=per_task([0.9] * 5))
    decision = await PairedGate().decide(Evaluation("a", 0.5, per_task=per_task([0.5] * 5)), few)
    assert decision.accepted is False and decision.feedback["reason"] == "insufficient_paired_tasks"
    with pytest.raises(ValueError):
        await PairedGate().decide(Evaluation("a", 0.5), Evaluation("b", 0.6))
    with pytest.raises(ValueError):
        PairedGate(min_win_probability=0.2)


async def test_evaluation_documents_round_trip():
    e = Evaluation("r", 0.5, aggregate={"k": 1}, per_task=per_task([1.0, 0.0]))
    again = Evaluation.from_document(e.to_document())
    assert again.score == 0.5 and again.per_task[1].task_id == "t1" and e.to_dict()["tasks"] == 2
