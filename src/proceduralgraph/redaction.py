# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Redaction of trace text before the refiner, the trace store or rejection memory see it (REQUIREMENTS D6).

A host that learns from production traffic should not hand customer or reviewer identifiers to the model, nor persist
them. ``evolve(..., redact=...)`` applies a redactor to every collected trace's ``text``, every ``Step.observation`` and every
recorded guidance text (generated guidance can echo observation content).
Structured fields (``prediction``, ``truth``, ``meta``, ``Step.args``) are the host's own shapes and are the host's job
to shape safely before it builds the trace.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping

from .traces import Step, Trace

Redactor = Callable[[str], str]


def regex_redactor(patterns: Mapping[str, str | re.Pattern[str]], *, replacement: str = "[REDACTED:{name}]") -> Redactor:
    """Build a redactor from named regular expressions; each match becomes ``replacement`` with ``{name}`` filled in.
    Patterns are applied in the given order, so put the most specific first."""
    compiled = [(name, re.compile(p) if isinstance(p, str) else p) for name, p in patterns.items()]

    def redact(text: str) -> str:
        for name, pattern in compiled:
            text = pattern.sub(replacement.format(name=name), text)
        return text

    return redact


EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
"""A ready pattern for e-mail addresses."""


def redact_trace(trace: Trace, redactor: Redactor) -> Trace:
    """A copy of ``trace`` with redacted ``text`` and step observations; the original is left untouched."""
    text = redactor(trace.text)
    steps = [dataclasses.replace(s, observation=redactor(s.observation)) if s.observation else s for s in trace.steps]
    guidance = [redactor(g) for g in trace.guidance]
    changed = (
        text != trace.text
        or guidance != list(trace.guidance)
        or any(a.observation != b.observation for a, b in zip(steps, trace.steps, strict=True))
    )
    if not changed:
        return trace
    return dataclasses.replace(trace, text=text, steps=[Step(s.action, s.args, s.observation) for s in steps], guidance=guidance)


__all__ = ["EMAIL", "Redactor", "redact_trace", "regex_redactor"]
