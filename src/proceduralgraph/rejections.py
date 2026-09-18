# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Rejection memory H_rejected (paper §3.3 Step 4; Alg. 1 lines 2, 8, 12, 19). REQUIREMENTS F.

A first-class, persistent, append-only artefact: every round's outcome with its edit set, the candidate graph (by
value; Table 7 graphs are small), the training trace ids and scores, and either the validation outcome or the
structural diagnostics. It is never rolled back or truncated by the loop (F4). ``render_for_refiner`` is
``SerializeRejections`` (R_k); ``render_markdown`` is the operator's ``rejections.md``.

Decisions recorded in docs/paper-differences.md: accepted rounds are stored too (``kind="accepted"``) so the refiner
and the operator see the full search trace (Table 11); the refiner sees the edit set plus score or diagnostics rather
than whole candidate graphs; ``no_action`` and ``duplicate_candidate`` rounds are entries as well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .documents import SCHEMA_VERSION, check_schema, digest, now_iso
from .edits import EditSet
from .gates import Evaluation
from .graph import Diagnostic, Graph
from .hooks import OUTCOMES

KINDS = OUTCOMES  # the five round outcomes (G8 / F1); defined once, in hooks
NON_ACCEPTED = tuple(k for k in KINDS if k != "accepted")


@dataclass
class RejectionEntry:
    iteration: int
    kind: str
    edits: EditSet = field(default_factory=EditSet)
    candidate: Graph | None = None  # None when unavailable (structural failure) or the host chose not to store it (F5)
    candidate_digest: str | None = None
    base_digest: str = ""
    trace_ids: list[str] = field(default_factory=list)
    trace_scores: dict[str, float] = field(default_factory=dict)
    validation: Evaluation | None = None
    retained_score: float | None = None  # S_{k-1}, so a rejection line can say what it lost to
    diagnostics: list[Diagnostic] = field(default_factory=list)
    duplicate_of: int | None = None  # for kind="duplicate_candidate": the earlier iteration
    mode: str | None = None
    created_at: str = field(default_factory=now_iso)
    # REQ-016 (optional, omitted from documents when absent): the gate's structured Decision.feedback and the identities
    # of the two evaluations it compared. ``decision["disposition"]`` distinguishes a measured rejection from
    # equivalent / unresolved / unmeasured / invalid, all of which share the outer ``kind="rejected"``.
    decision: dict[str, Any] | None = None
    baseline_ref: str | None = None
    candidate_ref: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")

    @property
    def score(self) -> float | None:
        return None if self.validation is None else self.validation.score

    @property
    def disposition(self) -> str | None:
        """The gate's disposition when recorded; ``None`` for a scalar gate or a round that never reached the gate."""
        return None if self.decision is None else self.decision.get("disposition")

    @property
    def measured_rejection(self) -> bool:
        """True when this entry is a rejection the gate actually measured (scalar gates, or disposition ``rejected``),
        as opposed to an equivalent / unresolved / unmeasured / invalid comparison that proved nothing (decision 1)."""
        return self.kind == "rejected" and self.disposition in (None, "rejected")

    def diagnostic_codes(self) -> list[str]:
        return sorted({d.code for d in self.diagnostics if d.is_error})

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "iteration": self.iteration,
            "kind": self.kind,
            "edits": self.edits.to_dict(self.candidate.attribute_fields if self.candidate is not None else None),
            "candidate": self.candidate.to_document() if self.candidate is not None else None,
            "candidate_digest": self.candidate_digest,
            "base_digest": self.base_digest,
            "trace_ids": list(self.trace_ids),
            "trace_scores": dict(self.trace_scores),
            "validation": self.validation.to_document() if self.validation is not None else None,
            "retained_score": self.retained_score,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "duplicate_of": self.duplicate_of,
            "mode": self.mode,
            "created_at": self.created_at,
        }
        if self.decision is not None:
            value["decision"] = self.decision
        if self.baseline_ref is not None:
            value["baseline_ref"] = self.baseline_ref
        if self.candidate_ref is not None:
            value["candidate_ref"] = self.candidate_ref
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RejectionEntry:
        return cls(
            iteration=value["iteration"],
            kind=value["kind"],
            edits=EditSet.from_dict(value.get("edits") or EditSet().to_dict()),
            candidate=Graph.from_document(value["candidate"]) if value.get("candidate") else None,
            candidate_digest=value.get("candidate_digest"),
            base_digest=value.get("base_digest", ""),
            trace_ids=list(value.get("trace_ids", [])),
            trace_scores=dict(value.get("trace_scores", {})),
            validation=Evaluation.from_document(value["validation"]) if value.get("validation") else None,
            retained_score=value.get("retained_score"),
            diagnostics=[Diagnostic.from_dict(d) for d in value.get("diagnostics", [])],
            duplicate_of=value.get("duplicate_of"),
            mode=value.get("mode"),
            created_at=value.get("created_at", now_iso()),
            decision=value.get("decision"),
            baseline_ref=value.get("baseline_ref"),
            candidate_ref=value.get("candidate_ref"),
        )

    # -- renderings -------------------------------------------------------------------------------------------------

    def headline(self) -> str:
        """One line: iteration, kind, candidate digest prefix, score or diagnostic codes, edit counts."""
        head = f"iteration {self.iteration}: {self.kind.upper()}"
        if self.candidate_digest:
            head += f" candidate {self.candidate_digest[:12]}"
        if self.disposition is not None and self.kind in ("accepted", "rejected"):
            head += f" | {self.disposition}"
            if self.decision.get("decisive_metric"):
                head += f" on {self.decision['decisive_metric']}"
            reasons = [r for r in self.decision.get("reasons", []) if r != "improvement"]
            if reasons:
                head += f" ({', '.join(reasons)})"
        elif self.kind in ("accepted", "rejected") and self.score is not None:
            retained = "" if self.retained_score is None else f" vs retained {self.retained_score:.4f}"
            head += f" | validation {self.score:.4f}{retained}"
        elif self.kind == "structural_failure":
            head += f" | diagnostics {', '.join(self.diagnostic_codes()) or '(none)'}"
        elif self.kind == "duplicate_candidate":
            head += f" | duplicate of iteration {self.duplicate_of}"
        elif self.kind == "no_action":
            head += " | the refiner proposed no edits"
        return head + f" | edits {self.edits.summary()}"

    def render_full(self) -> str:
        lines = [f"### {self.headline()}", f"Base graph: {self.base_digest[:12] or '(unknown)'}" + (f"; mode {self.mode}" if self.mode else "")]
        lines.append("Edits (JSON):")
        lines.append(json.dumps(self.edits.to_dict(), ensure_ascii=False, indent=1))
        if self.disposition is not None:
            lines.extend(render_decision(self.decision))
        elif self.kind == "rejected" and self.validation is not None:
            lines.append(f"Validation outcome: score {self.score} against retained {self.retained_score}; the candidate was discarded.")
        if self.diagnostics:
            lines.append("Structural diagnostics:")
            lines.extend(f"- {d}" for d in self.diagnostics)
        if self.trace_ids:
            lines.append(f"Training trajectories: {len(self.trace_ids)} ({', '.join(self.trace_ids[:8])}{', …' if len(self.trace_ids) > 8 else ''})")
        return "\n".join(lines)


def _num(value: Any, spec: str = ".4g") -> str:
    """Format a number a host's feedback may or may not have filled in; never raises."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "?"
    return format(value, spec)


def render_decision(decision: dict[str, Any]) -> list[str]:
    """Readable lines for a gate's structured decision (REQ-015/016): disposition, reasons, per-metric means and
    oriented intervals with their verdicts, unknown metrics and constraint results. Bounded: one line per metric.
    Tolerates partial feedback from a host's own gate: a missing number renders as ``?``."""
    reasons = decision.get("reasons") or []
    lines = [f"Decision: {decision.get('disposition')}" + (f" (reasons: {', '.join(str(r) for r in reasons)})" if reasons else "")]
    if decision.get("decisive_metric"):
        lines.append(f"Decisive metric: {decision['decisive_metric']}")
    for name, m in (decision.get("metrics") or {}).items():
        if not isinstance(m, dict):
            continue
        if m.get("known"):
            iv = m.get("oriented_interval") if isinstance(m.get("oriented_interval"), dict) else {}
            unit = m.get("unit", "")
            lines.append(
                f"- {name} [{unit}] {m.get('direction', '')}: baseline {_num(m.get('baseline_mean'))} → candidate {_num(m.get('candidate_mean'))} "
                f"(oriented {_num(iv.get('low'), '+.4g')} .. {_num(iv.get('high'), '+.4g')}; positive = better)"
                + (f"; {m['verdict']}" if m.get("verdict") else "")
                + ("; report only" if m.get("informational") else "")
            )
        else:
            lines.append(f"- {name}: unknown on at least one unit; no claim possible" + ("; report only" if m.get("role") == "report_only" else ""))
    for name, c in (decision.get("constraints") or {}).items():
        if isinstance(c, dict):
            lines.append(f"- constraint on {name}: {c.get('status')}")
    if decision.get("pairing"):
        lines.append(f"Pairing problem: {decision['pairing']}")
    if decision.get("context_mismatch"):
        lines.append(f"Context mismatch: {decision['context_mismatch']}")
    if decision.get("baseline_ref") or decision.get("candidate_ref"):
        lines.append(f"Compared baseline {str(decision.get('baseline_ref'))[:12]} with candidate {str(decision.get('candidate_ref'))[:12]}")
    return lines


@dataclass
class RejectionMemory:
    """Append-only. ``iteration`` is the last round whose outcome is recorded (the harness resumes from ``iteration+1``)."""

    entries: list[RejectionEntry] = field(default_factory=list)
    iteration: int = 0

    def record(self, entry: RejectionEntry) -> None:
        self.entries.append(entry)
        self.iteration = max(self.iteration, entry.iteration)

    def to_document(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "iteration": self.iteration, "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> RejectionMemory:
        check_schema(value, what="rejection memory")
        return cls(entries=[RejectionEntry.from_dict(e) for e in value.get("entries", [])], iteration=int(value.get("iteration", 0)))

    @property
    def digest(self) -> str:
        return digest(self.to_document())

    def rejected_digests(self) -> dict[str, int]:
        """candidate digest → iteration for F3's duplicate check: every measured rejection (scalar gates, or
        disposition ``rejected``) and every structural failure that recorded a candidate digest (the host-validation
        path does; a refiner-side failure usually has no candidate, App. B.6). An objective gate's equivalent /
        unresolved / unmeasured / invalid comparison proved nothing and never blocks a re-proposal (decision 1)."""
        found: dict[str, int] = {}
        for entry in self.entries:
            if (entry.measured_rejection or entry.kind == "structural_failure") and entry.candidate_digest:
                found.setdefault(entry.candidate_digest, entry.iteration)
        return found

    def counts(self) -> dict[str, int]:
        return {kind: sum(1 for e in self.entries if e.kind == kind) for kind in KINDS}

    def last_lines(self, n: int = 10) -> list[str]:
        return [e.headline() for e in self.entries[-n:]]

    # -- SerializeRejections (R_k), Alg. 1 line 8 -------------------------------------------------------------------

    def render_for_refiner(self, *, full_entries: int = 5, char_cap: int = 30_000, objective_summary: str | None = None) -> str:
        """The most recent ``full_entries`` non-accepted entries in full (edit JSON, score or diagnostics, base
        digest, iteration); older non-accepted entries and every accepted entry as one line. Capped at ``char_cap``
        characters, trimming the oldest one-liners first, then demoting the oldest full entries to one line (F2).
        ``objective_summary`` (REQ-015) is a header kept ahead of the entries: the frozen objective the gate applies,
        so the refiner reads every decision below against it."""
        header = f"{objective_summary}\n\n" if objective_summary else ""
        if not self.entries:
            return header + "(none yet)"
        non_accepted = [e for e in self.entries if e.kind in NON_ACCEPTED]
        full_ids = {id(e) for e in non_accepted[-full_entries:]} if full_entries > 0 else set()
        rendered: list[tuple[RejectionEntry, bool]] = [(e, id(e) in full_ids) for e in self.entries]

        def build(items: list[tuple[RejectionEntry, bool]]) -> str:
            parts = [e.render_full() if full else f"- {e.headline()}" for e, full in items]
            counts = self.counts()
            head = (
                f"{len(self.entries)} earlier round(s): {counts['accepted']} accepted, {counts['rejected']} rejected by validation, "
                f"{counts['structural_failure']} failed structural checks, {counts['no_action']} proposed nothing, "
                f"{counts['duplicate_candidate']} repeated an earlier candidate. Do not repeat rejected edits."
            )
            return header + head + "\n\n" + "\n".join(parts)

        text = build(rendered)
        # trim oldest one-liners first
        while len(text) > char_cap and any(not full for _, full in rendered):
            index = next(i for i, (_, full) in enumerate(rendered) if not full)
            rendered.pop(index)
            text = build(rendered)
        # then demote the oldest full entries
        while len(text) > char_cap and any(full for _, full in rendered):
            index = next(i for i, (_, full) in enumerate(rendered) if full)
            rendered[index] = (rendered[index][0], False)
            text = build(rendered)
        if len(text) > char_cap:
            text = text[: max(char_cap - 20, 0)] + "\n[... truncated ...]"
        return text

    # -- operator view --------------------------------------------------------------------------------------------------

    def render_markdown(self) -> str:
        counts = self.counts()
        lines = [
            "# Rejection memory",
            "",
            f"{len(self.entries)} round(s) recorded through iteration {self.iteration}: "
            + ", ".join(f"{v} {k}" for k, v in counts.items()),
            "",
        ]
        for entry in self.entries:
            if entry.kind == "accepted":
                lines.append(f"## Iteration {entry.iteration}: accepted")
                lines.append("")
                lines.append(entry.headline())
                if entry.disposition is not None:
                    lines.extend(render_decision(entry.decision))
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(entry.edits.to_dict(), ensure_ascii=False, indent=1))
                lines.append("```")
            else:
                lines.append(f"## Iteration {entry.iteration}: {entry.kind.replace('_', ' ')}")
                lines.append("")
                lines.append(entry.render_full())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


__all__ = ["KINDS", "NON_ACCEPTED", "RejectionEntry", "RejectionMemory", "render_decision"]
