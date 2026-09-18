---
goal: Release reusable quality and cost objectives in ProceduralGraph before integrating them into generation-agent
version: 1.0
date_created: 2026-09-18
last_updated: 2026-09-18
owner: ProceduralGraph maintainers; generation-agent maintainers own subsequent adoption
status: 'Released upstream (0.3.2); consumer adoption pending'
tags: [feature, proceduralgraph, objectives, evaluation, upstream-handoff]
---

# Introduction

> **Upstream release delivered 2026-09-18.** Version `0.3.2`, tag `0.3.2`, commit `7b2a011`. Wheel SHA-256
> `8d54f101403a09694026394d6401aa4e5a954d3f48e723e6f1f92820a43c27b8`, sdist SHA-256
> `c1dc72cbcec6b773864b3784406bde9d751c13f4bf2f73b535f6852399f3481c`. 136 tests, 0 skipped (PostgreSQL and S3
> contract tests ran). Tags `0.3.0` and `0.3.1` are mis-pointed and are not releases (see `CHANGELOG.md`).
> Phase 4 (TASK-007 to TASK-009) is generation-agent's adoption work and remains open.

![Status: Released upstream](https://img.shields.io/badge/status-Released%20upstream%200.3.2-green)

**Handoff to the ProceduralGraph implementation agent.** Implement this requirement in
`vikm2o/proceduralgraph`, test it, and deliver a new immutable tagged release with the
release evidence specified below. Generation-agent will adopt that release in a separate
change. This document authorizes neither a dependency upgrade here nor a change to the
current production selection policy.

The objective is to evolve decision graphs toward better task outcomes and lower actual
resource use. The objective must control evaluation, candidate acceptance, and refiner
feedback. A sentence in the refiner prompt alone does not implement this requirement.

The inspected baseline is ProceduralGraph source version `0.2.0`, pinned here at
`40a08066608c036d121fad9b5bb61f16bcc4efb9`. Its `Evaluation` supports arbitrary aggregates
and custom gates, but its built-in gates compare scalar scores. Rejection rendering
centers on scores, and `harness.evolve` does not persist `Decision.feedback`. These
existing extension points are sufficient for an additive implementation.

All paths in implementation tasks are relative to the **ProceduralGraph repository**
unless explicitly prefixed with `generation-agent/`. Keep the upstream implementation,
examples, and published documentation host-agnostic. This document contains the consumer
requirements so the receiving agent does not need access to generation-agent.

## 1. Requirements & Constraints

### Objective and measurement contract

- **REQ-001 — Public objective API:** Add `MetricSpec`, `ObjectiveSpec`, and
  `ObjectiveContext` in `src/proceduralgraph/objectives.py`, exported from
  `proceduralgraph`. They must validate inputs, serialize deterministically, round-trip,
  and expose a content digest. Objective definitions are immutable for an evolution run.
  Preserve caller-defined names and units; the library must not hard-code SKU terminology,
  vendors, currency prices, or regeneration-to-tool conversion weights.
- **REQ-002 — Metric definition:** Each `MetricSpec` declares a unique name, unit,
  direction (`maximize` or `minimize`), finite observation bounds, minimum meaningful
  improvement, equivalence margin, non-regression margin, optional absolute mean floor
  and ceiling, and role (`optimize` or `report_only`). Margins are finite and nonnegative;
  bounds have strictly positive width. Floors and ceilings use the original units.
  `report_only` metrics cannot decide acceptance or carry acceptance constraints.
- **REQ-003 — Objective definition:** `ObjectiveSpec` declares its schema version,
  ordered metrics, mode (`lexicographic` or `pareto`), uncertainty method, comparison
  error budget `alpha`, and minimum independent paired units. Require at least one
  optimized metric, `0 < alpha < 1`, and at least two independent units. No implicit
  quality/cost weights or hidden tolerances. Metric order is semantically significant
  in lexicographic mode and retained for deterministic reporting in both modes.
- **REQ-004 — Evaluation context:** `ObjectiveContext` binds the objective digest,
  host evaluation-design identity, evaluator version, cohort/scope identity, immutable
  expected task IDs, resource-budget profile identity, evidence-partition identity,
  and measurement basis for each metric. Baseline and candidate must match this context;
  their graph references must identify the respective evaluated graphs. Context and
  objective mismatches are refused with explicit reasons before acceptance.
  `harness.evolve` checks each returned or cached `Evaluation.ref` against the graph
  reference supplied to that evaluation; the evaluator's asserted identity is not enough.
- **REQ-005 — Structured observations:** Add optional `metrics` to `TaskOutcome` and
  optional `objective_context` to `Evaluation`, with backward-compatible constructors
  and document readers. Metric values are finite numbers or explicit unknowns (`None`).
  Monetary values require a consistent declared billing/pricing basis; action counts
  are valid independent metrics. `Evaluation.score` may remain `None`; scalar scores
  and caller-provided aggregate summaries cannot override the new gate's observations.
- **REQ-006 — Complete pairing:** The new gate requires unique task IDs on each side
  and exact coverage of the expected set. It must not silently compare an intersection,
  drop failed tasks, overwrite duplicate IDs, or count missing outcomes as zero cost.
  One task ID represents one independent analysis unit. Hosts aggregate repeated runs
  within that unit using a predeclared rule before supplying observations; poses or
  repeated attempts from one unit must not inflate the independent sample size.
  Means use all assigned units, including unsuccessful executions and paid failures.

### Acceptance rules

- **REQ-007 — Opt-in gate:** Add `ObjectiveGate(objective, context)` in
  `src/proceduralgraph/gates.py`, implementing the existing `Gate` protocol. It returns
  `Decision` with a structured comparison in `feedback`. Additive configuration on
  `EvolveConfig` carries the objective and context to `evolve` and the refiner. When
  configuring `ObjectiveGate`, the gate and run must agree on these digests. Existing
  custom evaluators and gates, including deferred host evaluation, remain supported.

For metric `j`, use oriented differences `d_i = candidate_i - baseline_i` when maximizing,
and `d_i = baseline_i - candidate_i` when minimizing. Positive always means improvement.
Let `[L_j, U_j]` be the paired interval, `g_j` the minimum meaningful improvement,
`e_j` the equivalence margin, and `r_j` the non-regression margin. Preserve original-unit
means and deltas alongside these oriented intervals in every report.

- **REQ-008 — Absolute constraints:** Check all configured optimized-metric absolute
  mean floors and ceilings before declaring a win. Use candidate mean uncertainty:
  the lower bound must meet a floor and the upper bound must meet a ceiling. A proven
  violation rejects; an unresolved constraint cannot pass. These are separate from
  per-observation bounds, which reject malformed/out-of-range data rather than clipping it.
- **REQ-009 — Lexicographic mode:** After absolute constraints, visit optimized metrics
  in declared order. Accept at the first `L_j > g_j`; reject at the first `U_j < 0`.
  Otherwise, proceed to the next metric only if the entire interval lies within
  `[-e_j, e_j]`; return unresolved when it does not. If every metric is equivalent,
  retain the incumbent. A win on an earlier metric grants no improvement claim about
  later metrics. The non-regression margin applies to pareto mode, not this ordering.
- **REQ-010 — Pareto mode:** After absolute constraints, accept only if every optimized
  metric has `L_j >= -r_j` and at least one has `L_j > g_j`. A demonstrated improvement
  plus a demonstrated material regression (`U_j < -r_j`) returns `rejected` with reason
  `trade_off`; a material regression without a demonstrated improvement returns
  `rejected` with reason `metric_regression`. Only when none of these decisions applies,
  return `equivalent` if every interval lies within its equivalence margin, otherwise
  `unresolved`. A tolerated loss must remain visible in feedback and cannot be described
  as an improvement of that metric.
- **REQ-011 — Missing data:** Unknown data for an absolute constraint, a pareto optimized
  metric, or the currently reached lexicographic metric makes that comparison unmeasured.
  Unknown report-only metrics or unreached lexicographic metrics do not invalidate a
  higher-priority win, but cannot support a savings claim. Invalid types, non-finite
  values, inconsistent units/bases, or out-of-bound known values are invalid evidence.
- **REQ-012 — Uncertainty:** Ship a dependency-free `paired_hoeffding_v1` method. For
  observation width `w`, use paired-difference width `2w`; for candidate absolute mean
  intervals use width `w`. With `n` independent paired units, interval radius is
  `width * sqrt(log(2 / alpha_interval) / (2 * n))`. Allocate `alpha_interval = alpha / M`,
  where `M` is the predeclared count of paired intervals for all optimized metrics plus
  one candidate-mean interval for each optimized metric having any absolute constraint.
  Allocate across the full objective even if lexicographic selection returns early.
  Return unresolved below the minimum unit count. Publish method/version, unit count,
  allocation, intervals, and thresholds. Identical observed values on a small sample
  are not proof of equivalence. Zero tolerance can require improvement rather than
  permit a finite-sample equivalence result; document this behavior.
- **REQ-013 — Evidence limits:** State that this uncertainty guarantee is for a fixed
  candidate and predeclared paired comparison on independent evidence. Adaptive reuse
  of the same validation tasks across graph proposals does not establish a new release
  confidence guarantee. Hosts own fresh qualification evidence and any sequential or
  multiple-candidate selection design. Objective mode must not imply that a development
  win automatically authorizes production promotion.
- **REQ-014 — Explicit outcomes:** Feedback includes a schema version, disposition
  (`accepted`, `rejected`, `equivalent`, `unresolved`, `unmeasured`, or `invalid`), stable
  reason codes, objective/context digests, baseline/candidate refs, coverage counts,
  per-metric means/intervals/thresholds, unknown metrics, decisive metric where relevant,
  and constraint results. Include a `trade_off` reason for opposing measured changes.
  Every disposition other than `accepted` maps to `Decision.accepted=False`.
  The identity probe `decide(best, best)` returns `accepted=False, stop=False` for the
  new gate. Perfect quality alone does not end cost optimization.

Illustrative decisions below assume adequate evidence and passing absolute constraints:

| Result | Lexicographic: quality first | Pareto: quality and resources protected |
| --- | --- | --- |
| Better quality, fewer resources | Accept | Accept |
| Equivalent quality, fewer resources | Accept | Accept |
| Better quality, equivalent resources | Accept | Accept |
| Better quality, materially more resources | Accept on quality; show resource increase | Retain incumbent; trade-off |
| Materially worse quality, fewer resources | Reject | Retain incumbent; trade-off |
| All endpoints equivalent | Retain incumbent | Retain incumbent |
| Incomplete pairs or insufficient evidence | No acceptance | No acceptance |

### Feedback, persistence, and compatibility

- **REQ-015 — Refiner context:** Render the frozen objective, metric directions,
  acceptance mode, relevant margins/constraints, and structured measured resource
  observations in `roles/refiner.py` and trace rendering. Supply bounded, readable
  baseline/candidate comparisons explaining which metric caused rejection or why the
  comparison remains unresolved. Preserve objective identity and disposition in summary
  rendering. Existing context caps and redaction apply; observations cannot edit the
  objective, evaluators, budgets, available tools, or acceptance criteria.
- **REQ-016 — Durable decisions:** Persist `Decision.feedback` and the baseline and
  candidate evaluation identities with rejection memory, iteration reports, and accepted
  graph revision metadata. Preserve unknown/unmeasured versus measured rejection even
  when the existing outer iteration outcome remains `rejected` for compatibility.
  CLI rejection views and exports must expose the same structured decision information.
  Custom-gate feedback, including deferred evaluation, must also survive persistence.
- **REQ-017 — Resume binding:** Persist the full objective/context and their digests
  with new-mode checkpoints and decisions. Validate these before a resumed paid stage,
  cached evaluation reuse, or the already-accepted-before-interruption recovery branch.
  A different mode, metric ordering, bound, tolerance, uncertainty setting, cohort,
  evaluator, resource budget, partition, or measurement basis cannot reuse that attempt.
  Record enough baseline and decision information before graph acceptance to recover
  the same decision without reevaluating under the accepted graph as its own control.
  An incompatible or legacy-unbound checkpoint requires a new explicit run/workspace;
  preserve its historical records instead of relabeling or deleting them.
- **REQ-018 — Compatibility:** With no objective configured, preserve current scalar
  gates, tie behavior, custom gates, graph schemas, guidance modes, and transport/store
  protocols. Old documents remain readable with their original meaning and stored
  digests. Omit absent new optional fields when serializing legacy-shaped records so
  they retain their old canonical document shape. Add new fields after existing
  constructor parameters. Document new-data writer/reader compatibility explicitly.

### Consumer boundaries and release handoff

- **CON-001:** Keep core runtime dependencies empty and the package usable offline.
  Objective evaluation is deterministic and adds no guidance/refiner/model call.
  Existing graph structural checks, revision compare-and-swap, metering, and budgets
  continue to apply. This feature evolves the harness; it performs no model training.
- **CON-002:** Measure resources actually consumed per task. Evolution-run spend is
  reported separately from task execution resources. Fewer steps, fewer tokens, or
  equal quota limits alone cannot establish lower total monetary cost.
- **CON-003:** Hosts retain authority over customer requirements, QC, operational
  safeguards, quotas, model/provider admission, deployment, and releases. The graph
  provides advisory guidance. Library acceptance selects a development graph only.
- **CON-004:** Generation-agent's current policy remains quality first, then actual
  regenerations, tools, tool failures, measured policy inference cost, and latency.
  Regenerations have priority because they are more expensive; there is no agreed
  fixed dollar conversion. Complete-SKU success requires all required artifacts to
  pass existing QC under the same per-pose `max_regens`/`max_tools`. Live execution ends
  on complete success or quota exhaustion; library search stopping is a separate loop.
  A future pareto policy is an explicit versioned host-policy change, not an upgrade
  side effect. Units, bounds, quality floors, and margins come from host criteria;
  synthetic example values must not become production defaults.
  The existing host comparison remains authoritative, including its independent quality
  safeguards and uncertainty allocation. Adopting the shared schema and feedback must
  not replace those controls with the library gate or assert statistical equivalence
  between the two evaluators without a separate validation.
- **REQ-019 — Release:** Deliver this in a new compatible minor release after the
  inspected `0.2.0` source, targeting `0.3.0` if that version is unused. Follow the
  repository's git-tag distribution practice; never repoint an existing tag. Update
  package version, changelog, API/host documentation, and paper-differences documentation.
  Build wheel and sdist and smoke-test the built wheel in a clean environment. Report
  actual version, tag, full commit SHA, artifact SHA-256s, tests and skips, and migration
  instructions. A local version field or changelog entry alone is not a released tag.
- **REQ-020 — Integration evidence:** Include a runnable offline example using scripted
  model responses and paired task outcomes that demonstrates both selection modes,
  unknown cost, useful refiner feedback, restart recovery, and deferred host evaluation.
  Include the example command and output in the release handoff. Provide a fixture
  proving that an objective-aware proposal with deferred evaluation cannot be promoted
  by the library. No paid benchmark is required to validate the library feature.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Implement objective contracts and deterministic comparison. Complete when
  contract, coverage, uncertainty, and both gate modes pass their acceptance tests.
  TASK-001 precedes TASK-002; tests develop alongside the task they verify.

| Task | Description | Completed | Date |
| --- | --- | --- | --- |
| TASK-001 | Add `objectives.py` contracts and serialization; extend `traces.TaskOutcome`, `gates.Evaluation`, and `config.EvolveConfig` with optional objective data; export public types in `__init__.py`. Satisfy REQ-001–006 and REQ-018. | true | 2026-09-18 |
| TASK-002 | Implement `gates.ObjectiveGate.decide` and bounded paired interval helpers using REQ-007–014; add `tests/test_objectives.py` and `tests/test_objective_gates.py`. Depends on TASK-001. | true | 2026-09-18 |

### Implementation Phase 2

- GOAL-002: Carry objective evidence through a complete evolution and recovery cycle.
  Complete when scripted end-to-end and interruption tests preserve decision meaning.
  Both tasks depend on Phase 1; TASK-004 also depends on TASK-003.

| Task | Description | Completed | Date |
| --- | --- | --- | --- |
| TASK-003 | Extend `harness.evolve`, `rejections.RejectionEntry`, `hooks.IterationReport`, and checkpoint/revision metadata with objective context and decision feedback, including recovery after graph acceptance. Add `tests/test_objective_resume.py`. Satisfy REQ-016–018. | true | 2026-09-18 |
| TASK-004 | Extend `roles/refiner.py`, `traces.Trace.rendered`, rejection renderings, and CLI export views; add scripted `examples/objective_evolution.py` and refiner/end-to-end tests. Satisfy REQ-015, REQ-016, and REQ-020. | true | 2026-09-18 |

### Implementation Phase 3

- GOAL-003: Ship an inspectable upstream release. Complete only when the tag resolves
  to the tested full commit SHA and release evidence covers every requirement/test.
  Depends on Phase 2; TASK-006 follows TASK-005.

| Task | Description | Completed | Date |
| --- | --- | --- | --- |
| TASK-005 | Run upstream lint, full tests, offline example, package build, and clean-wheel smoke. Update `README.md`, `docs/host-integration.md`, `docs/paper-differences.md`, `CHANGELOG.md`, and package version with compatibility/migration details. | true | 2026-09-18 |
| TASK-006 | Create the new release under the repository release workflow and supply tag, commit, artifact hashes, verification evidence, known limitations, and requirement-to-test mapping. Satisfy REQ-019. | true | 2026-09-18 |

### Implementation Phase 4

- GOAL-004: Adopt the released API in generation-agent in a subsequent change.
  **Deferred until TASK-006 is complete.** This phase belongs to generation-agent's
  agent and is not part of the upstream release completion claim.

| Task | Description | Completed | Date |
| --- | --- | --- | --- |
| TASK-007 | Review the actual upstream tag/commit and migration guide, then update `generation-agent/pyproject.toml`, `uv.lock`, and `agentloop/harness/evolution_libraries.py:ENGINE_REVISIONS` to the same immutable release commit. | false | — |
| TASK-008 | Extend versioned development-trace/run contracts in `generation-agent/agentloop/harness/evolution_libraries.py` and `contracts.py` with measured metrics and objective/context provenance. Map existing criteria in `evaluation.py` to the released API; preserve `DeferredEvaluation`, host qualification, legacy record hashes, and the existing ordered selection policy. Depends on TASK-007. | false | — |
| TASK-009 | Add consumer compatibility and replay tests in `generation-agent/tests/unit/test_evolution_libraries.py`; run affected evaluation, guidance, and lifecycle regressions with runtime files frozen during fingerprint-sensitive workflows. Update `docs/HARNESS_EVOLUTION_LIBRARIES.md`. Quality/cost gains still require a separately authorized measured pilot. Depends on TASK-008. | false | — |

## 3. Alternatives

- **ALT-001:** Prompt-only objective. Insufficient: acceptance can still optimize the
  wrong signal and later refiners cannot learn why a candidate lost.
- **ALT-002:** One weighted quality-minus-cost score. Not selected: hides trade-offs,
  requires arbitrary conversion weights, and can reward material quality loss.
- **ALT-003:** Replace scalar defaults with pareto acceptance. Not selected: breaks
  existing paper-compatible behavior and silently changes consumer selection policy.
- **ALT-004:** Implement the feature only inside generation-agent. Possible through
  existing custom gates, but duplicates reusable comparison and persistence work that
  this requested upstream release should provide.

## 4. Dependencies

- **DEP-001:** Existing `Evaluation`/`Gate` seams, `TaskOutcome`, `EvolveConfig`, revision
  stores, checkpoints, and scripted `ChatModel` transport in the inspected source.
- **DEP-002:** Host-supplied measurements, consistent measurement bases, independent
  analysis units, immutable evaluation designs, and predeclared thresholds. The library
  validates their contract; it cannot independently certify their real-world truth.
- **DEP-003:** An upstream release with the complete handoff is required before
  generation-agent dependency or runtime changes begin.

## 5. Files

- **FILE-001:** New upstream `src/proceduralgraph/objectives.py` and changes to
  `gates.py`, `traces.py`, `config.py`, and `__init__.py` for the public contracts/gate.
- **FILE-002:** Upstream `harness.py`, `rejections.py`, `hooks.py`, `roles/refiner.py`,
  and relevant `stores/revision.py`/CLI rendering paths for evidence and resume behavior.
- **FILE-003:** New upstream objective/gate/resume tests; existing gate, refiner,
  rejection, harness, store, and budget/resume tests; `examples/objective_evolution.py`.
- **FILE-004:** Upstream `pyproject.toml`, version/lock metadata, `README.md`,
  `CHANGELOG.md`, and host-integration/paper-differences docs for the release.
- **FILE-005:** Consumer files listed in Phase 4 are deferred integration work.

## 6. Testing

| ID | Acceptance case |
| --- | --- |
| TEST-001 | Objective/context round-trips are deterministic; changing any semantic field changes its digest. Reject duplicate metrics, invalid directions/modes, invalid bounds/margins/alpha, and inconsistent constraints. |
| TEST-002 | Test every row of the decision table in both modes. Prove that earlier lexicographic wins do not claim later cost savings and that pareto trade-offs cannot accept. |
| TEST-003 | Test absolute floor/ceiling pass, proven violation, and unresolved evidence; quality already at 100% still permits subsequent cost improvement. |
| TEST-004 | Assert interval calculations, metric directions, complete alpha allocation, exact threshold boundaries, insufficient units, and non-equivalence despite equal observed small-sample means. |
| TEST-005 | Refuse missing/extra/duplicate tasks, changed budgets/cohorts/evaluators/partitions, mismatched objective digests and graph refs, unknown required costs, NaN/infinite/out-of-bound values, and inconsistent measurement bases. |
| TEST-006 | Include failed and no-change paid attempts in host fixture totals. Missing monetary cost cannot win monetary comparison; legitimate higher-priority ordering and unknown report-only metrics remain supported. |
| TEST-007 | Verify refiner requests contain the frozen objective, per-metric feedback, and the decisive failure/unknown reason under context caps; custom deferred feedback survives storage and reports without claiming measured rejection. |
| TEST-008 | Interrupt before evaluation, after evaluation, and after graph acceptance. Resume with identical context reuses recorded work/decision; every context/objective mutation refuses before paid work and cannot change old decisions. |
| TEST-009 | Load legacy trace/evaluation/rejection/checkpoint fixtures, preserve their original shapes/digests, retain existing scalar/custom gates and paper defaults, and refuse silent adoption of unbound checkpoints into objective mode. |
| TEST-010 | Exercise memory and file revision stores, export/reload, compare-and-swap conflicts, the offline example, and clean built-wheel imports; run all existing upstream tests and report infrastructure-dependent skips explicitly. |
| TEST-011 | Demonstrate that rejected/equivalent/unresolved/unmeasured candidates leave the retained graph unchanged, objective evaluation causes no model calls, and configured evolution budgets still stop the offline loop. |
| TEST-012 | Verify the release version/tag/full SHA agree, wheel and sdist build from that source, artifact hashes are recorded, and the migration example runs using the released wheel. |

Use synthetic tasks and scripted model responses. Tests establish library behavior,
not a production quality gain or verified monetary savings. Supply a mapping from each
REQ identifier to concrete test names or documentation/release artifacts.

## 7. Risks & Assumptions

- **RISK-001:** Pareto protection can reject a useful quality/cost trade-off. Both modes
  must remain explicit; selecting a new product policy is separate from installing it.
- **RISK-002:** Conservative finite-sample intervals can leave many proposals unresolved.
  Report this honestly; thresholds are not relaxed automatically to obtain acceptance.
- **RISK-003:** Repeated validation reuse can overfit decisions. Development feedback
  and independent qualification remain separate responsibilities of the host.
- **RISK-004:** Resource counts are operational proxies. Decreased counts alone cannot
  prove dollar savings across different providers, models, tools, or pricing bases.
- **ASSUMPTION-001:** The receiving agent works in ProceduralGraph and follows its
  repository instructions for implementation and release. This handoff is the requested
  feature specification; generation-agent's existing runtime remains the baseline.
- **ASSUMPTION-002:** The next release can preserve the zero-dependency core and existing
  protocol shapes through additive optional fields. Any unavoidable breaking change
  must be called out in release versioning and migration evidence before adoption.

## 8. Related Specifications / Further Reading

- [ProceduralGraph repository](https://github.com/vikm2o/proceduralgraph): upstream
  implementation target; inspect `src/proceduralgraph/gates.py`, `harness.py`, and
  `rejections.py` first.
- [Current consumer integration](../docs/HARNESS_EVOLUTION_LIBRARIES.md): implemented
  behavior and the existing development-feedback/release boundary.
- [Consumer objective specification](../docs/SELF_IMPROVING_HARNESS_SPEC.md#obj-003-ordered-comparison):
  authoritative current quality-first ordering, summarized in CON-004.
- [Completed library integration plan](feature-evolution-libraries-1.md).
- [Procedural guidance pilot plan](feature-procedural-policy-guidance-2.md): separate
  measured comparison still required after software work.
