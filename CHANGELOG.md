# Changelog

## 0.3.0 (unreleased)

Quality and cost objectives (`feature-request/feature-proceduralgraph-objectives-1.md`). Additive and opt-in: with no
objective configured every 0.2.0 behaviour, document shape and digest is unchanged.

- `objectives.py`: `MetricSpec`, `ObjectiveSpec`, `ObjectiveContext` (validated, deterministic documents, content
  digests), `Interval`, `hoeffding_radius`, `paired_hoeffding_v1`.
- `TaskOutcome.metrics` and `Evaluation.objective_context` (optional; omitted from documents when absent).
- `ObjectiveGate(objective, context)`: lexicographic and pareto acceptance over paired Hoeffding intervals with
  absolute mean floors/ceilings, complete-pairing and context checks, six dispositions and stable reason codes in
  `Decision.feedback`; the identity probe never stops the run.
- `EvolveConfig.objective` / `objective_context`; the loop enforces `Evaluation.ref` against the graph it asked to
  evaluate, binds checkpoints to the objective and context digests, refuses a pending checkpoint bound to anything
  else (and refuses to finish an objective-bound checkpoint in scalar mode), records the decision before the head
  moves, and recovers an acceptance without evaluating the accepted graph as its own control.
- `Decision.feedback`, `baseline_ref` and `candidate_ref` persist on `RejectionEntry`, `IterationReport` and the
  accepted revision meta for every gated round, scalar gates included (their renderings are unchanged); duplicate
  refusal counts measured rejections and structural failures that recorded a digest, never an equivalent / unresolved
  / unmeasured / invalid comparison; on resume after an acceptance the recorded candidate evaluation becomes the
  baseline in scalar mode too, so the accepted graph is never re-evaluated as its own control; the refiner is shown the frozen
  objective and every decision's per-metric comparison; per-trace `measured:` observations in the trajectories.
- `examples/objective_evolution.py`: both modes, unknown cost, restart recovery, deferred host evaluation, offline.
- Reader/writer compatibility (REQ-018): 0.3.0 reads every 0.2.0 document unchanged, keeping its shape and digest
  (fixture-tested against a captured 0.2.0 run). The other direction is partial: 0.3.0 now persists the gate's
  `decision` on every gated round, scalar gates included, so rejection memory written by 0.3.0 carries a key 0.2.0
  did not write; 0.2.0's `RejectionEntry.from_dict` and `Evaluation.from_document` ignore unknown keys, but 0.2.0's
  `Trace.from_document` and `Evaluation.from_document` build `TaskOutcome(**dict)`, so a trace or evaluation document
  whose outcomes carry `metrics` (objective mode only) raises `TypeError` under a 0.2.0 reader. Do not read objective
  mode workspaces with 0.2.0.

### Upgrading to 0.3.0

Nothing is required for existing hosts: with no objective configured, gates, ties, documents, digests, guidance
modes, transports and stores behave as in 0.2.0. To adopt objective mode:

1. Put `metrics={name: value}` (original units; `None` for an explicit unknown) on every `TaskOutcome` your evaluator
   returns, one outcome per expected task id, failed and no-change paid attempts included.
2. Declare an `ObjectiveSpec` and an `ObjectiveContext` once per run; set `objective_context=context` on every
   `Evaluation` and return the `ref` the loop asked you to evaluate (`score` may stay `None`).
3. Pass `EvolveConfig(objective=objective, objective_context=context)` and `gate=ObjectiveGate(objective, context)`.
4. Start in a fresh workspace if the old one has a pending (uncompleted) checkpoint: it is bound to no objective and
   0.3.0 refuses to finish it under one. Completed history needs no migration.
5. Read decisions from `IterationReport.decision`, `RejectionEntry.decision` and the accepted revision's
   `meta["decision"]`; the CLI `rejections` view and `export` render the same lines.

## 0.2.0 (2026-09-18)

- `bootstrap_graph(problem_statement, solutions=(), tools=..., model=...)` and CLI `init --from-text FILE
  [--solutions DIR] [--tools a,b] --model SPEC [--retries N] [--cycle-policy allow|repair]`: draft a first graph from
  a task description and optional worked solutions with `refine_once` in the paper's `scratch_onetime` mode, seeded
  with `origin: bootstrapped` on both the CLI path and `evolve(initial_graph=result)` (paper-differences §2.19).
  `refine_once` gains `attempts_block=` and reuses a `HookedModel` it is given. CLI `open_model` accepts
  `anthropic:ID`, `openai:ID` and `scripted:FILE` (a JSON list of replies, for offline demos and tests).

## 0.1.0 (2026-09-18)

First release: an independent reimplementation of *Procedural Graphs: Self-Evolving Execution Structures for LLM
Agents* (arXiv:2609.09153). Every departure from the paper and every decision where it is silent is catalogued in
`docs/paper-differences.md`.

- Data model: `Graph`, `Node`, `Edge`, `Diagnostic`; canonical JSON documents with content digests; the paper's four
  relations and three edge attributes by default, configurable per task; case-insensitive node types and relations.
- Online runtime: `Guide` with exact-match localization, `h=2` hop neighbourhoods, `w=3` step windows, the App. B.5
  guidance prompt and serializer layout, full-graph fallback, raw / generative / none modes, a per-episode call cap.
- Edits: `EditSet` parsing tolerant of prose and fences; `prepare_candidate` in the paper's order with cycle repair and
  the App. B.6 structural checks; `unified_diff` over the markdown rendering.
- Offline loop: `evolve` (Algorithm 1) with the tie-accepting gate, rejection memory persisted after every round,
  duplicate-candidate refusal, `no_action` handling, stale-trace policy, host-canonical candidates, budgets, resume,
  hooks and metering; `refine_once` for the one-time modes.
- Refiner: the App. B.5 prompt in the paper's order with one inserted vocabulary line; one retry with diagnostics fed
  back (`role_retries=0` is paper-exact).
- Gates: `TieAcceptingGate` (default), `StrictImprovementGate`, `PairedGate`.
- Stores: `RevisionStore` protocol byte-for-byte compatible with `skillwiki` 0.2.0; memory, file, PostgreSQL (table
  configurable), S3 and GCS adapters; `RevisionGraphStore` / `RevisionRejectionStore` / `RevisionTraceStore` /
  `RevisionCheckpointStore`; `seed_workspace` transfer; `export_workspace` (`graph.json`, `graph.md`, `graph.mmd`,
  `rejections.md`).
- Initialisation from a hand-written graph or the `Start → End` skeleton, in code and from the CLI.
- CLI: `init`, `show`, `export`, `history`, `diff`, `rejections`, `transfer`, `guide`.
- Adapters: Anthropic and OpenAI, text protocol only.
- Examples: `filesystem_toy.py` (no network) and `anthropic_toy.py`.
- Review hardening before release: every digest comparison in the loop (duplicate refusal, resume-after-accept) is made
  on the host's materialised candidate and that candidate is run through the structural checks; a shared `HookedModel`
  meters guidance calls under the run budget; budget exhaustion inside the baseline evaluation stops cleanly; pending
  checkpoints are reconciled before the perfect-score stop; `Graph`, `Node` and `Edge` are read-only all the way down;
  an explicitly empty vocabulary round-trips; a refiner reply without the four arrays is malformed rather than
  `no_action`; stale traces are marked in the refiner context; a zero trajectory window shows nothing. Storage: the file
  store locks each chain against concurrent writers and percent-encodes workspace names; every adapter's `list` / `get`
  follows the committed chain so a losing writer's orphan is never read. Docs: `docs/architecture.md` with the flow,
  storage, metering and recovery diagrams.
