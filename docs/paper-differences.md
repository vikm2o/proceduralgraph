# proceduralgraph versus the Procedural Graphs paper

`proceduralgraph` is an independent reimplementation of *Procedural Graphs: Self-Evolving Execution Structures for LLM
Agents* (Lu, Chen, Wu, Arık; arXiv:2609.09153, v1). This document is the complete list of where the package follows the
paper exactly, where the paper is silent and the package had to choose, where it departs, and why. Every departure
states the production problem it solves. Requirement ids (`A6`, `E1`, …) refer to `REQUIREMENTS.md`.

Section references are to the paper (v1). "Alg. 1" is Appendix B.6; the prompts are Appendix B.5.

## 0. Decisions where the paper is silent

Each is a package decision, not fidelity. Change any of them by configuration or by subclassing.

| Question the paper leaves open | Package decision | Where |
| --- | --- | --- |
| `L_max`, and tokens or characters | 120,000 **characters**; a host `token_counter` switches the cap to tokens | `EvolveConfig.refiner_context_cap`, `token_counter` (D4) |
| Concatenation order before the tail cut | High-scoring traces first, low-scoring last (each group by descending score, then id), so the cut from the front keeps the failures the refiner is asked to contrast (§3.3 Step 2). `success_threshold=0.5` splits non-binary scores into the two groups; the `PASSED`/`FAILED` header word is the host's `passed` flag | `render_attempts_block` (D4, G2) |
| How many rejected candidates the refiner sees, in what detail | The 5 most recent non-accepted rounds in full (edit JSON, score or diagnostics, base digest, iteration); older rounds and accepted rounds one line each; 30,000-character cap trimming oldest one-liners first | `RejectionMemory.render_for_refiner` (F2) |
| What the refiner sees of a rejected candidate | App. B.6 says "prior candidate graphs"; the package renders the **edit set** (which, with the base digest, determines the candidate) plus the score or diagnostics, to keep `R_k` small. Candidate graphs are stored by value and available to operators | F2, F5 |
| Empty edit set | Outcome `no_action`: no candidate, no validation, one rejection-memory entry so the refiner sees it next round; counts toward `max_rejected_streak` when set | E3 |
| Malformed JSON | Outcome `structural_failure` with `malformed_edit` diagnostics (after the optional retry, §2.4) | B1, E2 |
| Duplicate of an already-rejected candidate | Outcome `duplicate_candidate`, validation skipped (§2.5) | F3 |
| Retry when the output cannot be applied | One retry with the diagnostics fed back (§2.4); `role_retries=0` is paper-exact | E2 |
| Guidance-call output length | 1,024 tokens | `GuidanceConfig.max_tokens` (C3) |
| Early stop | None by default: all `K` rounds (Alg. 1). `max_rejected_streak` and the gate's `perfect` are opt-in | G2, G5 |
| Reachability **from** `Start` | A warning (`unreachable_from_start`), never a failure; the paper only checks reachability **to** a terminal | A6 |
| Node-type vocabulary | The paper names only `ACTION` ("Status" appears once in the refiner prompt). Default `("ACTION", "REASONING", "STATUS")`; the list is **told to the refiner** (§2.3) because the structural checks enforce it | A1, A6, E1 |
| Case of node types and relation labels | Matched case-insensitively and stored upper-case, so a refiner that writes `Status` or `leads_to` does not fail round one on a case difference | A6 |
| The local-variant slot strings of the guidance prompt | App. B.5 states the `{graph_context_desc}` / `{graph_source}` bindings for the full-graph variant only ("the complete Procedural Graph governing the task structure and strategic guidance" / "complete Procedural Graph"). For the default local variant the package binds "the localized Procedural Graph context around the agent's current node" / "local graph context". The prompt text itself is verbatim | `guidance.SUBGRAPH_CONTEXT_DESC`, `SUBGRAPH_SOURCE` (C3) |
| Whether `End` must exist | Required, and protected from deletion (`reserved_node`); the paper's terminal check is zero out-degree "not specifically the node named End". Kept because the skeleton, refiner rule 6 and transfer all assume the sentinel | A6, B2 |
| Whether accepted rounds are recorded alongside rejections | Yes, `kind="accepted"` with the validation, so the refiner and the operator see the full search trace (Table 11) | F1 |
| Cycle policy default | `repair`: cycle-closing back edges found by a deterministic DFS are removed before validation and reported as `cycle_repaired` warnings (App. B.6 describes exactly this repair). `allow` skips it | B3 |
| `delete_edges` / `delete_nodes` naming something absent | A warning (`no_such_edge`, `no_such_node`), not a failure | B2 |
| Diagnostic codes beyond A6's list | `missing_sentinel` (no `Start`/`End`), `missing_guidance` (warning, refiner rule 3), `unknown_action` (warning, B4), `unknown_attribute` (warning), `cycle_repaired` (warning), `no_such_edge` / `no_such_node` (warnings), `skeleton_completed` / `type_defaulted` / `relation_defaulted` (human-input warnings) | A6, B2, G1a |
| What a gate's equivalent / unresolved / unmeasured / invalid comparison means for later proposals | It proves nothing, so it does **not** feed duplicate refusal: only measured rejections (scalar gates, or disposition `rejected`) block a re-proposal | `RejectionEntry.measured_rejection`, F3 |
| Intervals for report-only metrics | Computed at the same per-interval alpha as the optimized metrics, labelled informational, never decisive and not counted in the alpha allocation `M` | `ObjectiveGate` |
| Where the objective and decisions reach the refiner | Inside the two slots that are already the package's to fill: the frozen objective heads the rejected-candidates slot, per-metric comparisons follow each entry, per-trace `measured:` lines sit in the trajectories slot. The App. B.5 prompt lines stay verbatim | `RejectionMemory.render_for_refiner`, `Trace.rendered` |
| Persisting the gate's feedback in scalar mode | Every gated round records `Decision.feedback` (for the scalar gates: the two scores and the verdict) with `baseline_ref` / `candidate_ref` on the rejection entry, the iteration report and the accepted revision's meta, so custom-gate feedback survives too (REQ-016). Scalar renderings are unchanged: the headline and `rejections.md` still print the validation-versus-retained scores, never a disposition | `RejectionEntry.decision`, §2.16 |
| Baseline after a crash that followed an acceptance | On resume, when the pending checkpoint's candidate is the head and the head moved since the checkpoint, the recorded candidate evaluation is the baseline and no paid evaluation of the head runs, in scalar and objective mode alike; otherwise the head is evaluated as before | `evolve` (§2.9) |
| Refiner mode when the host does not say | `auto`: `scratch_incremental` when the head is the skeleton (exactly `{Start, End}` and one edge), else `static_incremental` | E4 |
| Duplicate `(source, relation, target)` triples | Invalid (`duplicate_edge`); several relations between the same endpoints are valid, as the `delete_edges` semantics imply | A5 |

## 1. Kept exactly

| Mechanism | Paper | proceduralgraph |
| --- | --- | --- |
| Data model | `G = (V, R, E, Φ)`, edges `(u, r, v)` with `condition / guidance / pitfalls`; relations `LEADS_TO, TRIGGERS, PROVIDES_INPUT_FOR, CONVERGES_TO` (§3.1, App. B.4) | `Graph`, `Node`, `Edge`; same defaults, per-task configurable |
| Localization | `u_t = Match(a_{t-1}, V)` by exact match; `a_0 = Start` (eq. 2) | `ExactActionLocalizer`; empty trajectory ⇒ `Start` |
| Context | `N_h(u_t)`: the node plus outgoing transitions up to `h` hops; the full graph when matching fails; window `w` (eq. 2) | `Graph.neighborhood(u, h)`, `Guide` falls back to the full graph and reports `used_full_graph=True` |
| `h`, `w` | `h=2`, `w=3` (§4) | `GuidanceConfig(hops=2, window=3)` |
| Guidance prompt | App. B.5, local and full-graph variants | `GUIDANCE_PROMPT`, verbatim; the full-graph slots bound as App. B.5 states, the local-variant slots a recorded decision (§0) |
| Serialized graph context | App. B.5 layout: active node header, `Immediate Transition Options (Hop 1)`, `Subsequent Horizon (Hop 2)`, `Guidance`, `Pitfalls to Avoid` | `serialize_context(include_relations=False)` reproduces the excerpt byte for byte (tested) |
| Graph frozen during an episode | §3 "Online Inference" | `Guide` holds one immutable `Graph` for its lifetime |
| Loop order | Alg. 1 lines 1–22 | `evolve`, one round per loop body, stages `baseline`, `rollout:k`, `refiner:k`, `validation:k` |
| Refiner prompt | App. B.5, one block | `REFINER_SYSTEM` (the first two sentences) + user message in the paper's order (§2.3 for the one inserted line) |
| Edit interface | `add_nodes / delete_nodes / add_edges / delete_edges`; `delete_edges` removes every relation between the endpoints; `add_edges` applied after deletion; attribute revision = delete + add (App. B.5) | `EditSet`, `prepare_candidate`, no `update_edges` |
| Structural checks | Deletes before adds; malformed edits, invalid node/relation types, missing endpoints fail; cycle repair when cycles are disallowed; every node reaches a zero-out-degree node; tool-catalog membership is **not** enforced (App. B.6) | `prepare_candidate` + `Graph.validate()`; `available_tools` only warns (`unknown_action`) |
| Structural failures skip validation | Alg. 1 lines 11–13; retained graph and cached score unchanged | Recorded as `structural_failure` with diagnostics; `Evaluator` never called (tested) |
| Gate | Accept iff `S_val(cand) >= S_val(retained)`, ties accepted (eq. 5, Alg. 1 l.16) | `TieAcceptingGate`, the default |
| Baseline once | "The initial graph is evaluated once to establish the reference score" (§3.3) | Line 1 runs once per `evolve`; hosts pass `baseline=` to skip it |
| Rejection memory | Rejected candidates with edits, training trajectories and validation outcome or diagnostics; supplied to the refiner as negative evidence (§3.3 Step 4, Alg. 1 l.8, 12, 19) | `RejectionMemory`, persistent and append-only; `render_for_refiner` = `SerializeRejections` |
| Tail | `Tail_Lmax` keeps the **end** of the concatenated trajectories (§3.3, App. B.6) | `tail()` (tested) |
| Strides | `S=100` / `S=20` sequential strides of the training split (App. B.1) | `StridedTraceSource(traces, stride)` |
| Rounds, token limits, decoding | ten rounds (§5.4, App. E); refiner 8,192 tokens (App. D.3); temperature 0 | `max_rounds=10`, `refiner_max_tokens=8192`; decoding is the host adapter's concern |
| Construction modes | `static_onetime`, `static_incremental`, `scratch_onetime`, `scratch_incremental` (App. D.2) | Incremental modes through `evolve`; one-time modes through `refine_once` (no gate, no memory, as the paper's ablation) |
| Skeleton | `G_skeleton = (Start → End)` (App. D.2) | `Graph.skeleton()` |

## 2. Departures

Each entry: what the paper does, what proceduralgraph does, why.

### 2.1 Storage is immutable JSON in a revision chain, not files
Paper: graphs and checkpoints saved as files. Package: the graph, rejection memory, traces and checkpoints are JSON
documents chained by SHA-256 digest behind a `RevisionStore` protocol (memory, file, PostgreSQL, S3, GCS), the same
shapes as `skillwiki` so one host adapter serves both. `GraphStore` is the abstraction; the JSON document in Postgres
is its first implementation, and a graph database can be added behind the same protocol later (§4).
Why: production runs on clusters with no shared disk; two workers must not fork a chain (`expected_head` / `HeadMoved`,
backed by the `(workspace, kind, seq)` primary key and object-store write preconditions). `list` and `get` on every
adapter follow the committed chain from HEAD through `parent_digest`, so an object a losing writer left behind is never
read back. The file store serialises writers on one machine with an advisory lock per chain and percent-encodes
workspace and kind names, so distinct names are distinct directories and none can escape the root.

### 2.2 Traces may come from production, not only fresh rollouts
Paper: every round re-runs a training stride under the current graph. Package: `TraceSource` is a protocol; a host may
return production episodes that ran under `graph_ref` and later received a score. `Trace.graph_ref` and
`stale_trace_policy` (`warn` default, `drop`, `allow`) tell the refiner which graph produced the behaviour it is
diagnosing: under `warn` each trace block carries its graph revision and a `STALE` marker when it differs from the
head, and the attempts-block summary counts them. Why: a production system already pays for those runs; they are rollouts of a known graph at zero extra
cost, and production traces arrive late.

### 2.3 One inserted prompt line: the allowed vocabularies
Paper: the refiner prompt never states which node types or relations are valid, yet the structural validator fails a
candidate on an unknown type or relation. Package: directly after the "Available Tool Actions" line the refiner is
told `Allowed node types (…): ACTION, REASONING, STATUS` and, only when a host changed the paper's four relations,
`Allowed relations: …`. Everything else in the prompt is verbatim and in the paper's order. Why: without it, a refiner
that writes `DECISION` fails round one for a rule it was never given.

### 2.4 One retry with the diagnostics fed back
Paper: a structurally invalid candidate is recorded and the round ends. Package: `role_retries=1` by default; the
diagnostics are appended verbatim to the original user message together with the previous reply, and the refiner
answers again. Two model calls, one stage, one checkpoint. `role_retries=0` is paper-exact. Why: a missing endpoint or
a typo in a relation name is usually a one-token fix, and a validation rollout is far more expensive than a refiner
call.

### 2.5 Known candidates are not re-validated
Paper: every structurally valid candidate is evaluated. Package: a candidate whose digest equals the head's or any
earlier rejected candidate's is recorded as `duplicate_candidate` and skips validation. The comparison runs **after**
`GraphStore.propose`, on the host's materialised graph, because §2.10 makes the host's candidate canonical and that is
also the digest rejection memory records; comparing the module's pre-materialisation digest would never match for a
host that normalises. A refiner-side structural failure usually carries no candidate graph (App. B.6: "G_k^cand may be
unavailable") and cannot match; a structural failure of the host's materialised candidate records its digest and does.
An objective gate's equivalent, unresolved, unmeasured or invalid comparison proved nothing and never blocks (§0). Why: validation is the expensive step (Alg. 1 l.15); paying it twice for
the same graph buys nothing, and the paper's own motivation for rejection memory is that refiners repeat themselves.

### 2.6 Accepted rounds join rejection memory
Paper: `H_rejected` holds gate rejections and structural failures. Package: accepted rounds are recorded too, with
their validation, as `kind="accepted"`; the refiner sees them as one line each. Why: the operator gets the full search
trace (Table 11) from one artefact, and the refiner learns what worked as well as what did not.

### 2.7 Pluggable gate and a statistical option
Paper: scalar `>=`. Package: `Gate` protocol; `StrictImprovementGate` (skillwiki's `>`), `PairedGate` (paired bootstrap
over per-task scores, accepting iff mean delta `>= 0` and `P(delta >= 0) >= 0.9`, so it keeps the paper's tie rule),
or a host's own. The perfect-score early stop lives on the gate (`perfect`), off for the default gate; when it is on,
the harness probes it once on the baseline (a workspace whose head already scores `perfect` returns with
`stopped_reason="perfect"` and zero rounds) and again after each acceptance. Why: §5.4 of the paper itself notes that
with 20 validation episodes single decisions turn on one or two episodes; a host with a larger validation set can
demand evidence.

### 2.8 Budgets
Package: `Budget(max_model_calls, max_output_tokens, max_evaluations, max_seconds)` checked before every model call
and every evaluation; exhaustion stops the run with `stopped_reason="budget_exhausted"`, abandoning the current round
and leaving everything already saved intact. Why: an unattended loop must have a hard ceiling on spend.

### 2.9 Resume
Package: after the refiner stage the module's output (edits, candidate, diagnostics, trace ids and scores) is
checkpointed; after validation the evaluation is added; after the round a completion marker. A restart on the same
stores finishes the interrupted round by re-proposing the recorded candidate through the host and reusing the recorded
evaluation, without calling `collect` or the refiner again. An acceptance that landed before the crash is recognised
from the head and recorded without re-accepting. Why: rollouts and validation are the expensive stages; a lost lease
must not repeat them.

### 2.10 The host's candidate is canonical
Package: `GraphStore.propose` may return a graph that differs from the module's (a host that normalises, re-keys or
annotates). The harness continues with the host's graph and emits one warning naming the node/edge counts. Why: what
gets validated must be what the host will serve.

### 2.11 Stratified sampling, optional
Paper: the whole stride reaches the refiner. Package: `failing_sample` / `passing_sample` (`None` = paper) sample with
seeded rotation across rounds for hosts whose batches are large. Why: production batches can be thousands of episodes.

### 2.12 Redaction before the refiner
Package: `evolve(redact=...)` applies a `str -> str` redactor to every trace's text and step observations before the
refiner, the trace store or rejection memory see them. Why: production traces carry customer identifiers.

### 2.13 Node usage table
Package: when traces carry `nodes_visited` (copied from `Guide.visited`), the attempts block ends with a per-node table
of visits on failing versus passing traces. Why: the refiner is asked to prune nodes that "repeatedly steer trajectories
into failure"; the counts make that visible without reading every trace.

### 2.13a Relation labels in the serialized graph context
Paper: App. B.5 notes that its serializer does not print the stored relation labels (`LEADS_TO`,
`PROVIDES_INPUT_FOR`). Package: prints them on every transition line, `[A] → [B] (LEADS_TO; Condition: …)`;
`GuidanceConfig(include_relations=False)` and the CLI's `--paper-layout` reproduce the paper's serializer exactly
(tested byte for byte against the App. B.5 excerpt). Why: the refiner chose the relation and it carries information
(`PROVIDES_INPUT_FOR` versus `LEADS_TO` distinguishes a data dependency from a sequence); dropping it wastes what the
offline loop learnt.

### 2.14 Raw and localized-raw guidance modes, a no-graph mode, a per-episode call cap
Paper: Table 3 ablates raw full-graph injection and generative full/sub-graph guidance. Package adds `raw_subgraph`
(the serialized `N_h(u_t)` with no model call; not in the paper's ablation), `none` (the no-graph baseline for A/B
tests) and `max_guidance_calls` (after which a runaway episode is served raw context instead of paying again).
Why: guidance tokens dominate cost (paper §5.5, conclusion); hosts need the cheap options and a clean baseline.

### 2.15 Human-made initial graphs, and the skeleton for "empty"
Paper: expert graphs are hand-crafted files. Package: `initial_graph` (or CLI `init --graph`) loads the refiner shape,
adds a missing `Start`/`End` with a warning and fills missing attribute fields with `null`; anything else that fails
the structural checks (a node without a type, an edge without a relation, a missing endpoint) is refused with the
diagnostics, never completed; "empty" means the skeleton. Supplying an initial graph to a populated workspace is an
error. Why: onboarding a hand-written prior must be one command, and it must never silently overwrite a validated graph.

### 2.16 Transfer between workspaces
Package: `seed_workspace(revisions, source=, target=)` copies the source's head graph into an **empty** target as
revision 0 with `origin: transfer:<source>`; rejection memory never travels. Why: a new project starts from a sibling's
validated graph; the sibling's rejections were judged against a different validation set.

### 2.17 A command line
Package: `proceduralgraph --store file:DIR|postgres:URL --workspace WS init|show|export|history|diff|rejections|transfer|guide`.
`guide` shows exactly what the agent would be given at a point in a trajectory, with no model call. Why: operators
need the paper's file-level visibility over a chain they cannot open in an editor.

### 2.18 Hooks and metering
Package: `Hooks.stage(name)` wraps `baseline`, `rollout:k`, `refiner:k`, `validation:k`; `on_model_call`,
`on_iteration`, `on_warning`; `HookedModel` meters calls and output tokens, and a host passes the same wrapped model
to its `Guide` so guidance calls share the budget: when `evolve` receives a `HookedModel` it reuses that meter (adopting
`config.budget` if the host's meter has no ceilings) instead of wrapping the model again. The host's candidate returned
by `propose` is itself run through the structural checks before validation. Roles are `refiner` and `guidance` on
`ModelRequest.role`. Why:
leases, idempotency and per-role pricing are the host's, not the module's.

### 2.19 Bootstrapping a graph from a problem statement
Paper: a graph is either hand-crafted by an expert (Modes 1–3) or synthesised by the refiner from agent traces
(Modes 4–5). Package: `bootstrap_graph(problem_statement, solutions=(), tools=..., model=...)` and CLI
`init --from-text` run `refine_once` (G6) in the paper's `scratch_onetime` mode with the problem statement as the
task context and each worked solution presented as one successful trajectory (score 1.0, id `solution-N`). Two
placeholders are the only prompt text not in the paper: when there are no solutions the trajectories slot holds one
sentence asking the refiner to design the procedure from the task context and the tool list
(`bootstrap.NO_SOLUTIONS_BLOCK`), and the rejected-candidates slot holds `refine_once`'s usual "(none: one-time mode)"
placeholder because there are no earlier rounds. Without solutions the tool list is mandatory: the prompt's rule 1
needs it and there are no trajectories to infer it from. The result passes the same structural checks as a refiner
candidate and is refused with diagnostics otherwise; it is seeded with `origin: bootstrapped` plus the solution count,
mode and refiner-call count, on both the CLI path and `evolve(initial_graph=result)`. Like the paper's one-time modes
it is committed without a gate: the first `evolve` run scores it as the baseline and gates everything after. Why: a
team adopting the method usually has a task description and a few solved examples before it has an agent producing
traces; without this the only options are an expert's hand-written file or rounds of scratch evolution from
`Start → End`.

### 2.20 Quality and cost objectives
Paper: one scalar validation score and the `>=` rule (eq. 5). Package: an opt-in `ObjectiveGate` over an
`ObjectiveSpec` (ordered `MetricSpec`s with unit, direction, finite bounds, minimum meaningful improvement,
equivalence and non-regression margins, optional absolute mean floor and ceiling, optimize or report-only role; mode
`lexicographic` or `pareto`; comparison error budget `alpha`; minimum independent paired units) and an
`ObjectiveContext` (objective digest, evaluation design, evaluator version, cohort, immutable expected task ids,
budget profile, evidence partition, per-metric measurement basis). Hosts put structured observations on
`TaskOutcome.metrics` and the context on `Evaluation.objective_context`. The gate refuses any context mismatch,
incomplete or duplicated pairing, non-finite or out-of-bound value, and a deferred evaluation before it compares; then
checks absolute constraints on candidate means; then applies the mode rule over dependency-free two-sided Hoeffding
intervals (`paired_hoeffding_v1`: radius `width * sqrt(log(2/alpha_interval) / (2 n))`, `alpha_interval = alpha / M`
with `M` fixed for the whole objective). Six dispositions (`accepted`, `rejected`, `equivalent`, `unresolved`,
`unmeasured`, `invalid`) with stable reason codes and per-metric original-unit means and oriented intervals are
returned in `Decision.feedback` and persisted with rejection memory, iteration reports and the accepted revision.
Under this gate "all metrics equivalent" **retains the incumbent**, the opposite of eq. 5's tie acceptance; the
identity probe never stops the run, so perfect quality does not end cost optimisation. Zero equivalence margin makes
`equivalent` unreachable on finite samples: a metric can then only be passed by demonstrated improvement, which is
documented behaviour, not a defect. The uncertainty guarantee is for one fixed candidate and a predeclared paired
comparison on independent units; reusing the same validation tasks across proposals is a search, hosts own
qualification evidence, and acceptance selects a development graph only. Checkpoints written under an objective are
bound to its digests and refused under any other; a pending legacy checkpoint is refused in objective mode with a
message naming the workspace, never relabelled or deleted. Why: production hosts must hold quality while lowering
measured resource use, and must be able to say afterwards exactly which metric decided; a single scalar or a hidden
weighting cannot do either.

## 3. Deliberately unchanged

- The refiner is one single-shot JSON call: no ReAct proposer, no tools, no maintainer, no pruner.
- Tool-catalog membership of `ACTION` nodes is a prompt rule, not a structural check (App. B.6). `available_tools`
  only warns.
- The gate is `>=`. Ties are accepted (scalar mode; the opt-in objective gate retains the incumbent on equivalence, §2.20).
- The graph never changes inside an episode.
- Rejection memory is never rolled back or truncated by the loop.
- The one-time modes commit the refiner's output without a gate, exactly as the paper's ablation does, and are only
  reachable through `refine_once`.

## 4. Open

- **Layers / shared graphs.** Skill layers have no obvious graph analogue. Options for later: per-workspace overlays (a
  project graph whose nodes `ref` a tenant graph's nodes), or transfer only. 0.1 ships transfer only.
- **Semantic localization.** The paper's `Match` is exact. A host whose actions do not map 1:1 onto node ids may want
  an LLM or embedding localizer. `Localizer` is the seam; no model-based default is shipped.
- **Guidance reuse across steps.** Named as future work in the paper's conclusion. `raw_*` modes and
  `max_guidance_calls` are the only mitigations shipped.
- **Interop with a skill store.** A node with `type="SKILL"` and `ref=<skill name>` would let guidance say "apply skill
  X". Allowed by `Node.ref`, not exercised anywhere.
- **Graph-database backend.** A `GraphStore` over Neo4j / Memgraph / Postgres AGE is possible behind the protocol but
  not shipped: the paper's graphs are tiny (Table 7), the runtime never queries the store, and revision chains give the
  immutability and fork-refusal a graph DB would need rebuilt on top. Revisit only when a host's graph outgrows a single
  JSONB value or needs cross-graph queries.

---

Written by [Vikash Ranjan](https://www.linkedin.com/in/vikash-ranjan-stylsai/), CTO, [styls.ai](https://styls.ai). Apache-2.0.
