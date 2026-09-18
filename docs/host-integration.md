# Integrating proceduralgraph into a production host

How a production system wires the package. Everything here is host-agnostic; the seams are named generically.

## 1. The online half: guidance inside your agent loop

Construct one `Guide` per episode over the graph revision you are serving. The graph is frozen for the episode.

```python
from proceduralgraph import Guide, GuidanceConfig, Step

guide = Guide(graph, model, GuidanceConfig(hops=2, window=3, task_description=task.description), localizer=my_localizer)
steps: list[Step] = []
while not done:
    g = await guide.guidance(task.query, steps)      # eq. 2
    prompt = f"{system_prompt}\nProcedural Graph Guidance: {g.text}\n...Current Trajectory: {render(steps)}\nThought:"
    action, args = await solver(prompt)              # eq. 3
    observation = await environment(action, args)
    steps.append(Step(action=action, args=args, observation=observation))
```

- `Step.action` is what the localizer matches against node ids. If your tool-call format differs from the node ids
  (`search(q=...)` versus `search`), pass `ExactActionLocalizer(normalize=...)` or your own `Localizer`.
- Host status events (`Month_Start`) are steps whose `action` is the status id, so the graph can route on them.
- When the episode ends, copy `guide.visited` into `Trace.nodes_visited` and `[g.text for g in guide.guidance_log]`
  into `Trace.guidance`, and set `Trace.graph_ref` to the revision you served. The refiner uses the node visits; the
  loop uses `graph_ref` to detect stale traces.
- Cost controls: `GuidanceConfig(mode="raw_subgraph")` serves the serialized context with no model call;
  `max_guidance_calls` caps generative calls per episode; pass the loop's `HookedModel` as the guide's model to meter
  guidance under the same budget when the rollout runs in-process.

## 2. Traces from production

Implement `TraceSource.collect(graph, *, graph_ref, iteration)`. Two shapes work:

- **Rollout**: run a training batch now, with a `Guide` over `graph`, and return the traces. `StridedTraceSource`
  gives the paper's sequential strides over a fixed list.
- **Labelled production episodes**: return episodes that already ran under some revision and later received a score.
  Set `graph_ref` on each. `EvolveConfig.stale_trace_policy` decides what happens to traces from an older revision:
  `warn` (default; they are used and reported), `drop`, or `allow`.

Build `Trace(id, outcome=TaskOutcome(task_id, score, passed, prediction, truth), text, steps, graph_ref, nodes_visited,
guidance, meta)`. `text` is what the refiner reads; when it is empty the steps are rendered instead. Pass
`evolve(redact=regex_redactor({...}))` to strip identifiers from `text` and step observations before anything
persists them.

## 3. Validation and the gate

`Evaluator.evaluate(ref, graph, *, iteration, purpose)` runs your validation split under `graph` and returns
`Evaluation(ref, score, aggregate, per_task)`. Return `per_task` outcomes keyed by a stable `task_id` so `PairedGate`
can pair them. `purpose` is `"baseline"` for the head and `"candidate"` for a proposal; if you already scored the
head, pass `baseline=Evaluation(...)` to `evolve` and skip the baseline stage.

Gates: `TieAcceptingGate()` (the paper), `StrictImprovementGate()`, `PairedGate(min_win_probability=0.9,
min_tasks=20)`. All take `perfect=` for an early stop; the default gate leaves it off.

## 3a. Objective mode

To evolve toward better outcomes *and* lower measured resource use, declare an `ObjectiveSpec` and an
`ObjectiveContext` once per run and pass `gate=ObjectiveGate(objective, context)` with
`EvolveConfig(objective=..., objective_context=...)`. Your evaluator then returns one `TaskOutcome` per expected task id
with `metrics={name: value}` in the declared units (`None` for an explicit unknown) and `objective_context=context` on
the `Evaluation`; `score` may stay `None`. Rules the loop enforces in this mode:

- `Evaluation.ref` must be the ref the loop asked you to evaluate (the head ref for the baseline, the candidate's
  `propose` ref otherwise). A host-supplied `baseline=` must carry the head ref too.
- One task id is one independent analysis unit. Aggregate repeated runs inside a unit with a predeclared rule before
  building the outcome; include failed and no-change paid attempts in the totals.
- Measure what was actually consumed. Fewer steps or tokens alone are not a monetary saving; if you compare money, the
  `measurement_basis` must say what basis, and it must be identical on both sides.
- Checkpoints are bound to the objective and context digests. Changing any metric, margin, bound, mode, alpha, cohort,
  evaluator, budget profile, partition or basis is a new run in a fresh workspace; the pending record is kept.
- The refiner is shown the frozen objective and every decision's per-metric comparison inside the existing prompt
  slots; observations cannot edit the objective.
- Library acceptance selects a development graph. Promotion to production, qualification on fresh evidence and any
  sequential or multi-candidate selection design remain yours.

Feedback lives on `IterationReport.decision`, `RejectionEntry.decision` (with `baseline_ref` / `candidate_ref`) and the
accepted revision's `meta["decision"]`; the CLI `rejections` view and `export` show the same lines.

## 4. The model transport

`ChatModel.complete(ModelRequest) -> ModelResponse`. Requests carry one system string, one user message of text and
image parts, `max_tokens` and `role` (`"refiner"` or `"guidance"`), and never any tool definitions. A metered or
audited transport that allows exactly one user message and forbids tools works unchanged. Return `.text` and a
`.usage` dict with `output_tokens` (or `completion_tokens`) for the budget to count. The same adapter class serves
`skillwiki` by duck typing.

## 5. Stores

Two options.

**Implement `RevisionStore` once** (four methods: `head`, `append`, `get`, `list`) and use the default layer stores:

```python
from proceduralgraph.stores import RevisionCheckpointStore, RevisionGraphStore, RevisionRejectionStore, RevisionTraceStore
from proceduralgraph.stores.postgres import PostgresRevisionStore

revisions = PostgresRevisionStore.from_url("postgresql+asyncpg://user:pw@host/db")   # table="proceduralgraph_revisions"
await revisions.create_tables()
graph_store = RevisionGraphStore(revisions, workspace)
rejection_store = RevisionRejectionStore(revisions, workspace)
trace_store = RevisionTraceStore(revisions, workspace)
checkpoint_store = RevisionCheckpointStore(revisions, workspace)
```

The graph is one JSONB value per accepted revision; only accepted graphs join the chain. The `(workspace, kind, seq)`
primary key refuses forks. A host that already runs `skillwiki` can point both packages at one table: the chain kinds
(`graph`, `rejections`, `traces`, `checkpoints`) do not collide with skillwiki's.

**Or implement the layer protocols against your own tables.** `GraphStore` is the abstraction; the JSON document is
its first implementation. Its four methods:

- `load() -> (ref, graph)`: the head, or `(None, Graph.skeleton())` for an empty workspace.
- `seed(graph, *, meta) -> ref`: revision 0; raise `HeadMoved` if a graph exists.
- `propose(current_ref, current, edits, candidate, *, iteration, rejections_ref) -> Candidate`: materialise the
  candidate in your format (a row, a bundle, a versioned object). **Idempotent per `(iteration, edits)`**: a resumed
  run calls it again for the same round. Whatever graph you return is what gets validated and, if accepted, served.
- `accept(candidate, *, expected_ref) -> ref`: make it the head; refuse when the head moved.

`RejectionStore.load()/save()` persists `RejectionMemory.to_document()`; it is saved once per round after the outcome
is known. `CheckpointStore.load(iteration)/save(iteration, payload)` enables resume. `TraceStore` is optional
(`NullTraceStore` when your raw episodes live elsewhere).

## 6. Leases, idempotency, budgets

`Hooks.stage(name)` wraps every paid stage: `baseline`, `rollout:k`, `refiner:k`, `validation:k`. Acquire a lease
there, and release it on exit. `on_model_call(role, request, response)` sees every model call with its usage;
`on_iteration(report)` sees each `IterationReport`; `on_warning(message)` sees every warning.

`EvolveConfig.budget = Budget(max_model_calls, max_output_tokens, max_evaluations, max_seconds)` is checked before
every paid step; exhaustion stops the run cleanly with `stopped_reason="budget_exhausted"`. With a `CheckpointStore`,
a run that stops or dies after the refiner stage is finished by the next run on the same stores: `propose` is called
again for the same round, the recorded evaluation is reused, and neither `collect` nor the refiner runs again.

## 7. Initialising a workspace

- From nothing: `evolve(...)` on an empty workspace seeds `Start → End` and the refiner runs in `scratch_incremental`.
- From a hand-written graph: `evolve(initial_graph=Path("expert.json"))` or `proceduralgraph --store ... init --graph
  expert.json`. The file is the refiner's `{"nodes": [...], "edges": [...]}` shape; a missing `Start`/`End` is added
  with a warning, anything that fails the structural checks is refused. The refiner then runs in `static_incremental`.
- From a problem statement: `result = await bootstrap_graph(problem_statement=..., solutions=[...], tools=[...], model=model)`
  then `evolve(initial_graph=result)`, or `proceduralgraph --store ... init --from-text problem.md --solutions ./solved
  --tools a,b --model anthropic:ID`. Each worked solution is presented to the refiner as a successful trajectory; the
  tool list is mandatory when there are no solutions. Both paths seed with `origin: bootstrapped`. The graph is a
  starting point, not a validated one: the first `evolve` scores it as the baseline. Pass the loop's `HookedModel` as
  `model` to meter the bootstrap call under the same budget.
- From a sibling workspace: `seed_workspace(revisions, source="tenant-a", target="tenant-b")` or the CLI `transfer`.
  Rejection memory does not travel.

Supplying `initial_graph` to a workspace that already has a graph is an error, never an overwrite.

## 8. Operating it

```bash
proceduralgraph --store postgres:postgresql+asyncpg://user:pw@host/db --workspace tenant-a show
proceduralgraph --store postgres:... --workspace tenant-a rejections --refiner-view
proceduralgraph --store postgres:... --workspace tenant-a guide --query "…" --step search --step read
proceduralgraph --store postgres:... --workspace tenant-a diff --from 3 --to 4
proceduralgraph --store postgres:... --workspace tenant-a export ./out    # graph.json, graph.md, graph.mmd, rejections.md
```

---

Written by [Vikash Ranjan](https://www.linkedin.com/in/vikash-ranjan-stylsai/), CTO, [styls.ai](https://styls.ai). Apache-2.0.
