# proceduralgraph

Self-evolving Procedural Graphs for LLM agents: a per-step guidance runtime your agent loop calls, plus an offline,
gated refiner loop that edits the graph from execution traces and remembers what it rejected.

An independent, production-hardened reimplementation of *Procedural Graphs: Self-Evolving Execution Structures for
LLM Agents* (Lu, Chen, Wu, Arık; [arXiv:2609.09153](https://arxiv.org/abs/2609.09153)). The paper's authors published
no code; this package is built from the paper text alone, Apache-2.0. It follows the paper's mechanism exactly by
default and keeps every production option opt-in. Storage-agnostic (memory, files, PostgreSQL, S3, GCS behind one
`RevisionStore` protocol), provider-agnostic (`ChatModel`), budgeted, resumable. Built the way
[`skillwiki`](https://github.com/vikm2o/skillwiki) reimplemented WikiSkill, with storage shapes that let one host
adapter serve both packages; neither imports the other.

## What the paper does, and what this package keeps

| Paper | Package |
| --- | --- |
| A directed attributed graph `G = (V, R, E, Φ)`: nodes are tools, reasoning steps or task statuses; edges carry `condition`, `guidance`, `pitfalls` (§3.1) | `Graph`, `Node`, `Edge`; the four paper relations and three attribute fields by default, configurable per task |
| Online: localize the agent by exact match of its last action, serialize the 2-hop neighbourhood (or the full graph when matching fails), ask a guidance model, append `g_t` to the solver prompt (§3.2, eq. 2–3) | `Guide.guidance(query, trajectory)`; `h=2`, `w=3`; the App. B.5 guidance prompt verbatim; the App. B.5 serializer layout |
| Offline: rollout → refiner → PrepareCandidate → structural checks → validation → accept iff `>=` (ties accepted) → rejection memory (§3.3, Alg. 1) | `evolve(...)`, one round per Alg. 1 loop body, same order; `TieAcceptingGate` default; `RejectionMemory` persisted after every round |
| Refiner: one call, one JSON block with `add_nodes / delete_nodes / add_edges / delete_edges`; `delete_edges` removes every relation between the endpoints; attribute revision is delete + add (App. B.5) | `Refiner`, `EditSet`, `prepare_candidate`; the App. B.5 refiner prompt in the paper's order, plus one line naming the allowed node types |
| Structural checks: malformed edits, unknown node/relation types, missing endpoints, cycle policy, every node reaches a zero-out-degree terminal (App. B.6) | `Graph.validate()`, `Diagnostic` codes; `cycle_policy="repair"` removes cycle-closing edges as warnings |
| Trajectory context `Tail_Lmax(ConcatTrajectories)` keeps the end (§3.3 Step 4) | `render_attempts_block` + `tail`; 120,000 characters by default, tokens with a host counter |
| Strides `S=100` / `S=20`, ten rounds, refiner 8,192 tokens (App. B.1, D.3, §5.4) | `StridedTraceSource`, `max_rounds=10`, `refiner_max_tokens=8192` |

Where the paper is silent the package chooses, implements, tests and records the choice. Every departure and every
decision is in [`docs/paper-differences.md`](docs/paper-differences.md). [`docs/architecture.md`](docs/architecture.md)
has the diagrams: the per-step guidance flow, one round of Algorithm 1, the host seams, the storage chains, metering
and crash recovery.

## Install

```bash
pip install "proceduralgraph @ git+https://github.com/vikm2o/proceduralgraph@0.3.2"
pip install "proceduralgraph[postgres] @ git+https://github.com/vikm2o/proceduralgraph@0.3.2"   # production store
```

Extras: `postgres`, `s3`, `gcs`, `anthropic`, `openai`, `dev`. The core has no dependencies.

## Sixty-second example

Both halves: the `Guide` inside your solver loop, then `evolve` over the traces it produced. This is the shape of an
integration; the five `my_*` names are your code. The same thing complete and runnable without an API key is
`examples/filesystem_toy.py`.

```python
import asyncio
from proceduralgraph import (EvolveConfig, Evaluation, Guide, GuidanceConfig, Step, StaticTraceSource,
                             TaskOutcome, Trace, evolve)
from proceduralgraph.adapters.anthropic import AnthropicChatModel        # or implement ChatModel yourself
from proceduralgraph.stores import MemoryRevisionStore, RevisionGraphStore, RevisionRejectionStore

model = AnthropicChatModel("claude-sonnet-5")

# your code: my_solver_step(question, steps, guidance) -> (action, args, observation); my_scorer(question, steps) -> float;
#            my_render(steps) -> str; my_training_batch(iteration) -> list[str]; my_validation_questions -> list[str]

async def run_episode(question, graph, graph_ref):
    guide = Guide(graph, model, GuidanceConfig(task_description="two-hop QA"))   # frozen for the episode
    steps = []
    for _ in range(6):
        g = await guide.guidance(question, steps)                 # eq. 2: localize, serialize N_h(u_t), ask Ψ
        action, args, observation = my_solver_step(question, steps, g.text)   # eq. 3: g_t goes into the solver prompt
        steps.append(Step(action, args, observation))
        if action == "answer":
            break
    score = my_scorer(question, steps)
    return Trace(id=question, outcome=TaskOutcome(question, score, score >= 0.5), text=my_render(steps), steps=steps,
                 graph_ref=graph_ref, nodes_visited=guide.visited, guidance=[g.text for g in guide.guidance_log])

class MyTraceSource:                                                # Alg. 1 line 6: the training rollout
    async def collect(self, graph, *, graph_ref, iteration):
        return [await run_episode(q, graph, graph_ref) for q in my_training_batch(iteration)]

class MyEvaluator:                                                  # Alg. 1 lines 1 and 15: the validation split
    async def evaluate(self, ref, graph, *, iteration, purpose):
        traces = [await run_episode(q, graph, ref) for q in my_validation_questions]
        return Evaluation(ref=ref, score=sum(t.outcome.score for t in traces) / len(traces), per_task=[t.outcome for t in traces])

revisions = MemoryRevisionStore()                                   # or PostgresRevisionStore.from_url(...)
report = asyncio.run(evolve(
    config=EvolveConfig(task_description="two-hop QA", max_rounds=10),
    model=model,
    graph_store=RevisionGraphStore(revisions, "qa"),
    rejection_store=RevisionRejectionStore(revisions, "qa"),
    trace_source=MyTraceSource(),
    evaluator=MyEvaluator(),
    available_tools=["search", "read", "answer"],
))
print(report.outcomes(), report.best.score)
```

Start from a hand-written graph by passing `initial_graph=Path("expert.json")` (the refiner's `{nodes, edges}` shape;
a missing `Start`/`End` is added for you) or from nothing: an empty workspace seeds the `Start → End` skeleton and the
refiner runs in `scratch_incremental` mode. The solver prompt the paper uses is:

```
{system_prompt}
Procedural Graph Guidance: {procedural_graph_guidance}     <- g.text goes here
You must interleave Thought and Action. ...
Current Trajectory: {trajectory}
Thought:
```

`examples/filesystem_toy.py` runs the whole loop with a scripted model and no API key, producing one accepted round,
one structural failure, one rejection and one duplicate refusal, then exports `graph.json`, `graph.md`, `graph.mmd`
and `rejections.md`; `examples/anthropic_toy.py` does the same with a real model.

## Bootstrapping a graph from a problem statement

You do not need an expert or a trace corpus to get a first graph. Give the refiner the problem statement, the tool
list and, if you have them, a few worked solutions, and it drafts one with the paper's `scratch_onetime` call:

```python
from glob import glob
from proceduralgraph import bootstrap_graph, evolve

result = await bootstrap_graph(
    problem_statement=open("problem.md").read(),
    solutions=[open(p).read() for p in sorted(glob("solved/*.md"))],   # optional; each becomes one successful trajectory
    tools=["search", "read", "answer"],                                # required when there are no solutions
    model=model,
)
if result.ok:
    report = await evolve(..., initial_graph=result)                 # seeds with origin "bootstrapped", then gates every change
else:
    print(result.refusal())                                          # the structural diagnostics, never a half-valid graph
```

Or from the shell, then evolve against the same store:

```bash
proceduralgraph --store file:./ws --workspace qa init --from-text problem.md --solutions ./solved \
    --tools search,read,answer --model anthropic:claude-sonnet-5
```

A bootstrapped graph passes the same structural checks as any candidate and is refused with diagnostics otherwise. It
is a starting point, not a validated graph: like the paper's one-time modes it is committed without a gate, so the
first `evolve` run scores it as the baseline and every change after that is gated. Two placeholders are the only
prompt text not in the paper: one sentence in the trajectories slot when there are no worked solutions, and the
"(none)" placeholder in the rejected-candidates slot that `refine_once` always uses (paper-differences §2.19).

## Objectives: quality first, then cost

The paper gates on one scalar score. Production usually wants more than one: hold quality while lowering measured
resource use, or trade them off under explicit rules. Declare the objective once, put structured observations on each
task outcome, and use the opt-in `ObjectiveGate`:

```python
from proceduralgraph import EvolveConfig, MetricSpec, ObjectiveContext, ObjectiveGate, ObjectiveSpec, TaskOutcome, Evaluation

objective = ObjectiveSpec(
    metrics=(
        MetricSpec("quality", "fraction solved", "maximize", 0.0, 1.0, min_improvement=0.02, equivalence_margin=0.05, floor=0.8),
        MetricSpec("cost", "provider calls per task", "minimize", 0.0, 50.0, min_improvement=1.0, equivalence_margin=2.0, non_regression_margin=3.0),
        MetricSpec("latency", "seconds", "minimize", 0.0, 600.0, role="report_only"),
    ),
    mode="lexicographic",          # or "pareto": every optimized metric protected, at least one must improve
    alpha=0.05, min_units=30,
)
context = ObjectiveContext(objective_digest=objective.digest, evaluation_design="val-design-3", evaluator_version="scorer-2",
                           cohort="validation-a", expected_task_ids=validation_ids, budget_profile="prod-quota",
                           evidence_partition="2026-09", measurement_basis={"cost": "billed provider calls"})

class MyEvaluator:                                        # one TaskOutcome per expected task id, metrics in original units
    async def evaluate(self, ref, graph, *, iteration, purpose):
        outcomes = [TaskOutcome(t.id, t.score, t.passed, metrics={"quality": t.score, "cost": t.calls, "latency": t.seconds})
                    for t in run_validation(graph)]
        return Evaluation(ref=ref, score=None, per_task=outcomes, objective_context=context)

report = await evolve(config=EvolveConfig(objective=objective, objective_context=context), gate=ObjectiveGate(objective, context), ...)
print(report.iterations[0].disposition, report.iterations[0].decision["reasons"])
```

What you get: paired Hoeffding intervals per metric with the error budget split across the whole objective;
lexicographic (accept at the first demonstrated improvement, reject at the first demonstrated regression, pass a metric
only through equivalence) or pareto (every optimized metric held within its non-regression margin, at least one
improves; opposing changes are a `trade_off`); absolute floors and ceilings on candidate means; six dispositions
(`accepted`, `rejected`, `equivalent`, `unresolved`, `unmeasured`, `invalid`) with stable reason codes, persisted with
rejection memory, iteration reports and the accepted revision, and rendered to the refiner so it learns which metric
lost. Incomplete pairing, a changed cohort or evaluator, an unknown required cost or a deferred evaluation can never
promote a graph. Checkpoints are bound to the objective; a run with a different objective refuses to finish them.

Read [paper-differences §2.20](docs/paper-differences.md) for the exact rules and their limits: the guarantee is for one
fixed candidate on a predeclared paired comparison; reusing validation tasks across many proposals is a search, and
acceptance selects a development graph, not a release. `examples/objective_evolution.py` runs both modes, unknown cost,
restart recovery and deferred evaluation offline.

## Integrating with your own system

Implement small `async` protocols:

- `TraceSource.collect(graph, *, graph_ref, iteration) -> list[Trace]`: where traces come from. Run a rollout with a
  `Guide` over `graph`, or return production episodes that already ran under `graph_ref` and later received a score.
  Copy `guide.visited` and `guide.guidance_log` into the trace so the refiner sees which nodes were in play.
- `Evaluator.evaluate(ref, graph, *, iteration, purpose) -> Evaluation`: run the validation split. Return `per_task`
  outcomes to use `PairedGate`.
- `Gate.decide(best, candidate) -> Decision`: accept or reject. `TieAcceptingGate` (paper), `StrictImprovementGate`,
  `PairedGate` are provided.
- `ChatModel.complete(ModelRequest) -> ModelResponse`: one system prompt, one user message of text and image parts,
  text back. No tool definitions are ever sent.
- `Localizer.locate(graph, trajectory) -> str | None`: only if your actions do not map 1:1 onto node ids.

Stores: implement `RevisionStore` (four methods: `head`, `append`, `get`, `list`) and use `RevisionGraphStore` /
`RevisionRejectionStore` / `RevisionTraceStore` / `RevisionCheckpointStore`, or implement `GraphStore` /
`RejectionStore` / `TraceStore` directly against your own tables. `GraphStore` is the abstraction; the JSON document in
a Postgres JSONB column is its first implementation. `GraphStore.propose()` is where a host materialises a candidate in
its own format and `Hooks.stage()` is where it wraps paid stages in leases, checkpoints and spend metering.
[`docs/host-integration.md`](docs/host-integration.md) walks through the wiring.

Command line: `proceduralgraph --store file:DIR|postgres:URL --workspace WS init|show|export|history|diff|rejections|transfer|guide`.

## Fidelity notes

- Bootstrapping from a problem statement is not in the paper; it is `refine_once` in the paper's `scratch_onetime` mode
  with the problem statement as task context, plus two recorded placeholders (paper-differences §2.19).

- Guidance: `h=2` hops, `w=3` steps of trajectory, generative guidance over the local subgraph; the full graph when
  the last action matches no node. `raw_*` modes return the serialized context without a model call.
- Gate: accept iff the validation score matches or exceeds the retained graph's cached score; no early stop; all
  `K=10` rounds. `max_rejected_streak`, `PairedGate`, `ObjectiveGate` and the perfect-score stop are opt-in.
- Structural failures never reach validation and are recorded with diagnostics; a candidate identical to the head or
  to an earlier rejection is refused without validation.
- The refiner sees every trace in the batch, high-scoring first and low-scoring last, cut from the front at 120,000
  characters; one retry with the diagnostics fed back (`role_retries=0` for the paper's exact behaviour).
- The serializer prints relation labels; `include_relations=False` reproduces the paper's serializer exactly.

## Development

```bash
uv venv && uv pip install -e ".[dev,postgres,s3]"
uv run ruff check src tests examples
uv run pytest -q                    # Postgres and S3 contract tests run under testcontainers / moto, skipped without Docker
uv run python examples/filesystem_toy.py ./toy-workspace
uv build -o dist
```

Written by [Vikash Ranjan](https://www.linkedin.com/in/vikash-ranjan-stylsai/), CTO, [styls.ai](https://styls.ai). Apache-2.0.
