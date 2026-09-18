# proceduralgraph: requirements for the implementing agent

**Status:** requirement only. Nothing in this repository is built yet. Written 2026-09-17.
**Deliverable:** a Python package `proceduralgraph` that is an independent, production-hardened reimplementation of
*Procedural Graphs: Self-Evolving Execution Structures for LLM Agents* (Lu, Chen, Wu, Arık; arXiv:2609.09153),
built the same way `skillwiki` reimplemented WikiSkill: the paper's mechanism kept exactly by default, storage
behind an immutable digest-chained `RevisionStore`, every host seam a small `Protocol`, every departure from the paper
recorded with its reason.

Read this file top to bottom before writing code. Section 3 is the paper, quoted. Section 5 is the numbered
requirement list; MUST is binding, SHOULD is expected unless you record why not, MAY is optional. Section 8 is the
acceptance checklist the work is judged against.

---

## 1. Sources in reach

| Source | Where | Use it for |
| --- | --- | --- |
| The paper, full text | `docs/paper/2609.09153v1.txt` (see `docs/paper/README.md` for the section map) | Every mechanism, prompt, threshold and algorithm. Quote it; do not work from memory. |
| `skillwiki` 0.2.0, the pattern to follow | public: `https://github.com/vikm2o/skillwiki` tag `v0.2.0`; locally `~/skillwiki` if present | Package layout, store protocols, `ChatModel`, `Budget`, `Hooks`, `CheckpointStore`, gates, CLI, `docs/paper-differences.md` format, test style. Section 6 of this file embeds the parts you must match byte-for-byte, so the network is not required. |

Do **not** import `skillwiki`. Do **not** copy any third-party code. The paper's authors published no code; this is
an independent reimplementation from the paper text, Apache-2.0, like `skillwiki`.

## 2. Identity and tooling (mirror skillwiki exactly)

| Item | Value |
| --- | --- |
| Package / import name | `proceduralgraph` (checked 2026-09-17: free on PyPI and as `github.com/vikm2o/proceduralgraph`; `procgraph` is taken) |
| Repo | this directory, `~/proceduralgraph`, branch `main`. The user commits and tags; you do not commit unless asked. |
| Version | `0.1.0` |
| Build | hatchling, `src/proceduralgraph/` layout, `py.typed`, Python `>=3.11` |
| Dependencies | core: none. Extras exactly as skillwiki 0.2.0: `postgres = ["sqlalchemy[asyncio]>=2.0", "asyncpg>=0.29"]`, `s3 = ["boto3>=1.34"]`, `gcs = ["google-cloud-storage>=2.16"]`, `anthropic = ["anthropic>=0.40"]`, `openai = ["openai>=1.40"]`, `dev = ["pytest>=8.2", "pytest-asyncio>=0.23", "ruff>=0.6", "moto[s3]>=5.0", "testcontainers[postgres]>=4.5"]` |
| `pyproject` details | `[project.scripts] proceduralgraph = "proceduralgraph.cli:main"`; `[tool.ruff] line-length = 120, target-version = "py311"`; `[tool.ruff.lint] select = ["E4", "E7", "E9", "F", "I", "UP", "B"]`; classifiers and sdist include list as skillwiki's |
| Licence | Apache-2.0. `LICENSE` file. One-line header on every `.py` under `src/`, `tests/`, `examples/`: `# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.` |
| Authorship | `Vikash Ranjan (CTO, styls.ai)` in `pyproject.toml` authors, `__author__`, README footer (link `https://www.linkedin.com/in/vikash-ranjan-stylsai/`, `https://styls.ai`) |
| Tooling | `uv venv && uv pip install -e ".[dev,postgres,s3]"`, `ruff check src tests examples`, `pytest -q`, `uv build -o dist` |
| Distribution | git tag, not PyPI (hosts pin `proceduralgraph @ git+https://github.com/vikm2o/proceduralgraph@v0.1.0`) |
| `.gitignore` | already present; keep `HANDOFF.md` ignored (host-specific notes never go public) |

Keep everything in this repo host-agnostic. Describe the first production host's seams generically (a metered
transport that forbids tool definitions, leases around paid stages, labelled production episodes, per-task scores)
and never by system name.

## 3. The paper's mechanism, precisely

This section is the specification of default behaviour. Section references are to the paper (v1). Where this
section and your memory of the paper disagree, this section and `docs/paper/2609.09153v1.txt` win.

### 3.1 Data model (§3.1, App. B.4, App. B.5)

> Formally, a Procedural Graph is a directed, attributed graph G = ⟨V, R, E, Φ⟩, E ⊆ V × R × V, where V is the set of
> abstract nodes, R is a vocabulary of transition relations, and each element of E is a directed, attributed triplet:
> an edge e = (u, r, v) ∈ E states that node v is admissible after node u under relation r. Each node abstracts a tool
> function, a skill, an internal reasoning step, or a task status. The attribute mapping Φ associates each edge with
> a set of named attributes whose schema can be specified for the task. In our implementation, we use three textual
> fields: condition, guidance, and pitfalls, describing when the transition applies, how to proceed, and what to avoid.

- **Node:** `{"id": str, "type": str, "description": str}`. Types in the paper: `ACTION` (must name an available tool
  action, refiner rule 1), plus reasoning steps and task statuses (`Start`, `End`, `Month_Start`, `Decide_Capital`
  are examples of non-ACTION nodes in App. E.3). Node ids are stable identifiers the environment's state tracker can
  match (refiner rule 6: never rename).
- **Edge:** `(source, relation, target)` + attributes `condition` (natural-language precondition or `null` when
  unconditional), `guidance` (mandatory on every added edge), `pitfalls`. The attribute schema is configurable per
  task; these three are the default.
- **Relations** used across all paper graphs (App. B.4): `LEADS_TO`, `TRIGGERS`, `PROVIDES_INPUT_FOR`, `CONVERGES_TO`.
- **Sentinels:** `Start` (a₀ = Start, §3.2) and `End`. Minimal skeleton `G_skeleton = (Start → End)` (App. D.2).
- **Terminal node:** zero out-degree (App. B.6). "This is a reachability check to a terminal node, not specifically
  to the node named End."
- **Sizes:** 7 to 17 nodes and 7 to 27 triplets on six of seven benchmarks; 131 nodes / 265 triplets on BFCL (Table 7).
- **Refiner-facing JSON** (`{current_graph_json}`, App. B.5) uses the same shapes as the edit arrays:
  `{"nodes": [{"id","type","description"}], "edges": [{"source","target","relation","condition","guidance","pitfalls"}]}`.

### 3.2 Online guidance (§3.2, eq. 2 and 3; §4; §5.5; App. B.5)

> Let q be the user query and T_t = (a₁, o₁, …, a_{t−1}, o_{t−1}) be the interleaved history of actions and
> observations up to decision step t. We use a₀ = Start as an initialization marker, so the first step is localized at
> u₁ = Start. At each step,
> u_t = Match(a_{t−1}, V),  G_t = N_h(u_t) if u_t ≠ ∅, else G,  g_t = Ψ(G_t, q, T_{t−w:t}),   (2)
> where Match locates the agent by exactly matching its most recent procedure (e.g., a tool call) to a node in V. The
> directed edge neighborhood N_h(u_t) contains u_t and the outgoing transitions reached by expanding for up to h steps.
> The window T_{t−w:t} contains the last w trajectory steps, and Ψ is the guidance language model.
> The guidance g_t is appended to the task solver's prompt. The solver then selects its next action from the query,
> trajectory, and guidance: a_t ∼ P_solver(· | q, T_t, g_t).   (3)

- Defaults: `h = 2`, `w = 3` (§4). The graph is frozen during an episode (§3, "Online Inference").
- Fallback: when `Match` fails, the guidance model receives the **full graph** (eq. 2, "otherwise").
- The guidance model and refiner share the solver's LLM in the paper; greedy decoding, temperature 0 (§4).
- Modes evaluated in Table 3 (Gemini 3.5 Flash): no graph; full graph raw injection; full graph generative; subgraph
  generative (paper's default, best on all three benchmarks). Generative guidance costs tokens: total tokens 33.4%
  (GDPval) and 55.4% (ALFWorld) above the no-graph baseline while solver steps fell (28.20 → 18.57, 21.84 → 18.80).
  Localisation cut tokens 70.9% / 18.1% / 14.8% against full-graph generative.

**Guidance Generation Prompt (Local subgraph; default)** — App. B.5, verbatim:

```
You are an expert cognitive architect and execution guide for an AI agent solving the task: {task_description}
Here is {graph_context_desc}: {subgraph_summary}
Here is the current active query / observation: {query}
Here is the agent's recent execution trajectory: {recent_context}
Analyze this {graph_source} in the context of the agent's current progress. Using the condition, guidance,
and pitfalls attributes carried by the edges in the graph context, generate clear, detailed, and actionable guidance
advising the agent on exactly what step or strategy to pursue next, what pitfalls to avoid, and how to recover from
recent failures if any. You must include any specific command patterns, file paths, tools, or arguments defined in the
graph context if they are relevant to the next steps.
```

Full-graph variant: identical text; `{graph_context_desc}` = "the complete Procedural Graph governing the task
structure and strategic guidance", `{subgraph_summary}` = the full graph summary, `{graph_source}` = "complete
Procedural Graph".

**Serialized local graph context** — App. B.5, the paper's own serializer output (HotpotQA excerpt):

```
Active Cognitive Node: [First_Hop_Retrieve] (Type: ACTION)
Description: Execute first_hop_retrieve to fetch primary evidence passages.
Immediate Transition Options (Hop 1):
- Transition: [First_Hop_Retrieve] → [Scan_Index] (Condition: first_hop_retrieve)
* Guidance: Review the retrieved primary passages via Scan_Index to locate specific
bridge terms (such as birth dates, locations, or associated entities).
* Pitfalls to Avoid: Do not skip reading evidence details; missing the exact bridge
entity name causes second-hop search failure.
Subsequent Horizon (Hop 2):
- Transition: [Scan_Index] → [Bridge_Extract] (Condition: scan_index)
* Guidance: Extract the explicit connecting entity or bridge term linking the first
passage to the target question.
* Pitfalls to Avoid: Ensure the extracted bridge term matches exact Wikipedia
capitalization conventions.
```

"The two stored relation labels, LEADS_TO and PROVIDES_INPUT_FOR, are not printed by this serializer." (Decision for
this package: print them; see §5.C.)

**Solver Execution Prompt (ReAct loop)** — App. B.5. The package does not run the solver (hosts own their agent), but
the README must show where `g_t` goes:

```
{system_prompt}
Procedural Graph Guidance: {procedural_graph_guidance}
You must interleave Thought and Action. Your output format must be exactly:
Thought: <your reasoning about what to do next>
Action: <tool_name>(arg1=val1, arg2=val2, ...)
Example: …
DO NOT write any "Observation:" block or any subsequent steps. Only output exactly one Thought and one Action
block. Do NOT simulate the environment's responses.
Current Trajectory: {trajectory}
Thought:
```

### 3.3 Offline self-evolution (§3.3, eq. 4 to 6; App. B.6 Algorithm 1)

> Let G₀ be the initial Procedural Graph and G_k the retained graph after round k. Each round starts from G_{k−1}; a
> rejected candidate never becomes the starting graph of the next round.

**Step 1, Diagnostic Rollout.** Solver runs on a training batch B_k ⊂ D_train with G_{k−1}; record traces and scores
E_k = {(q_i, T_i, S_i)}, S_i ∈ [0, 1]. "The refiner compares high-scoring traces with low-scoring ones; for tasks
with binary outcomes, this reduces to successes versus failures." Batches are sequential strides of the training
split: S = 100 (HotpotQA), S = 20 (MultiChallenge) (App. B.1, D.2).

**Step 2, Feedback-Driven Mutation.** One LLM refiner call over the partitioned traces produces a structured edit set
ΔG_k with two topological operations, Add and Delete. "Attribute revisions use the same edit interface: an edge is
deleted and re-added with updated attribute values." Candidate `G_k^cand = G_{k−1} ⊕ ΔG_k`, "where ⊕ applies edits to
a copy and performs any configured cycle repair."

**Step 3, Validation Gating.** Structurally valid candidates are evaluated on a held-out D_val:
`S_val(G) = (1/|D_val|) Σ S(f_solver(q | G), y)` (eq. 4). "The initial graph is evaluated once to establish the
reference score." Accept iff `S_val(G_k^cand) ≥ S_val(G_{k−1})` (eq. 5). **Ties are accepted.** Invalid candidates are
discarded before any validation rollout; the retained graph and its cached score are unchanged.

**Step 4, Rejection Memory.** A rejected candidate is logged with its edits, the training trajectories E_k and the
validation outcome into H_rejected. Trajectory context for the refiner: concatenate training trajectories and, when
the configured maximum token length L_max is exceeded, discard tokens from the **beginning**, preserving the final
L_max tokens in order (C_k). `ΔG_{k+1} ∼ P_refiner(· | G_k, C_{k+1}, H_rejected)` (eq. 6).

**Algorithm 1 (App. B.6)** — verbatim:

```
Require: Initial graph G0; training/validation sets Dtrain, Dval; round budget K; trajectory limit Lmax; cycle policy c.
 1: S0 ← Evaluate(G0, Dval)
 2: Hrejected ← [ ]
 3: for k = 1, …, K do
 4:    Gk ← Gk−1; Sk ← Sk−1                      {Retain unless accepted}
 5:    Select training batch Bk ⊂ Dtrain
 6:    Ek ← Rollout(Gk−1, Bk)                     {Training traces and scores}
 7:    Ck ← TailLmax(ConcatTrajectories(Ek))
 8:    Rk ← SerializeRejections(Hrejected)
 9:    ΔGk ← Refiner(Gk−1, Ck, {Si(k)}i, Rk)
10:    (Gkcand, dk) ← PrepareCandidate(Gk−1, ΔGk, c)
11:    if dk ≠ ∅ then
12:        Append (ΔGk, Gkcand, Ek, dk) to Hrejected
13:        continue                              {No validation rollout; retained state is unchanged}
14:    end if
15:    Skcand ← Evaluate(Gkcand, Dval)
16:    if Skcand ≥ Sk−1 then
17:        Gk ← Gkcand; Sk ← Skcand              {Accept, including ties}
18:    else
19:        Append (ΔGk, Gkcand, Ek, Skcand) to Hrejected
20:    end if
21: end for
22: return GK
```

> **Candidate Preparation and Structural Checks.** PrepareCandidate applies edits to a copy of the retained graph,
> deleting edges and nodes before adding nodes and edges. It reports malformed edits, invalid node or relation types,
> and missing edge endpoints as failures. When cycles are disallowed, the implementation removes detected
> cycle-closing edges before validation; when cycles are allowed, that repair and the acyclicity check are skipped.
> The remaining checks require valid edge endpoints and a directed path from every node to a terminal node, defined by
> zero out-degree. … Matching action-node names to the available tool list is a refiner-prompt requirement; the
> generic structural validator does not independently enforce tool-catalog membership. On failure, d_k contains
> diagnostics and G_k^cand may be unavailable; on success, d_k = ∅.

> Each entry in delete_edges removes all edges with the specified source and target, regardless of relation. To
> retain selected transitions between the same endpoints, include them in add_edges, which is applied after deletion.

**Refiner Prompt (Self-evolution)** — App. B.5, verbatim:

```
You are an expert cognitive architect optimizing a Procedural Graph for an intelligent agent. The Procedural Graph
encodes structured procedural guidance.
Task context: {task_description}
Refinement mode: {mode}
Available Tool Actions (the agent can only execute these actions): {available_tools_list}
Recent execution trajectories: {attempts_block}
Current Procedural Graph representation: {current_graph_json}
Previously rejected candidates: {rejected_block}
Your job is to refine the Procedural Graph. Follow these guidelines based on the mode:
• static_onetime / static_incremental: Prune edges/nodes that lead to loops, deadlocks, or failures.
  Add missing nodes and edges that could fix the failures and improve performance for future tasks.
• scratch_onetime / scratch_incremental: If starting from scratch (the graph contains only
  Start → End), synthesize a brand new, complete Procedural Graph using the Available Tool Actions list, Status,
  and successful patterns in the trajectories. Otherwise, prune edges/nodes that lead to loops, deadlocks, or
  failures, and add missing nodes and edges based on the given graph.
Rules for nodes and edges. Rules 2–4 describe the edge attributes in Φ(e): condition, guidance, and
pitfalls. The remaining rules govern node compatibility, generality, and graph structure.
1. Action Nodes. Any node of type ACTION must match one of the action/tool names in the "Available Tool Actions"
   list above.
2. Transition Conditions. If an edge has a condition, provide a natural-language semantic precondition under
   which this transition should fire (e.g., "When dialogue history has been parsed but target constraints are
   unknown"). Use null if the transition is unconditional.
3. Execution Guidance. For every edge added in add_edges, you MUST provide a guidance string detailing
   exactly what action to take next and the strategic rationale behind it.
4. Pitfalls. Provide a pitfalls string warning about premature actions, forbidden words, or common formatting
   pitfalls to avoid during this step.
5. Generality & Leak Prevention. The updated Procedural Graph must guide the agent effectively without
   overfitting to specific details of a single trajectory. Use high-level conceptual descriptions.
6. Node ID Compatibility. If refining an existing graph (static modes), you MUST preserve the existing node IDs
   (such as Month_Start, Decide_Capital, and the tool names) so they remain compatible with the
   environment's state tracker. Do not rename them.
7. Graph Structure. Follow the task's configured cycle policy. Every edge must reference existing nodes, and every
   node must have a directed path to a terminal node. The environment loop handles repetition across simulation
   cycles.
Please propose the exact set of edits to perform. You must output your edits as a single valid JSON block containing
four arrays: add_nodes, delete_nodes, add_edges, and delete_edges. Output format must be exactly:
{ "add_nodes": [{"id":..., "type": "ACTION", "description":...}],
  "delete_nodes": ["node_id"],
  "add_edges": [{"source":..., "target":..., "relation":...,
                 "condition":..., "guidance":..., "pitfalls":...}],
  "delete_edges": [{"source":..., "target":...}] }
Make sure to output ONLY the raw JSON block.
```

**Construction modes (App. D.2):** `static_onetime` (expert prior, one refiner pass over all traces, no gate),
`static_incremental` (expert prior, strides, gate), `scratch_onetime` (Start→End, one pass, no gate),
`scratch_incremental` (Start→End, strides, gate). Only the incremental modes are Algorithm 1; the one-time modes commit
the refiner's output directly and are the paper's ablation, not its method.

**Other numbers (App. D.3, §5.4):** refiner max generation 8,192 tokens; agent 2,048; ten rounds in the evolution
study; validation set of 20 episodes on EnterpriseArena means single accept/reject decisions "turn on one or two
episodes and should be read as a search trace rather than as significance tests" (§5.4).

### 3.4 What the paper does not specify

Record each of these as a package decision in `docs/paper-differences.md`, not as fidelity:

- L_max value, and whether the tail cut is by tokens or characters.
- Concatenation order of trajectories before the tail cut.
- How many rejected candidates the refiner sees and in what detail.
- What happens on an empty edit set, malformed JSON, or a duplicate of an already-rejected candidate.
- Whether the refiner is retried when its output cannot be applied.
- Guidance-call output length.
- Any early-stop rule (Algorithm 1 runs all K rounds).
- Whether nodes must also be reachable from `Start` (only reachability **to** a terminal is checked).
- The node-type vocabulary. Only `ACTION` is named; "Status" appears once in the refiner prompt. The package fixes a
  default list and tells the refiner (A1, E1).
- Whether `End` must exist. The terminal check is zero out-degree "not specifically the node named End" (App. B.6);
  the package requires `End` and protects it (A6, B2).
- What "high-scoring" means for non-binary scores (`success_threshold`, G2).
- What the refiner sees of a rejected candidate: the paper says "candidate graphs"; the package renders the edit set
  plus score or diagnostics (F2).
- Whether accepted rounds are recorded alongside rejections (F1).

## 4. Where this is NOT WikiSkill

The implementing agent will know `skillwiki` well. These are the places where cloning its shape would be wrong:

1. **There is an online, per-step component.** The solver reads graph-derived guidance at every decision step
   (eq. 2, 3). In WikiSkill the inference agent never sees the wiki. This package therefore ships a storage-free
   runtime (`Guide`) that a host calls inside its own agent loop, and the offline loop only ever hands the host a
   frozen `Graph`.
2. **The gate accepts ties** (`≥`, eq. 5). WikiSkill's rule is strict `>`. The paired-bootstrap gate stays the
   production option.
3. **Rejection memory is a first-class persistent artefact**, not an impact log: it stores full candidate graphs,
   their edit sets, the training trace ids and either the validation score or the structural diagnostics
   (Alg. 1 lines 12 and 19). It is the *only* persistent memory besides the graph; there is no wiki.
4. **One single-shot JSON refiner role.** No maintainer, no pruner, no ReAct proposer with tools, no minimum-trace-read
   rule. The refiner receives everything in one message and answers with one JSON block.
5. **The artefact is a graph, not markdown.** There is no `SKILL.md`, no frontmatter, no patch operations. Edits are
   `add_nodes / delete_nodes / add_edges / delete_edges`. The "file rendering" is `graph.json` plus a human-readable
   serialization, not a directory of markdown pages.
6. **Structural validation replaces the proposal validator.** A candidate can fail before validation for graph
   reasons (missing endpoint, unknown relation, node with no path to a terminal), and that failure is itself
   recorded in rejection memory with diagnostics.
7. **No layers.** Skill layers do not map onto a single graph. Leave it as an open question (§9), do not invent one.

## 5. Requirements

Numbering: `A1`, `B3`, … Reference these ids in tests and in `docs/paper-differences.md`.

### A. Graph data model and documents

- **A1 (MUST)** `Graph` is an immutable value object: `nodes: dict[str, Node]`, `edges: list[Edge]`, `relations:
  tuple[str, ...]` (default the four paper relations), `attribute_fields: tuple[str, ...]` (default
  `("condition", "guidance", "pitfalls")`), `node_types: tuple[str, ...]` (default at least `ACTION`, `REASONING`,
  `STATUS`; `Start` and `End` are `STATUS`; a host extends the tuple per task). The configured list is **injected
  into the refiner prompt** (E1) so the refiner is told what A6 will enforce. Every mutation returns a new graph; `Guide` and `evolve` never mutate in
  place.
- **A2 (MUST)** `Node(id, type, description)`; `Edge(source, relation, target, attributes: dict[str, str | None])`.
  `Edge.condition / guidance / pitfalls` are convenience accessors over `attributes` when those fields exist.
- **A3 (MUST)** `Graph.skeleton()` returns `Start → End` with one `LEADS_TO` edge and empty attributes. `Graph.is_skeleton`
  is true iff the node set is exactly `{Start, End}` and there is one edge (structural test, not a digest comparison).
- **A4 (MUST)** `Graph.to_document()` / `from_document()`: JSON with `"schema": 1`, nodes sorted by id, edges sorted by
  `(source, relation, target)`, so `Graph.digest` (SHA-256 of canonical JSON, §6.1) is content-addressed and
  order-independent. `to_refiner_json()` renders the App. B.5 `{current_graph_json}` shape.
- **A5 (MUST)** Duplicate `(source, relation, target)` triples are invalid. Multiple relations between the same
  endpoints are valid (the paper's `delete_edges` semantics imply it).
- **A6 (MUST)** Graph invariants checked by `Graph.validate() -> list[Diagnostic]`: every edge endpoint exists; every
  relation is in `relations`; every node type is in `node_types`; `Start` and `End` exist; every node has a directed
  path to some zero-out-degree node. **Decision, record it**: requiring `End` to exist (and B2 protecting it) is
  stricter than the paper, whose terminal check is zero out-degree "not specifically the node named End"; the
  skeleton, refiner rule 6 and the transfer path all assume the sentinel, so the package keeps it.
  `Diagnostic(code, message, subject)` with stable codes: `missing_endpoint`,
  `unknown_relation`, `unknown_node_type`, `duplicate_node`, `duplicate_edge`, `reserved_node`,
  `no_path_to_terminal`, `malformed_edit`. Reachability **from** `Start` is a warning (`unreachable_from_start`),
  never a failure (§3.4). Node types and relations are matched **case-insensitively** and stored upper-case: the
  paper's refiner prompt writes `ACTION` but also `Status`, and a case difference must not become a round-1
  structural failure. Record as a decision.
- **A7 (SHOULD)** `Graph.neighborhood(node_id, hops)` returns the sub-graph `N_h(u)`: `u` plus all nodes and edges
  reached by following **outgoing** edges up to `hops` steps, tagging each edge with its hop distance for the
  serializer. Deterministic order: BFS by hop, then edge sort order.
- **A8 (SHOULD)** A node MAY carry `ref: str | None` (opaque host identifier, e.g. the name of a skill in another
  system) and `meta: dict`. Both are excluded from `digest` only if the host asks (`digest(exclude_meta=True)`);
  by default everything in the document is digested, as in skillwiki.

### B. Edits, candidate preparation and structural checks (Alg. 1 line 10, App. B.6)

- **B1 (MUST)** `EditSet` is the refiner's JSON: `add_nodes: list[Node]`, `delete_nodes: list[str]`, `add_edges:
  list[Edge]`, `delete_edges: list[{source, target}]`. `EditSet.parse(text) -> EditSet` accepts the raw JSON block
  (tolerating surrounding prose and code fences), raising `EditError(diagnostics=[malformed_edit …])` otherwise.
  `EditSet.is_empty` is true when all four arrays are empty.
- **B2 (MUST)** `prepare_candidate(graph, edits, *, cycle_policy) -> tuple[Graph | None, list[Diagnostic]]` applies to
  a copy in this order: `delete_edges` (every relation between the endpoints), `delete_nodes` (and their incident
  edges), `add_nodes`, `add_edges`. Then runs `Graph.validate()`. Deleting `Start` or `End` is `reserved_node`.
  Adding a node whose id exists is `duplicate_node`; adding an edge whose triple exists after the deletes is
  `duplicate_edge`. Non-empty diagnostics ⇒ the candidate is invalid and validation is skipped (Alg. 1 lines 11–13).
- **B3 (MUST)** `cycle_policy` ∈ `{"allow", "repair"}`. `repair`: before validation, detect cycles (DFS from every node
  in sorted id order, edges in sort order) and remove the back edges that close them; the removed edges are reported
  as `Diagnostic("cycle_repaired", …)` **warnings** in the candidate's `meta`, not failures. `allow`: skip detection.
  Default `"repair"` (the refiner prompt's rule 7 and App. E.3 describe acyclic graphs with the environment loop
  supplying repetition). Record the default as a decision.
- **B4 (MUST)** Tool-catalog membership of `ACTION` nodes is **not** enforced by `prepare_candidate` (App. B.6). It is
  a refiner-prompt rule. A host MAY pass `available_tools` to `evolve`; when it does, the harness emits a **warning**
  (not a failure) for each `ACTION` node outside the list, and records it on the candidate.
- **B5 (MUST)** `unified_diff(before: Graph, after: Graph) -> str` over the human-readable serialization, for the
  CLI and iteration reports.
- **B6 (SHOULD)** `EditSet.inverse(graph)` is not required. Attribute revision is delete + add as in the paper; do not
  add an `update_edges` operation to the model-facing schema.

### C. Online guidance runtime (§3.2), storage-free

- **C1 (MUST)** `Step(action: str, args: Any = None, observation: str = "")` is one trajectory element; a host's
  status events (`Month_Start`) are steps whose `action` is the status id. `Trajectory = Sequence[Step]`.
- **C2 (MUST)** `Localizer` protocol: `locate(graph, trajectory) -> str | None`. Default `ExactActionLocalizer`: the
  last step's `action` exactly equals a node id; empty trajectory ⇒ `"Start"` (a₀ = Start). A host with its own
  tool-call format supplies its own localizer or a `normalize: Callable[[str], str]`.
- **C3 (MUST)** `GuidanceConfig(hops=2, window=3, mode="generative_subgraph", include_relations=True,
  max_tokens=1024, task_description="…")`. `mode` ∈ `{"generative_subgraph", "generative_full", "raw_subgraph",
  "raw_full", "none"}`. `generative_*` calls the guidance model with the App. B.5 prompt; `raw_*` returns the
  serialized graph context with no model call (Table 3's "raw injection"); `none` returns empty guidance (the
  no-graph baseline, for A/B tests). `raw_subgraph` is not in the paper's ablation; say so in its docstring.
- **C4 (MUST)** `Guide(graph, model: ChatModel | None, config, localizer=None)` is frozen for its lifetime.
  `await guide.guidance(query: str, trajectory) -> Guidance(text, node_id, subgraph_digest, used_full_graph,
  usage)`. When `Match` fails and mode is `*_subgraph`, fall back to the full graph and set `used_full_graph=True`
  (eq. 2). The last `window` steps are rendered as `{recent_context}`.
- **C5 (MUST)** `serialize_context(graph, active: str | None, hops) -> str` reproduces the App. B.5 layout: active node
  header with type and description; `Immediate Transition Options (Hop 1)`; `Subsequent Horizon (Hop 2)` …; per edge
  `Condition`, `Guidance`, `Pitfalls to Avoid`; `null` condition rendered as `unconditional`. **Decision:** print the
  relation label on each transition line (`[A] → [B] (LEADS_TO; Condition: …)`) because it carries information the
  refiner wrote; `include_relations=False` reproduces the paper's serializer exactly. Record as a departure.
- **C6 (MUST)** The runtime imports nothing from `stores`. A host that keeps its graph in its own tables constructs
  `Graph.from_document(...)` and a `Guide`; no store is needed to serve.
- **C7 (SHOULD)** `Guide` records what it did so the host can put it in the trace: `guide.visited -> list[str]` (node
  ids localized, in order) and `guide.guidance_log -> list[Guidance]`. A host copies these into
  `Trace.nodes_visited` / `Trace.guidance` (see D3). This is the analogue of skillwiki's `skills_in_play`.
- **C8 (SHOULD)** `Guide.guidance_sync(...)` wrapper for synchronous hosts, as skillwiki ships `evolve_sync`.
- **C9 (MAY)** Per-episode cap `max_guidance_calls` after which the runtime returns `raw_subgraph` output instead of
  calling the model, so a runaway episode cannot spend unboundedly. Off by default.

### D. Traces (Alg. 1 line 6)

- **D1 (MUST)** `TaskOutcome(task_id, score: float in [0,1], passed: bool, prediction=None, truth=None, meta={})` and
  `Trace(id, outcome, text, steps: list[Step] = [], media: list[ImagePart] = [], meta={}, graph_ref: str | None =
  None, nodes_visited: list[str] = [], guidance: list[str] = [])`. Same field names as skillwiki where the concept is
  the same (`id`, `outcome`, `text`, `media`, `meta`) so a host's trace builder can serve both packages.
- **D2 (MUST)** `TraceSource` protocol: `collect(graph: Graph, *, graph_ref: str | None, iteration: int) ->
  list[Trace]` (positional graph, keyword-only rest, the shape of skillwiki's `collect(skill_set, *, iteration)`). The host runs the rollout (with a `Guide` over `graph`) or returns production episodes that later
  received a score. `StaticTraceSource(traces)` and `StridedTraceSource(traces, stride)` (sequential strides
  S over a fixed list, wrapping) are provided.
- **D3 (MUST)** `Trace.graph_ref` names the graph revision the episode ran under. `EvolveConfig.stale_trace_policy` ∈
  `{"warn", "drop", "allow"}`, default `"warn"`: traces whose `graph_ref` is set and differs from the current head
  are counted and reported (`warn`), excluded (`drop`), or used silently (`allow`). Production traces arrive late;
  the refiner must know which graph produced the behaviour it is diagnosing.
- **D4 (MUST)** Refiner context `C_k` (Alg. 1 line 7): render each trace as a block
  `=== Trace <id> | task <task_id> | score <s> | <PASSED|FAILED> | nodes: a → b → c ===` followed by `text`, then
  concatenate and apply `tail(text, cap)` which keeps the **end**. `EvolveConfig.refiner_context_cap: int = 120_000`
  characters by default; `EvolveConfig.token_counter: Callable[[str], int] | None` switches the cap to tokens when a
  host supplies a counter. **Decision:** concatenate passing traces first and failing traces last so the tail cut
  keeps the failures the refiner is asked to contrast (§3.3 Step 2). Record both as departures/decisions.
- **D5 (SHOULD)** `EvolveConfig.failing_sample / passing_sample: int | None = None`. `None` = the paper (all traces
  in the batch). When set, stratified sampling with seeded rotation across iterations, as skillwiki does, for hosts
  whose batches are large. `summarize_node_usage(traces) -> table` of per-node in-play counts on failing vs passing
  traces, appended to `{attempts_block}` when any trace has `nodes_visited`.
- **D6 (MUST)** `evolve(redact=Redactor)` applies `str -> str` to every trace `text` and `Step.observation` before
  the refiner, the trace store or rejection memory see it. `regex_redactor({name: pattern})` and `EMAIL` as in
  skillwiki.
- **D7 (MAY)** The refiner request MAY attach up to `EvolveConfig.refiner_images: int = 0` images from `Trace.media`
  (most recent failing traces first). Default 0: the paper is text-only.

### E. The refiner role (Alg. 1 line 9; App. B.5)

- **E1 (MUST)** `Refiner(model, config)` builds the App. B.5 prompt **in the paper's block order**: the system message
  is the first two sentences only ("You are an expert cognitive architect … encodes structured procedural guidance.");
  the user message is everything that follows, verbatim and in order (task context, mode, tools, trajectories, graph,
  rejected candidates, guidelines, rules 1–7, output format, "ONLY the raw JSON block"). Do not move the rules into
  the system message. Placeholders: `mode` ∈ the four App. D.2 modes, `available_tools_list` (host-supplied; when
  absent, "not specified: preserve existing ACTION node ids and add ACTION nodes only for actions that appear in the
  trajectories"), `attempts_block` = `C_k`, `current_graph_json` = `graph.to_refiner_json()`, `rejected_block` =
  `R_k` (see F). **One added line, directly after the "Available Tool Actions" line:**
  `Allowed node types (the "type" of every node must be one of these): {node_types_list}` filled from
  `Graph.node_types` (A1), and the same for relations when the host has changed the default vocabulary:
  `Allowed relations: {relations_list}` (omitted when the four paper relations are in use, since rule 7 and the
  examples already imply them). These lines are the only text not in the paper; they exist because A6 fails a
  candidate on an unknown type or relation and the paper's prompt never states the vocabulary. **Departure, record
  it.** One model call, text protocol (one system + one user message, plain text back), `max_tokens=8192`.
- **E2 (MUST)** Output parsing: `EditSet.parse`. On `EditError` or on `prepare_candidate` diagnostics, the harness
  MAY retry once (`EvolveConfig.role_retries: int = 1`) feeding the diagnostics back verbatim as a second user turn
  appended to the same folded transcript. A retry still counts as one refiner stage in checkpoints and budget
  (two model calls). The retry is a departure (the paper records the structural failure and moves on); with
  `role_retries=0` the paper's behaviour is exact.
- **E3 (MUST)** An empty edit set is outcome `no_action`: no candidate, no validation, one rejection-memory entry of
  kind `no_action` (so the refiner sees it next round), and it counts toward `max_rejected_streak` when that is set.
- **E4 (MUST)** Mode selection: `EvolveConfig.mode` default `"auto"` = `scratch_incremental` when the current graph
  `is_skeleton` (A3), else `static_incremental`. The one-time modes are only reachable through
  `refine_once(...)` (see G6), never through `evolve`.
- **E5 (MUST)** The refiner and the guidance model never receive tool definitions. Native tool calling is not needed
  anywhere in this package; the `ChatModel` protocol still carries `supports_tools`/`tools` fields for shape
  compatibility with skillwiki adapters (§6.3) and ignores them.

### F. Rejection memory (Alg. 1 lines 12, 19; §3.3 Step 4)

- **F1 (MUST)** `RejectionMemory` is a persistent, append-only document: `entries: list[RejectionEntry]` with
  `RejectionEntry(iteration, kind ∈ {"accepted", "rejected", "structural_failure", "no_action",
  "duplicate_candidate"}, edits: EditSet, candidate: Graph | None, candidate_digest: str | None, base_digest: str,
  trace_ids: list[str], trace_scores: dict[str, float], validation: Evaluation | None, diagnostics:
  list[Diagnostic], created_at)`. Accepted rounds are recorded too (`kind="accepted"`, with the validation) so the
  refiner and the operator see the full search trace (Table 11 is exactly this). **Departure, record it**: the
  paper's `H_rejected` holds only gate rejections and structural failures. The same five values are the
  `IterationReport.outcome` enum (G8); there is no other state.
- **F2 (MUST)** `RejectionMemory.render_for_refiner(full_entries=5) -> str` (= `SerializeRejections`, `R_k`): the
  most recent `full_entries` non-accepted entries in full (edit JSON, score or diagnostics, base digest, iteration),
  older ones as one line each (`iteration, kind, candidate digest prefix, score/diagnostic codes, edit counts`).
  Accepted entries appear as one line each with their score so the refiner knows what worked. Cap the whole block at
  `EvolveConfig.rejections_char_cap: int = 30_000`, trimming oldest one-liners first. **Decision, record it**: App.
  B.6 says `SerializeRejections` "supplies prior candidate graphs"; this package renders the edit set (which, with the
  base digest, determines the candidate) rather than the whole graph, to keep `R_k` small.
- **F3 (MUST)** Duplicate refusal: before validation the harness computes the candidate digest; if it equals the
  current head's digest or any `rejected`/`structural_failure` entry's `candidate_digest`, record
  `kind="duplicate_candidate"` (pointing at the earlier iteration) and skip validation. Validation is the expensive
  step (Alg. 1 line 15); paying it twice for one graph buys nothing. Departure, record it.
- **F4 (MUST)** Rejection memory is never rolled back or truncated by the loop. A `RejectionMemory.render_markdown()`
  gives the operator view (`rejections.md` in exports).
- **F5 (SHOULD)** `RejectionMemory.to_document()` stores candidate graphs by value (they are small: Table 7). A host
  with very large graphs MAY store `candidate=None` and keep `candidate_digest`; the render then shows the edits only.

### G. The loop (Alg. 1) and host protocols

- **G1 (MUST)** Entry point, same shape as skillwiki's:

  ```python
  async def evolve(*, config: EvolveConfig, model: ChatModel, graph_store: GraphStore,
                   rejection_store: RejectionStore, trace_source: TraceSource, evaluator: Evaluator,
                   gate: Gate | None = None, trace_store: TraceStore | None = None,
                   checkpoint_store: CheckpointStore | None = None, hooks: Hooks | None = None,
                   initial_graph: Graph | dict | Path | None = None, baseline: Evaluation | None = None,
                   available_tools: Sequence[str] | None = None, redact: Redactor | None = None) -> RunReport
  def evolve_sync(**kwargs) -> RunReport
  ```

  Order per iteration exactly Alg. 1 lines 4–20, with the stages named `baseline`, `rollout:k`, `refiner:k`,
  `validation:k` wrapped in `Hooks.stage`. `initial_graph` seeds an empty chain as revision 0 (`meta.origin =
  "onboarded"`); an empty chain with no `initial_graph` seeds `Graph.skeleton()` (`meta.origin = "skeleton"`).
  `baseline` lets a host skip line 1 when it already scored the head.
- **G1a (MUST)** Initialization from a human-made graph, or from nothing. Both the expert-prior modes (App. D.2
  Modes 2–3, `G_expert`) and the scratch modes (Modes 4–5, `G_skeleton`) must be reachable without writing Python:
  - `Graph.from_human(value: dict | Path) -> tuple[Graph, list[Diagnostic]]` loads a hand-written graph in the refiner
    JSON shape (`{"nodes": [...], "edges": [...]}`, §3.1; `"schema"` optional on input; a `Path` is read as a JSON
    file). Missing `Start` / `End` nodes are **added** with a `Diagnostic("skeleton_completed", …)` warning, so a human
    can write only the middle of the graph; missing attribute fields default to `null`. Everything else goes through
    `Graph.validate()` and a failing human graph is refused with the diagnostics, never silently repaired (the paper
    repairs refiner output, not human input). `Graph.from_document` (A4) stays the strict inverse of `to_document`
    and never completes or repairs; only the human-input path does.
  - **"Empty" means the skeleton.** A graph with no nodes cannot pass A6 (the paper's own checks need `Start` and a
    terminal), so an omitted, `None` or `{"nodes": [], "edges": []}` initial graph seeds `Graph.skeleton()`
    (`meta.origin = "skeleton"`) and E4 `auto` selects `scratch_incremental`. A non-skeleton human graph seeds with
    `meta.origin = "onboarded"` and `auto` selects `static_incremental`.
  - `GraphStore.seed(graph, *, meta) -> str` is part of the protocol (see G3) so a host store can be initialized the
    same way; it MUST refuse (`HeadMoved`) when the chain already has a graph. Re-seeding is a `transfer` (H7) or a
    new workspace, never an overwrite.
  - `evolve(initial_graph=...)` accepts a `Graph`, a `dict` or a `Path` (`dict` / `Path` go through `from_human`);
    CLI `init` (H8) does the same from the shell. When the workspace **already has a graph**, a supplied
    `initial_graph` is an error (`ValueError` naming the head ref), not a warning: silently continuing from a
    different graph than the caller asked for is the failure mode this exists to prevent. Use `transfer` or a new
    workspace instead.
- **G2 (MUST)** `EvolveConfig` defaults follow the paper where it states a number:

  | Field | Default | Source |
  | --- | --- | --- |
  | `task_description` | `"tasks in this workspace"` | prompt placeholder |
  | `max_rounds` (K) | `10` | §5.4, App. E |
  | `max_rejected_streak` | `None` (run all K rounds) | Alg. 1 has no early stop; skillwiki's 3 is available |
  | `cycle_policy` | `"repair"` | B3 decision |
  | `mode` | `"auto"` | E4 |
  | `refiner_context_cap` / `token_counter` | `120_000` chars / `None` | D4 decision (L_max unspecified) |
  | `rejections_full_entries` / `rejections_char_cap` | `5` / `30_000` | F2 decision |
  | `refiner_max_tokens` | `8192` | App. D.3 (guidance output length lives in `GuidanceConfig.max_tokens`, C3, because `evolve` never calls the guidance model) |
  | `role_retries` | `1` | E2 departure |
  | `failing_sample` / `passing_sample` | `None` | D5, paper uses the whole batch |
  | `stale_trace_policy` | `"warn"` | D3 |
  | `success_threshold` | `0.5` | decision: partitions non-binary scores into high/low for the `attempts_block` headers (paper: "high-scoring vs low-scoring") |
  | `workspace` | `"default"` | store chain name; the CLI `--workspace` sets it |
  | `seed` | `17` | sampling |
  | `budget` | `Budget()` (unlimited) | H1 |

- **G3 (MUST)** `GraphStore` protocol (the live artefact; ≈ skillwiki's `SkillStore`):

  ```python
  class GraphStore(Protocol):
      async def load(self) -> tuple[str | None, Graph]: ...          # empty chain: (None, Graph.skeleton()); ref None tells the harness to seed (G1a)
      async def seed(self, graph: Graph, *, meta: dict[str, Any]) -> str: ...   # revision 0; raises HeadMoved if a graph already exists (G1a)
      async def propose(self, current_ref, current: Graph, edits: EditSet, candidate: Graph, *, iteration: int,
                        rejections_ref: str | None) -> Candidate: ...  # idempotent per (iteration, edits)
      async def accept(self, candidate: Candidate, *, expected_ref: str | None) -> str: ...
  ```

  `Candidate(ref, graph, edits, meta)`. **The host's candidate is canonical**: after `propose`, the harness
  validates and continues with `candidate.graph`; if its digest differs from the module's it emits one warning
  naming the differing node/edge counts. `propose` MUST be idempotent for a given `(iteration, edits)` because resume
  calls it again.
- **G4 (MUST)** `RejectionStore` protocol (the persistent memory; ≈ `WikiStore`): `load() -> tuple[str | None,
  RejectionMemory]`, `save(memory, *, expected_ref, iteration) -> str`. Saved **once per iteration, after the
  outcome is known** (accepted, rejected, structural failure, no action or duplicate), so every entry is final when
  written. The refiner's output is protected between the refiner stage and that save by the G7 checkpoint, not by a
  provisional entry. A test (§8 item 10) covers the interruption window.
- **G5 (MUST)** `Evaluator.evaluate(ref, graph: Graph, *, iteration, purpose ∈ {"baseline", "candidate"}) ->
  Evaluation(ref, score, aggregate, per_task)`; `Gate.decide(best: Evaluation, candidate: Evaluation) -> Decision(
  accepted, feedback, stop)`. Default gate `TieAcceptingGate(perfect=None)` (eq. 5: `>=`). As in skillwiki, the
  perfect-score early stop lives on the **gate**, not on `EvolveConfig`: every gate takes `perfect: float | None`;
  the harness probes it after each acceptance and stops with `stopped_reason="perfect"` when `best.score >= perfect`.
  The default gate's `perfect=None` keeps Alg. 1's run-all-K behaviour; `StrictImprovementGate` and `PairedGate`
  default to `perfect=1.0` as skillwiki's do. `StrictImprovementGate` (`>`) and
  `PairedGate(min_win_probability=0.9, min_tasks=20, resamples=2000, seed=17)` (paired bootstrap over `per_task`,
  accepts iff mean delta ≥ 0 and P(delta ≥ 0) ≥ threshold — note `≥`, matching the paper's tie rule) are provided.
  `PairedGate` raises when `per_task` is missing on either side.
- **G6 (MUST)** `refine_once(*, config, model, graph, traces, available_tools=None) -> tuple[Graph | None, EditSet,
  list[Diagnostic]]`: the one-time modes (App. D.2 Modes 2 and 4). No store, no gate, no rejection memory. Used for
  bootstrapping an expert prior from a trace dump and in tests.
- **G7 (MUST)** Resume via `CheckpointStore` (`load(iteration) -> dict | None`, `save(iteration, payload)`): after the
  refiner stage save `{edits, candidate_digest, rejections_ref, graph_ref}`; after validation add `{evaluation}`. A
  restart on the same stores finishes the iteration (re-`propose` through the host, reuse the recorded evaluation)
  instead of re-running rollout and refiner. `NullCheckpointStore` default.
- **G8 (MUST)** `RunReport(iterations: list[IterationReport], graph_ref, graph, rejections_ref, best: Evaluation,
  stopped_reason ∈ {"max_rounds", "rejected_streak", "perfect", "budget_exhausted", "gate_stop"}, budget: dict,
  warnings)`, `to_dict()`. `IterationReport(iteration, trace_ids, stale_trace_ids, edits, candidate_digest, outcome ∈
  {"accepted", "rejected", "structural_failure", "no_action", "duplicate_candidate"}, evaluation, diagnostics,
  warnings)`.
- **G9 (MUST)** `Hooks` with `stage(name)` async context manager, `on_model_call(role, request, response)`,
  `on_iteration(report)`, `on_warning(message)`. `HookedModel` wraps the host model, meters calls and
  `usage["output_tokens"]`, and is what the refiner uses. The `Guide` accepts any `ChatModel`, so a host passes the
  same `HookedModel` to have guidance calls metered under the same budget when it runs the rollout in-process.
- **G10 (SHOULD)** Roles are identified in `ModelRequest.role` as `"refiner"` and `"guidance"` so a host can price and
  admit them separately.

### H. Budget, stores, transfer, CLI, adapters (skillwiki 0.2 production items carried over)

- **H1 (MUST)** `Budget(max_model_calls, max_output_tokens, max_evaluations, max_seconds)`; checked before every model
  call and every evaluation; exhaustion raises `BudgetExceeded`, the harness stops with
  `stopped_reason="budget_exhausted"` and everything already saved stays saved.
- **H2 (MUST)** `RevisionStore` protocol and `Revision` record **byte-for-byte compatible** with skillwiki's (§6.1,
  §6.2) so one host adapter can back both packages. Adapters: `MemoryRevisionStore`, `FileRevisionStore`,
  `PostgresRevisionStore` (table `proceduralgraph_revisions`, same DDL shape, §6.4), `BlobRevisionStore` over
  `BlobStore` with `S3BlobStore` / `GCSBlobStore` / `MemoryBlobStore` using write preconditions. All pass the
  contract test in §6.5 verbatim.
- **H3 (MUST)** Default layer stores over any `RevisionStore`: `RevisionGraphStore` (kind `graph`; only accepted
  graphs join the chain; `seed(graph, *, meta)` per G3), `RevisionRejectionStore` (kind `rejections`), `RevisionTraceStore` (kind
  `traces`, one revision per iteration, `get(trace_id)`), `RevisionCheckpointStore` (kind `checkpoints`). These kind
  names do not collide with skillwiki's (`wiki`, `skills`, `raw`, `checkpoint`), so a host MAY point both packages at
  one table; the default DDL still uses a separate table.
- **H4 (MUST)** `NullTraceStore`, `NullCheckpointStore` for hosts whose raw layer lives elsewhere.
- **H4a (MUST)** Storage principle: **`GraphStore` is the abstraction; a JSON document in PostgreSQL is its first
  implementation.** In 0.1 the only shipped implementation is `RevisionGraphStore` over a `RevisionStore`: the whole
  graph (`Graph.to_document()`) is one JSONB value per accepted revision in `proceduralgraph_revisions`, with no
  per-node or per-edge table and no graph database. Further implementations (a normalised schema, a graph database, a
  host's own tables) are added later as new classes under `stores/` behind the same protocol; the harness only ever
  calls `load`, `propose`, `accept` (and `seed`) and MUST NOT depend on how any implementation stores the graph.
  Nothing in `src/` outside `stores/` may import or assume a JSON store. `examples/` and `docs/host-integration.md`
  show the Postgres wiring (`PostgresRevisionStore.from_url` + `RevisionGraphStore`) as the primary path and the file
  store as the no-infrastructure fallback.
- **H5 (MUST)** Graph revision `meta` carries `{iteration, validation_score, edits, origin}` so `history` and `diff`
  can explain each accepted step without loading rejection memory.
- **H6 (MUST)** `export_workspace(dir, graph, rejections)` writes `graph.json` (refiner shape), `graph.md` (the full
  serialization, every node as active once, plus a node/edge count header), `graph.mmd` (Mermaid `flowchart LR` with
  relation labels), `rejections.md`. This is the rendering; the storage is the chain.
- **H7 (MUST)** `seed_workspace(revisions, *, source, target)`: copies the source's head graph into an **empty**
  target as revision 0 with `meta.origin = "transfer:<source>"`; refuses (returns `None`) when the target already has
  a graph; never copies rejection memory (it records what the source validated).
- **H8 (MUST)** CLI `proceduralgraph --store file:DIR | postgres:URL --workspace WS <cmd>`: `init [--graph FILE]` (G1a: seed the
  workspace from a hand-written graph JSON, or the skeleton when `--graph` is omitted; refuses a non-empty workspace), `show` (node/edge counts,
  head digest, last validation score, last 10 rejection lines), `export DIR`, `history [--kind]`, `diff [--from
  --to]` (B5), `rejections [--full N]`, `transfer --to WS`, `guide --query Q [--step ACTION]...` (runs the serializer
  in `raw_subgraph` mode with no model, so an operator can see what the agent would be shown). The `postgres:` store
  fails with a plain message when the extra is missing.
- **H9 (MUST)** `adapters/anthropic.py` and `adapters/openai.py`: `ChatModel` over one system + one user message,
  returning text and `usage`. Default model ids per the current Claude/OpenAI naming used in skillwiki's adapters; do
  not hard-code a retired id. `ScriptedChatModel` for tests (list of responses or callable).
- **H10 (SHOULD)** `documents.SCHEMA_VERSION = 1` written into every graph, rejection and trace document; loaders
  refuse unknown higher versions with a plain message.

### I. Documentation and examples

- **I1 (MUST)** `README.md` in skillwiki's structure: one paragraph what it is and that it is independent; the
  "what the paper does / what this package keeps" table; install; a sixty-second example that shows **both** halves
  (a `Guide` inside a toy solver loop, then `evolve`); "Integrating with your own system" listing the protocols
  (`TraceSource`, `Evaluator`, `Gate`, `ChatModel`, `GraphStore`, `RejectionStore`, `RevisionStore`); fidelity notes;
  development commands; authorship footer.
- **I2 (MUST)** `docs/paper-differences.md` in skillwiki's exact format: §1 *Kept exactly* table (mechanism / paper /
  package), §2 *Departures* each as **Paper / package / Why** naming the production problem it solves, §3
  *Deliberately unchanged*, plus a §0 *Decisions where the paper is silent* covering every item in §3.4 of this file.
  Every requirement above marked "departure" or "decision" MUST appear here.
- **I3 (MUST)** `CHANGELOG.md` with `0.1.0 (date)`.
- **I4 (MUST)** `examples/filesystem_toy.py`: no API key, no network, `FileRevisionStore` under
  `./toy-workspace`. A toy two-hop QA environment: tools `search(q)`, `read(doc)`, `answer(text)`; the scripted
  solver answers straight after the first search unless the guidance text contains a cue (e.g. "read" or "second
  hop"); questions needing two hops fail without the graph. The scripted refiner, over three or more rounds, MUST
  produce: one accepted candidate that adds `search → read → search → answer`, one structural failure (an edge to a
  node that does not exist), one rejected candidate (validation drops), and one duplicate-candidate refusal. The
  example prints the `RunReport`, then `export_workspace` so the operator can open `graph.md`, `graph.mmd`,
  `rejections.md`. `examples/anthropic_toy.py` does the same with a real model.
- **I5 (SHOULD)** `docs/host-integration.md`: how a production host wires the package, host-agnostic. Cover: calling
  `Guide.guidance()` per step and copying `visited`/`guidance` into the trace; returning labelled production
  episodes from `TraceSource.collect` with `graph_ref` set; implementing `GraphStore.propose` against the host's own
  table idempotently; wrapping paid stages in leases via `Hooks.stage`; a text-only metered transport (no tool
  definitions) behind `ChatModel`; per-task scores so `PairedGate` can be used; pointing the host's existing
  `RevisionStore` implementation at both packages.

## 6. Storage compatibility: what must match skillwiki byte-for-byte

The first production host implemented skillwiki's store protocols directly against its own tables and also had
skillwiki's `PostgresRevisionStore` available. The same must be true here with **no adapter changes** on the host
side. These signatures are copied from skillwiki 0.2.0 and are the contract.

### 6.1 Canonical JSON and digests

```python
def canonical_json(value) -> str   # json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
def digest(value) -> str           # sha256(canonical_json(value).encode("utf-8")).hexdigest()
```

`_plain` converts dataclasses via `asdict`, dict keys to `str`, sets/frozensets to sorted lists, tuples to lists,
`datetime` to ISO strings.

### 6.2 `Revision` and `HeadMoved`

```python
class HeadMoved(Exception):
    def __init__(self, workspace: str, kind: str, expected: str | None, actual: str | None): ...

@dataclass(frozen=True)
class Revision:
    workspace: str
    kind: str
    seq: int                    # 0 for the first revision of a chain
    digest: str                 # sha256 of `document` ONLY (identical content twice => identical digest)
    parent_digest: str | None
    document: dict[str, Any]
    created_at: str             # ISO-8601 UTC
    meta: dict[str, Any]
    @classmethod
    def build(cls, workspace, kind, document, *, parent: Revision | None, meta=None) -> Revision: ...
    def to_document(self) -> dict: ...      # all eight fields
    @classmethod
    def from_document(cls, value) -> Revision: ...
    def verify(self) -> None: ...           # raises ValueError when digest(document) != digest
```

### 6.3 `RevisionStore` and `ChatModel`

```python
class RevisionStore(Protocol):
    """An append-only chain per (workspace, kind). ``append`` must refuse to fork: when ``expected_head`` is not the
    digest of the current head (or ``None`` for an empty chain) it raises HeadMoved."""
    async def head(self, workspace: str, kind: str) -> Revision | None: ...
    async def append(self, workspace: str, kind: str, document: dict[str, Any], *,
                     expected_head: str | None, meta: dict[str, Any] | None = None) -> Revision: ...
    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None: ...
    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]: ...   # newest first

@dataclass(frozen=True) class TextPart: text: str
@dataclass(frozen=True) class ImagePart: media_type: str; data_base64: str; ref: str = ""
@dataclass(frozen=True) class ToolSpec: name: str; description: str; parameters: dict
@dataclass(frozen=True) class ToolInvocation: name: str; args: dict
@dataclass(frozen=True)
class ModelRequest:
    role: str; system: str; parts: Sequence[TextPart | ImagePart]; max_tokens: int = 4096; tools: Sequence[ToolSpec] = ()
    @property text -> str        # joined TextParts
    @property images -> list[ImagePart]
@dataclass(frozen=True)
class ModelResponse: text: str; usage: dict = {}; tool_calls: Sequence[ToolInvocation] = ()
class ChatModel(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
```

A host's existing `ChatModel` adapter reads `request.role`, `request.system`, `request.parts`, `request.max_tokens`
and returns an object with `.text` and `.usage`. Because these names match, the same adapter class serves both
packages by duck typing; neither package imports the other.

### 6.4 PostgreSQL DDL (same shape, different table)

```sql
CREATE TABLE IF NOT EXISTS proceduralgraph_revisions (
    workspace     TEXT        NOT NULL,
    kind          TEXT        NOT NULL,
    seq           INTEGER     NOT NULL,
    digest        CHAR(64)    NOT NULL,
    parent_digest CHAR(64),
    document      JSONB       NOT NULL,
    meta          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, kind, seq)
);
CREATE INDEX IF NOT EXISTS ix_proceduralgraph_revisions_digest ON proceduralgraph_revisions (workspace, kind, digest);
```

The primary key is what makes the chain fork-free: two writers that both read head `seq = n` both try to insert
`n + 1` and exactly one succeeds; the loser raises `HeadMoved`. Rows are never updated or deleted. `table` MUST be a
constructor argument so a host can point at its own table (an **addition** over skillwiki 0.2.0, whose
`PostgresRevisionStore.__init__(self, engine)` takes only the engine; keep `engine` first so the host's existing
construction code still works). File store: `<root>/<workspace>/<kind>/HEAD` holding
`<seq:06d>-<digest>` and one JSON file per revision, written atomically (temp file + `os.replace`). Blob store, as
skillwiki's `blob.py` does it: `BlobStore` protocol with `get(key) -> (bytes, tag) | None`, `put_if_absent(key,
data)`, `put_if_match(key, data, tag=)`; `append` writes the revision object create-only, then advances
`<workspace>/<kind>/HEAD` (`{"digest", "seq"}`) under the tag it read; a failed precondition raises `HeadMoved`. S3
implements the preconditions with `IfNoneMatch="*"` and the ETag; GCS with `if_generation_match=0` / `=<generation>`.
`MemoryBlobStore` is the reference implementation used by the contract test's race case.

### 6.5 Store contract test (copy into `tests/test_stores_contract.py`, run against every adapter)

```python
async def contract(store):
    assert await store.head("ws", "graph") is None
    first = await store.append("ws", "graph", {"n": 1, "text": "é"}, expected_head=None, meta={"iteration": 1})
    assert first.seq == 0 and first.parent_digest is None and first.meta == {"iteration": 1}
    second = await store.append("ws", "graph", {"n": 2}, expected_head=first.digest)
    assert second.seq == 1 and second.parent_digest == first.digest
    head = await store.head("ws", "graph")
    assert head.digest == second.digest and head.document == {"n": 2}
    head.verify()
    with pytest.raises(HeadMoved):
        await store.append("ws", "graph", {"n": 3}, expected_head=first.digest)
    with pytest.raises(HeadMoved):
        await store.append("ws", "graph", {"n": 3}, expected_head=None)
    assert [r.seq for r in await store.list("ws", "graph")] == [1, 0]
    assert [r.seq for r in await store.list("ws", "graph", limit=1)] == [1]
    assert (await store.get("ws", "graph", first.digest)).document == {"n": 1, "text": "é"}
    assert await store.get("ws", "graph", "0" * 64) is None
    assert await store.head("ws", "rejections") is None and await store.head("other", "graph") is None
    other = await store.append("other", "graph", {"n": 1}, expected_head=None)
    assert other.seq == 0
```

Postgres and S3 tests run under testcontainers / moto and are skipped when Docker is absent, as in skillwiki.

## 7. Package layout (expected; deviate only with a reason)

```
src/proceduralgraph/
  __init__.py          public API re-exports (mirror skillwiki's __init__)
  graph.py             Node, Edge, Graph, Diagnostic, neighborhood, validate, digest, refiner json     [A]
  edits.py             EditSet, EditError, prepare_candidate, cycle repair, unified_diff                [B]
  serialize.py         serialize_context (App. B.5 layout), markdown and mermaid renderers              [C5, H6]
  guidance.py          Step, Localizer, ExactActionLocalizer, GuidanceConfig, Guide, Guidance          [C]
  traces.py            TaskOutcome, Trace, TraceSource, Static/StridedTraceSource, sampling, node usage [D]
  rejections.py        RejectionEntry, RejectionMemory, render_for_refiner, render_markdown            [F]
  roles/refiner.py     Refiner (prompt + parse + retry)                                                [E]
  harness.py           evolve, evolve_sync, refine_once, RunReport, IterationReport, resume            [G]
  gates.py             Evaluation, Decision, Evaluator, Gate, TieAcceptingGate, StrictImprovementGate, PairedGate
  config.py            Budget, EvolveConfig
  hooks.py             Hooks, HookedModel, BudgetMeter, BudgetExceeded
  model.py             TextPart, ImagePart, ToolSpec, ToolInvocation, ModelRequest, ModelResponse, ChatModel, ScriptedChatModel
  documents.py         SCHEMA_VERSION, canonical_json, digest, Revision, HeadMoved
  redaction.py         Redactor, regex_redactor, EMAIL, redact_trace
  stores/{base,memory,file,postgres,blob,s3,gcs,revision,transfer}.py, stores/__init__.py
  adapters/{anthropic,openai}.py
  cli.py
tests/  test_graph.py test_edits_and_structure.py test_guidance.py test_serialize.py test_refiner.py
        test_rejections.py test_harness_end_to_end.py test_gates.py test_budget_and_resume.py
        test_stores_contract.py test_traces_redaction_cli.py test_transfer.py test_adapters.py
examples/filesystem_toy.py examples/anthropic_toy.py
docs/paper-differences.md docs/host-integration.md docs/paper/
```

## 8. Acceptance checklist

The work is done when every line below is true and you have reported each one with the command and its output.

1. `uv run ruff check src tests examples` clean; `uv run pytest -q` all passing; `uv build -o dist` produces sdist
   and wheel.
2. `uv run python examples/filesystem_toy.py ./toy-workspace` runs with no network and prints a `RunReport` whose
   iterations include, in some order, outcomes `accepted`, `structural_failure`, `rejected`, `duplicate_candidate`;
   the exported `graph.mmd` shows the `search → read → search → answer` path; `rejections.md` lists every non-accepted
   round with its edits.
3. Every adapter in `stores/` passes the §6.5 contract test unchanged (Postgres/S3 skipped without Docker, and the
   skip is reported, not hidden).
4. A test asserts eq. 5 tie acceptance: a candidate scoring exactly the head's score is accepted by the default gate
   and rejected by `StrictImprovementGate`.
5. A test asserts Alg. 1 lines 11–13: a structurally invalid candidate never reaches `Evaluator.evaluate`, is recorded
   with diagnostics, and the head and its cached score are unchanged.
6. A test asserts `delete_edges` removes every relation between the endpoints and that `add_edges` is applied after.
7. A test asserts `Tail`: the refiner context keeps the end of the concatenation when the cap is exceeded, and that
   failing traces sit at the end.
8. A test asserts eq. 2 fallback: with a trajectory whose last action matches no node, `Guide` uses the full graph and
   reports `used_full_graph=True`; with a match, the serialized context contains exactly the nodes within `hops`.
9. A test asserts the serializer output for a two-hop example matches the App. B.5 layout (headers, `Hop 1`, `Hop 2`,
   `Guidance`, `Pitfalls to Avoid`) with `include_relations=False`.
10. A test asserts resume: interrupting after the refiner stage and restarting on the same stores calls
    `GraphStore.propose` again for the same iteration and does not call the refiner or `TraceSource.collect` again.
11. A test asserts `BudgetExceeded` before a paid stage stops the run with `budget_exhausted` and leaves the head and
    rejection memory saved.
12. A test asserts the host-canonical candidate: a `GraphStore.propose` that returns a graph differing from the
    module's is what gets evaluated, with one warning.
13. A test asserts `stale_trace_policy="drop"` excludes traces whose `graph_ref` is not the head and reports them.
13a. A test asserts G1a: `evolve` on an empty workspace with no `initial_graph` seeds the skeleton and runs
    `scratch_incremental`; with a hand-written JSON missing `End` it seeds the completed graph with a
    `skeleton_completed` warning and runs `static_incremental`; with a graph that fails `validate()` it refuses before
    any model call; `seed` on a non-empty chain raises `HeadMoved`.
14. `docs/paper-differences.md` lists every departure/decision from §5 with a Why; the README table and the
    sixty-second example run as written.
15. No file under `src/`, `docs/`, `README.md` names any production host, product or customer.
16. Nothing committed; `git status` shows the new tree for the user to review.

## 9. Open questions (do not resolve by inventing; leave them in `docs/paper-differences.md` §4 "Open")

- **Layers / shared graphs.** Skill layers (global → tenant → project) have no obvious graph analogue. Options for
  later: per-workspace overlays (a project graph whose nodes may `ref` a tenant graph's nodes), or transfer only.
  Ship transfer (H7) only.
- **Semantic localization.** The paper's `Match` is exact. A host whose actions do not map 1:1 onto node ids may want
  an LLM or embedding localizer. The `Localizer` protocol is the seam; do not ship a model-based default.
- **Guidance reuse across steps.** The paper's conclusion names it as future work (guidance tokens dominate cost).
  `raw_*` modes and C9 are the only mitigations shipped.
- **Graph-database backend.** A `GraphStore` over Neo4j / Memgraph / Postgres `AGE` is possible (H4a) but not
  shipped: the paper's graphs are tiny (Table 7), the runtime never queries the store, and revision chains give the
  immutability and fork-refusal a graph DB would need rebuilt on top. Revisit only when a host's graph outgrows a
  single JSONB value or needs cross-graph queries.
- **Interop with a skill store.** A node with `type="SKILL"` and `ref=<skill name>` would let guidance say "apply
  skill X". Allowed by A8, not exercised anywhere.

## 10. Working rules for the implementing agent

- Work in this repo only. Do not modify `~/skillwiki`. Do not commit; the user reviews and commits.
- When the paper and this document disagree, follow the paper text in `docs/paper/2609.09153v1.txt` and note the
  discrepancy in your final report.
- When you must choose where the paper is silent, choose, implement, test, and record it in
  `docs/paper-differences.md` §0. Never leave a silent default.
- Keep the core dependency-free. Anything that needs a library goes behind an extra and an import inside the function.
- Write the test before the fix for anything in §8 that fails on first run.
- Your final report to the user lists: each acceptance item with evidence, every decision made under §3.4, anything
  in §5 you did not build and why, and the exact commands to run the suite and the toy.
