# Paper source

`2609.09153v1.txt` is `pdftotext -layout` output of arXiv:2609.09153v1, *Procedural Graphs: Self-Evolving
Execution Structures for LLM Agents* (Yuxing Lu, Yicheng Chen, Shanchan Wu, Sercan Ö. Arık; Google, Georgia Tech,
Peking University; submitted 8 Sep 2026). PDF: https://arxiv.org/pdf/2609.09153

Read it from this file, not from memory. Figures came out as scattered words (Figures 1, 2 and 5); tables and
prose are intact. The parts the implementation depends on:

| Where | What |
| --- | --- |
| §3.1 | Formal graph `G = (V, R, E, Φ)`, edge attributes `condition / guidance / pitfalls` |
| §3.2, eq. (2) and (3) | Online guidance: `Match`, `N_h(u_t)`, full-graph fallback, window `w`, guidance model `Ψ` |
| §3.3, eq. (4) to (6) | Offline evolution: rollout, refiner, gate (`>=`, ties accepted), rejection memory, `Tail_Lmax` |
| §4 "Procedural Graph Configuration" | `h = 2`, `w = 3`, greedy decoding |
| §5.5, Table 3 | Full graph vs subgraph, raw injection vs generative guidance |
| App. B.1 | Strides `S = 100` (HotpotQA) / `S = 20` (MultiChallenge); validation sizes |
| App. B.4, Table 7 | Graph sizes; relation vocabulary `LEADS_TO, TRIGGERS, PROVIDES_INPUT_FOR, CONVERGES_TO` |
| App. B.5 | Solver prompt, guidance prompt, serialized graph-context example, refiner prompt and its JSON edit format |
| App. B.6, Algorithm 1 | The loop, `PrepareCandidate`, structural checks, cycle policy |
| App. D.2 | The five construction modes (`static_onetime`, `static_incremental`, `scratch_onetime`, `scratch_incremental`) |
| App. D.3 | Max generation length 2,048 (agent) / 8,192 (refiner), temperature 0 |
| App. E | Ten-round evolution trace; what accepted, rejected and structurally-failed rounds look like |
