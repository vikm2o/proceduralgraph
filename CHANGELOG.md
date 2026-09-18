# Changelog

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
