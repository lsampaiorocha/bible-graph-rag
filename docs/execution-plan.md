# Bible Graph RAG — Execution Plan

## Context

**Why this project exists.** It is a portfolio piece for interviews, aimed first at xNILIO's Senior AI Engineer role (LLMs, RAG & Knowledge Graphs). Their stack is a TypeScript monorepo (Vite + Node.js), Python for batch work, and Docker. The role asks for:
- LLM knowledge extraction
- graph and vector databases
- evaluation harnesses with golden datasets

**What it does.** You type a question about the Bible and get an answer. The graph lights up the people, places and relationships the search used, each tagged with one of three levels:
- **retrieved**: found by the search
- **sent to the LLM**
- **cited**: used in the answer

**Approved decisions** (`docs/discovery/00-decision-sheet.md`, plus the changes made on 2026-10-07)

| Area | Decision |
|---|---|
| Corpus | KJV Bible. Tune on Genesis, then index all 66 books. |
| Text source | `verseText` from Theographic's `verses.json`. Only the verse ID and text are read: no tags, no links. |
| Graph | Built entirely by our pipeline. An LLM extracts entities and relationships, then our own entity resolution (ER) step merges duplicate names and tells apart people who share a name. |
| Benchmark | **Theographic**, a free hand-made Bible graph built by one person (CC-BY-SA-4.0). It is used **only as the answer key**, never as an input. A test enforces this. |
| Search modes | Local (start from the entities in the question and expand to their neighbours) and Path (how is X connected to Y?) |
| Baselines | Plain vector RAG, and the LLM with no retrieval |
| Storage and search | Neo4j Community Edition (2026.09) in Docker Compose holds the graph and the vector index. Cypher fetches; shared TypeScript ranks. |
| Live mode | A Node.js (Hono) API, run locally only |
| Frontend | Vite + React + Tailwind + Sigma.js 3 (`@react-sigma/core`), in a pnpm TypeScript monorepo |
| Public demo | $0 GitHub Pages site: JSON export of the graph, search in the browser with transformers.js, answers only with the visitor's own API key (BYOK) |
| Outside benchmark | LightRAG on the same text |
| Deadline | None fixed |

## Extraction schema (approved)

**Entities**

| Type | Properties | How it is graded against Theographic |
|---|---|---|
| Person | name, aliases, gender, description, verses where named | Person-mention F1 over 28,240 (person, verse) tags; gender accuracy; split and merge errors |
| Place | name, aliases, kind (City/Region/Landmark/Mountain/Water/Valley/Island), description, verses | Place-mention F1 over 7,310 (place, verse) tags; kind accuracy on the 720 places that have a kind |
| Group | tribes, nations, sects | Tribe membership only (Theographic has 23 groups) |
| Event | title, participants, place, verses | Matched by verse overlap against 450 events: event recall and participant F1 |

**Relationships**
- **Graded:** FATHER_OF, MOTHER_OF, SIBLING_OF, SPOUSE_OF, MEMBER_OF, BORN_IN, DIED_IN, PARTICIPATED_IN, HAPPENED_AT.
- **Ungraded, fixed list:** RULED, SUCCEEDED, ANOINTED, FOUGHT, KILLED, BETRAYED, BLESSED, CURSED, HEALED, DISCIPLE_OF, SERVED, SPOKE_TO, and OTHER with free text. Precision is measured by hand-labelling 100 random edges, with a 95% confidence interval.
- **Every edge carries** a one-sentence description, its evidence verse IDs, and a support count (the number of passages that back it).

**Rules**
- **Mentions are tagged by name only,** matching Theographic's convention: "the LORD" is tagged as God, pronouns are never tagged.
- **Relationships are extracted per passage** (about 600 tokens, never crossing a chapter), so the LLM can resolve pronouns from the surrounding verses.

**Not in v1:**
- objects
- birth and death years
- map coordinates
- Easton's dictionary text

**Grading method:** map each of our entities to the Theographic ID whose verse set overlaps it most. That mapping is the basis for every person and place metric.

---

## Milestones

Each milestone ends with something you can see. Hours are focused work, and each task inside a milestone is at most about 4 h.

| # | Milestone | You can now… | Hours |
|---|---|---|---|
| M0 | Toolchain and Neo4j | Open Neo4j Browser at localhost:7474. `pnpm check` and `uv run pytest` pass on an empty skeleton. | 3–5 |
| M1 | Corpus and answer-key loader | Print any passage with its verse markers, and see the Genesis answer-key statistics table | 6–9 |
| M2 | Extraction v1 and model bake-off | See two models compared on 20 Genesis passages (F1 and $), and pick the model | 9–13 |
| M3 | **Vertical slice** | Type "Noah" at localhost:5173 and watch his 2-hop neighbourhood light up, with the trace JSON alongside | 12–16 |
| M4 | Entity resolution v1 and scoreboard | Run the ER scoreboard for before vs after. The two Lamechs are split; Abram = Abraham and Jacob = Israel are merged. | 14–20 |
| M5 | Benchmark harness and regression test | Open `#/eval` and see P/R/F1 per category. CI fails if a metric silently moves. | 11–15 |
| M6 | Local mode complete | Get cited answers. Clicking [C2] highlights the verses and the subgraph. | 11–15 |
| M7 | Path mode | Ask "How is Lot connected to Abraham?" and see the chain highlighted and cited | 5–8 |
| M8 | Baselines and QA evaluation | See Graph vs Vector vs Closed-book on 25 Genesis questions | 11–15 |
| M9 | Full Bible | See the whole-Bible graph, plus Genesis (tuned) vs Bible (held-out) scores | 9–14 |
| M10 | In-browser search and parity test | Stop the API, type a question in the static build, and still see the graph light up | 10–15 |
| M11 | **Ship: GitHub Pages and BYOK** | Share a public URL with 25 replayable queries. Requires your go-ahead to push. | 6–9 |
| M12 | Replay and Compare views | Step through seeds → hops → dropped → context → cited; split view of vector vs graph | 10–16 |
| M13 | LightRAG benchmark | See a LightRAG column in the QA table | 5–8 |
| M14 | README, GIF, design notes | Send a polished repo link with the results table first | 5–8 |

**Total:** 127–186 h of focused work, or about **145–220 h** including time spent re-orienting.
- The public demo ships after M11, at about 107–154 h.
- At 10 h a week that is about 13–19 weeks; at 20 h a week, about 7–10 weeks.

**LLM spend:** budget **$15–40 in total**.
- Genesis extraction: about $0.1–1.
- Full Bible: about $2–24, halved with the Batch API.
- Each eval run: under $2.

### Milestone tasks

**M0 Toolchain (3–5 h)**
- Fix the `pnpm` on PATH, which currently resolves to the Windows shim `/mnt/c/Users/leona/AppData/Roaming/npm/pnpm` (verified).
- Install Node 24 with nvm, required by vitest 5.0.3 (`node ^22.12 || ^24`, verified), plus `.nvmrc` and corepack pnpm.
- `git init`, local only. Add `.gitignore` and `.env.example`.
- `docker-compose.yml` with `neo4j:2026.09.0-community` (tag verified, pushed 2026-10-06): heap 1G, page cache 512m, a named volume, a healthcheck.
- uv project `pipeline/` with ruff, pytest and a typer CLI.
- pnpm workspace with `packages/{retrieval,api,web}`, Biome 2.5 as formatter and linter, tsc, Vitest. `pnpm check` = `biome ci && tsc -b && vitest run`.
- `.githooks/pre-commit` running ruff format --check, ruff check and biome ci, enabled with `git config core.hooksPath .githooks`.

**M1 Corpus and answer key (6–9 h)**
- `kgrag corpus fetch/build`: download `verses.json` pinned to commit `cfb1c48`, and write `data/corpus/verses.jsonl` with **only** osis_ref, verse_id, book, chapter, verse and text.
- Passage chunker: up to 600 tiktoken tokens, never crossing a chapter, no overlap. IDs look like `Gen.4.p2`. That gives about 127 Genesis passages and about 2,677 for the Bible.
- `kgrag_eval/theographic.py`: gold loader, plus a parser for the `mdText` mention spans (gold surface text for each mention).
- `kgrag-eval gold-stats --scope Gen`.
- Boundary test: the `kgrag` package never imports `kgrag_eval`, checked by an AST scan.

**M2 Extraction and bake-off (9–13 h)**
- Pydantic schema, exported to `schemas/vocabulary.json`.
- `prompts/extract_v1.md` with KJV idioms ("begat" → FATHER_OF, "knew his wife … bare" → SPOUSE_OF + MOTHER_OF). The few-shot example comes from Ruth 4, outside the eval scope.
- `llm.py`: provider adapters, structured output, retries, a sha256 disk cache, cost tracking, and a `KGRAG_BUDGET_USD` guard.
- `validate.py`: drops `named_in` verses that don't contain the name, checks that verses fall inside the passage, and enforces domain/range rules.
- Bake-off: 20 fixed passages with renames and shared names, two models. Pick the cheapest model within 3 F1 points of the best.

**M3 Vertical slice (12–16 h)**
- Extract all of Genesis.
- **Naive ER v0**: exact name match. It deliberately merges the two Lamechs, giving the "before" picture.
- `build.py`: support counts, degree, ForceAtlas2 layout, and Leiden clusters (`metric="modularity"`).
- `embed.py`: `BAAI/bge-small-en-v1.5`, quantized to int8 and dequantized, so Neo4j and the browser get identical vectors.
- `neo4j_load.py`: constraints, the two vector indexes, batched UNWIND loads.
- `packages/retrieval`: zod trace types (02 §4.4), the `GraphSource` interface, Local mode v0.
- `api`: `Neo4jGraphSource` plus Hono `POST /api/query` and `GET /api/graph`.
- `web`: query box, Sigma reducers that colour by status, a raw trace panel.

**M4 Entity resolution (14–20 h)**
- Build the scoreboard first: verse-overlap mapping, split and merge errors, mention F1, gender and place-kind accuracy. Run it on naive v0.
- Then the resolution steps:
  - Join aliases within a passage (Gen 17:5, 32:28, 35:10; Acts 13:9).
  - Cannot-link on a gender conflict or a different named parent; must-link on shared kin.
  - Embedding similarity for nearby same-name candidates.
  - LLM confirmation of the remaining ambiguous pairs.
  - `config/aliases.yaml`, keyed `Name@osisRef`.
- An audit report lists every merge.
- **Acceptance:** all 6 Genesis hard cases pass, and split + merge errors are lower than v0.

**M5 Benchmark harness (11–15 h, including 2 h of your labelling)**
- Triple scorer: parent edges directed, sibling and spouse edges undirected, plus a sibling closure variant.
- Event matching: verse Jaccard ≥ 0.25, fixed in advance.
- Narrative hand-label CSV with 100 edges.
- `gold_corrections.yaml`: the answer key links "land of Canaan" to the *person* Canaan 66 times (reported by the agent; re-check in M1). The report shows raw and corrected numbers.
- Golden regression test: commit the Genesis extractions and assert the whole `results.json`.
- UI `#/eval` page.

**M6 Local mode (11–15 h)**
- Seed linking: alias n-grams, choosing among same-name entities by embedding, with vector fallback.
- `ppr.ts`: Personalized PageRank (α 0.85, degree damping β 0.5, nodes in id order).
- Passage scoring and a context builder with E/R/C handles and a token budget.
- Structured cited answer, citation validation, and the promotion rules (02 §4.3).
- UI: clickable citations and funnel counts.

**M7 Path mode (5–8 h)**
- Cypher fetches a 2-hop corridor around X and Y.
- `paths.ts` runs Yen's k-shortest paths (k=3, at most 4 hops), with cost = support term + hub penalty.
- The path is narrated with citations.

**M8 QA evaluation (11–15 h, including 4 h of your question-writing)**
- You write 25 Genesis questions in `eval/questions.yaml` (single-hop, multi-hop kinship, path, memory trap).
- Vector and closed-book modes.
- `runEval.ts`.
- `score_qa.py`: verse recall, entity recall, invalid-citation rate, tokens, latency, $.
- An LLM judge from a different model family, checked against your grades with Cohen's kappa.

**M9 Full Bible (9–14 h)**
- Batch extraction of about 2,677 passages.
- ER at full scale.
- New Testament hard cases: Saul vs Paul, the 6 Marys, the 4 Judases, the 26 Zechariahs.
- Tribe membership graded.
- You add 15 questions that cross the Old and New Testaments.

**M10 Browser and parity (10–15 h)**
- `export_web.py`: graph with layout, passages per book, int8 embeddings, aliases, and a size-budget check.
- `jsonGraphSource.ts`.
- transformers.js embedder running in a Web Worker.
- `parity.test.ts`: Node + Neo4j must equal the browser over JSON for 40 questions × 2 modes.

**M11 Ship (6–9 h)**
- **Ask before creating the repo and pushing.**
- `pages.yml`; model and wasm files self-hosted.
- Strict CSP.
- BYOK: the key kept in memory, optionally in sessionStorage.
- 25 precomputed traces.
- Playwright smoke test.

**M12–M14:**
- Replay stepper with dropped candidates, split-screen compare, and a 66-book timeline.
- LightRAG in its own uv project under `benchmarks/lightrag/`.
- README with the results table first, a GIF, cost and latency tables, "Where it breaks" and Mermaid diagrams.

---

## Design details

**Repo layout**
```
docker-compose.yml  package.json  pnpm-workspace.yaml  biome.json  .githooks/  config/aliases.yaml
schemas/{vocabulary,trace.schema}.json
pipeline/src/kgrag/       cli corpus schema llm extract validate resolve build embed neo4j_load export_web prompts/
pipeline/src/kgrag_eval/  theographic mapping score_{mentions,entities,triples,events,qa} judge report   # only reader of gold
packages/retrieval/       trace graphSource jsonGraphSource seeds expand ppr paths passages context citations embedder modes/
packages/api/             server neo4jGraphSource cypher/*.cypher runEval test/parity
packages/web/             App GraphView pages/{Explore,Eval} retrieval.worker
eval/                     questions.yaml bakeoff_passages.txt gold_corrections.yaml labels/ fixtures/ results/
benchmarks/lightrag/      data/ (gitignored)
```
Intermediate files are JSONL plus `.npy`.

**Neo4j model**
- **Node labels:** `:Entity` plus one of `:Person`, `:Place`, `:Group` or `:Event`. Also `:Verse`, `:Passage`, `:Alias` and `:Meta`.
- **Structural edges:** `MENTIONED_IN` (Entity→Verse), `CONTAINS` (Passage→Verse), `ALIAS_OF` (Alias→Entity).
- **Semantic edges:** the approved types, created with dynamic types `[:$(row.type)]`.
- **Constraints:** uniqueness on the IDs. Community Edition supports nothing else, so the shape of the data is validated in Python.
- **Vector indexes:** `entity_embedding` and `passage_embedding`, 384-d, cosine, queried with the Cypher 25 `SEARCH … VECTOR INDEX`. The `queryNodes` procedure is deprecated.

**Retrieval split**
- **Neo4j:** seed lookup by alias and by vector, capped expansion one hop per query (fetches 2×top_m so the dropped candidates can be shown), induced edges, passage candidates.
- **Shared TypeScript:** PPR, Yen's paths, context building, trace building. GDS isn't in Community Edition, and the browser needs the same code.
- **The two implementations:** `GraphSource` has exactly two, `Neo4jGraphSource` and `JsonGraphSource`. Everything else is shared.

**Parity rules**
- ASCII IDs; every ordering breaks ties by id.
- Identical dequantized vectors on both sides.
- Exact re-ranking after the approximate vector search.
- transformers.js embeds queries in both Node and the browser.
- The same 0..1 cosine mapping on both sides.

---

## Cut list (drop from the top first)
1. LightRAG benchmark (saves 5–8 h)
2. Split-screen compare and timeline (saves 5–8 h)
3. Full replay → a simple 4-step stepper (saves 3–5 h)
4. Cross-passage event resolution and event grading (saves 3–4 h; needs your OK, because the schema is approved)
5. The LLM-confirmation pass in ER (saves about 3 h)
6. BYOK answers in the browser (saves 3–4 h)
7. Full Bible → a "genealogy spine" (Genesis, Ruth, 1–2 Samuel, Matthew, Luke, Acts) (saves 4–7 h)
8. In-browser search and parity (saves 10–15 h). Pages becomes replay-only.

**Never cut:**
- the benchmark harness
- entity resolution
- Neo4j and the Node API
- Local and Path modes

## Top risks
1. **KJV language hurts extraction** (pronouns, "begat", titles). The M2 bake-off measures this before any money is spent, and `validate.py` enforces the name-only rule.
2. **Shared names and renames break ER, and the errors cascade.** Cannot-link rules, LLM confirmation and `aliases.yaml`. Thresholds are tuned on Genesis; the full Bible is reported as held-out.
3. **The answer key is noisy** (Canaan person/place, siblings that are inferred). Report raw and corrected numbers and hand-check 30 false positives. It also makes a good interview point.
4. **Neo4j and the browser drift apart.** The parity rules above, plus a parity test in CI.
5. **Browser payload and WSL2 setup.**
   - The model is about 34 MB (bge-small q8, measured by the agent) and loads only when a question is typed.
   - Fix the pnpm shim.
   - Pin Docker memory: WSL has 11 GB, about 5 GB available (verified).

## Inputs needed from you
- **Before M2:** which LLM API keys you have (Anthropic, OpenAI or Gemini). This decides the bake-off pair.
- **M5:** about 2 h labelling 100 narrative edges.
- **M8 and M9:** about 4 h writing 25 Genesis questions, plus 15 cross-book questions.
- **M11:** your go-ahead to create the GitHub repo and push.

## Housekeeping on approval (before M0)
- Update `docs/discovery/00-decision-sheet.md`:
  - D1 text source.
  - Graph = full extraction, with Theographic as the benchmark only.
  - The approved schema.
- Copy this plan into the repo as `docs/execution-plan.md`.
- Add a memory note: build the core ourselves; outside datasets are benchmarks only.

## Verification (end to end)

| After | Check | Expected |
|---|---|---|
| M0 | `docker compose up -d --wait`, then `CALL dbms.components()` | `community`, `2026.09.0` |
| M1 | `uv run kgrag corpus build` | 31,102 verses |
| M1 | `uv run kgrag-eval gold-stats --scope Gen` | 2,151 person–verse pairs |
| M3 | `curl localhost:8787/api/query -d '{"text":"Who were the sons of Noah?","mode":"local"}'` | A trace with a Noah seed |
| M4 | `uv run kgrag-eval er --scope Gen --run naive` vs `--run v1` | Fewer split and merge errors; `MATCH (p:Person {name:'Lamech'})` returns 2 rows |
| M5 | `uv run kgrag-eval report --scope Gen`, then `uv run pytest -k regression` | Report written; regression test passes |
| M8 | `pnpm --filter api run-eval`, then `uv run kgrag-eval qa` | A table per mode; every trace matches `schemas/trace.schema.json` |
| M10 | `pnpm --filter api test:parity` | Identical traces; scores within 1e-6 |
| M11 | Playwright smoke test on the Pages build | No CSP violations; `localStorage` empty after a key is entered |
| Every commit | Pre-commit hook | `ruff format --check`, `ruff check` and `biome ci` pass |

## First task (under 30 min)

M0.1: get Neo4j running.
1. In the repo, run `git init` (local only).
2. Create `.env` with `NEO4J_PASSWORD=<8+ chars>`.
3. Add the compose file.
4. Run `docker compose up -d --wait`.
5. Open http://localhost:7474 and run `RETURN 1`.

This settles the riskiest setup question, Docker on WSL2, first.
