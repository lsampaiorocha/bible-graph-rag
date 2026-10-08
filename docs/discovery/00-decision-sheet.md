# 00 — Decision sheet (Phase 2 gate)

*2026-10-05. Merges the three discovery reports into the choices that block the execution plan. Nothing here is decided until you approve it.*

**Status:** Phase 1 (discovery) is done, and you are at Phase 2 (decisions). Phase 3, the execution plan, is written only after this sheet is approved.

## Decisions log

| # | Decision | Status (2026-10-05) |
|---|---|---|
| D1 | Books | **Approved: the Bible, King James Version** (Gutenberg 10; ~820k words / ~1.1M tokens, measured). This replaces the three candidates in `01-book-candidates.md`: you rejected them as too little-known. *Game of Thrones* and *The Godfather* were ruled out by copyright, because the hosted demo must show and ship the source text. |
| D2 | Live queries | **Approved:** visitor's own key (BYOK). Static Pages site, ~25 pre-run queries, in-browser retrieval, answers on the visitor's own key. $0 to you. |
| D3 | Scope | **Approved:** Option A: Local + Path search, plain-RAG baseline and no-retrieval baseline, the three-tier trace. No Global mode in v1. |
| B1–B5 | Stack | **Approved (revised 2026-10-07 to match the xNILIO job's stack):** build the core yourself, LightRAG as a benchmark, Sigma.js 3.0.x, Vite + React, with the storage changes in D4. |
| D4 | Storage and backend | **Approved: option 3, Neo4j plus a Node.js/TypeScript API.** This replaces "files only". |
| D5 | How the graph is built | **Approved (2026-10-07): full extraction.** Our LLM pipeline extracts every entity and relationship from the raw KJV text, and our own entity resolution merges and tells apart names. Theographic is **only the benchmark** (answer key), never an input. The text source is `verseText` from Theographic's `verses.json`: verse ID and text only, with a test that guards against leaking its tags. |
| D7 | LLM access and budget | **Approved (2026-10-08):** local runs go through `claude -p` on your Claude subscription, so there is no API key. Every call runs with settings, plugins, MCP servers and tools disabled. Budget caps, all in API-equivalent USD: $0.10 per call, $2 per run, $40 for the project; any run projected above $1 asks for confirmation. Every call is logged to `data/ledger/llm_calls.jsonl`, and `kgrag usage` summarizes the ledger. |
| D8 | Extraction model | **Approved (2026-10-08): `claude-haiku-5-5` at medium effort.** Bake-off on 20 Genesis passages: person F1 97.2, place F1 82.5, parent F1 86.6, 4.7% dropped, $0.0035 per passage, about $9.50 projected for the full Bible. Sonnet 5.5 scored 89.6 combined against Haiku medium's 88.7, at 11× the cost. Results: `eval/results/bakeoff/20261008T173615/`. |
| D9 | Graph visualization | **Changed (2026-10-08): `react-force-graph-2d` replaces Sigma.js.** Sigma, a WebGL renderer with a precomputed layout, felt static, showed no node info and needed custom camera code. react-force-graph gives live physics, hover tooltips, click, drag and particles along links. The focused view (the activated subgraph, revealed hop by hop) is the default; the whole graph remains as a toggle with pinned layout positions. Tradeoff: canvas rendering, so the whole-Bible overview may be slow at 5–10k nodes. |
| D10 | How questions are answered | **Approved (2026-10-08): a small query agent.** An LLM makes at most 3 tool calls per question, choosing from a fixed menu of tested graph-query tools: `find_entity`, `explore` (the Graph RAG neighbourhood search), `ancestors`/`descendants`, `relatives` and, from M7, `path`. Then it writes a cited answer. The LLM never writes raw Cypher. Why: the neighbourhood search ignores what a question asks ("ancestors of Noah" came back the same as "Noah": the father link ranked 16th of 45 and was capped out), while precise questions need exact, typed traversals. Keyless visitors get a keyword fallback to the same tools. This replaces the proposed standalone "lineage" feature. M4 still comes first. |
| D6 | Extraction schema | **Approved (2026-10-07):** Person, Place, Group (tribes, nations, sects) and Event; no objects. Graded relationships: family, MEMBER_OF, BORN_IN/DIED_IN, PARTICIPATED_IN/HAPPENED_AT. Ungraded narrative relationships: a fixed list of 12 plus OTHER. Full table: `docs/execution-plan.md`. |

**What D4 changes.** The target is xNILIO's Senior AI Engineer role, whose stack is a TypeScript monorepo (Vite frontend, Node.js backend), Python for batch work, and Docker. The role asks for graph databases, graph queries and vector databases.
- **Python** does batch work only: extraction, merging duplicate entities, embeddings, loading Neo4j, and evaluation.
- **Neo4j Community Edition, run through Docker Compose,** holds both the graph and the vector index. Community Edition has built-in vector search, available since 5.13 (verified). Retrieval is written in Cypher.
- **The Node.js/TypeScript API** handles local live mode. It holds your LLM key in `.env` and runs locally only, so nothing is hosted and there is nothing to pay for.
- **TypeScript monorepo** with three packages: a shared `retrieval` package that runs in both Node and the browser, `api` (Node) and `web` (Vite + React).
- **The public site does not change:** GitHub Pages, a JSON export of the graph, retrieval in the browser, and the visitor's own key.
- **Design point for the execution plan:** split the work between Neo4j and the shared TypeScript code. Neo4j handles seed lookup and neighbour expansion; the shared TypeScript handles ranking and building the trace. The parity test then compares Node + Neo4j against the browser over JSON.
- **Added effort (my estimate, not from the reports):** about 6–10 h for Neo4j, plus about 1–2 days for the Node API.

**The evaluation dataset for the Bible: Theographic Bible Metadata** ([repo](https://github.com/robertrouse/theographic-bible-metadata); checked 2026-10-05)
- **What it is:** an expert-curated knowledge graph containing 3,067 people, 1,274 places and 450 events. Each person record has `father`, `mother`, `children`, `siblings` and `partners` fields.
- **It is aligned to the same text.** `verses.json` holds the KJV text of every verse (e.g. `Gen.1.1`, "In the beginning God created the heaven and the earth."), together with the IDs of the people, places and events tagged in it.
- **That makes it a ready-made answer key.** It covers both entity extraction (which entities each verse mentions) and relation extraction (family edges). Interview line: "I measured my extracted graph against an expert-made one." Per D5, it is used only for grading.
- **License: CC-BY-SA-4.0.** Any file we publish that is derived from Theographic must carry the same license with attribution. The graph we extract from the public-domain KJV text is not affected.
- **Also available:** OpenBible.info's ~340,000 verse cross-references (CC-BY), a possible second check for which passages are related.

**Settled in the execution plan:** use the full Bible, not a subset. The default is to tune the pipeline on Genesis, which is about 38k words of dense family trees and quick to iterate on, then index all 66 books. The links between the Old and New Testaments are the best showcase for Path queries.

**What the approved choices imply (carry these into the execution plan):**
- **BYOK in v1 means retrieval also has to run in TypeScript** in the browser, with a CI parity test against the Python version (02 §6.4). It also needs an in-browser embedding model, about 20–35 MB on first load (that size is unverified), and the key-safety rules in 02 §6.5: strict CSP, key kept in memory or sessionStorage only, no third-party scripts.
- **Typed queries still work without a key.** Retrieval runs keyless, so any visitor can type a question and watch the subgraph light up. Only the written answer needs a key (02 §6.3).
- **The 52–85 h MVP estimate (03 §6) covers less than what was approved.** It assumes pre-run queries only, Local mode only, and a single baseline. Path mode, the no-retrieval baseline, and the BYOK/TypeScript port are all in its "polished" column. Neither report estimates the approved scope; the execution plan has to.

**Terms**
- **Local mode**: start from the entities named in the question, expand to their neighbours, and answer from what is collected. It does not use cluster summaries.
- **Path mode**: find the chain of links between two entities, for "how is X connected to Y?".
- **Global mode**: answer from LLM-written summaries of graph clusters, for whole-corpus questions such as "what are the factions?". This is Microsoft GraphRAG's signature mode.
- **Trace**: the JSON record each answer returns. It lists every entity, relationship, cluster and passage involved. The UI highlights from it.
- **Three-tier funnel**: each element in a trace is tagged with one of three levels:
  - *retrieved*: found by the search
  - *sent to the LLM*
  - *cited*: used in the answer

---

## A. Decisions that need your choice

### D1. Books — and what "a set of different books" means

| Option | Books (Gutenberg ID) | Size | Strength | Main risk |
|---|---|---|---|---|
| **Trojan War cycle** (recommended, 36/40) | *Iliad* 2199 and *Odyssey* 1727, Butler's prose translations; *Aeneid* 22456, Mackail's prose translation | 382k words / ~535k tokens | Every interviewer knows the cast. One story world across three books by two authors. The gods form ready-made factions. Nolan's *The Odyssey* (2026) makes it topical. | The translations use Roman names (Ulysses 128×, Odysseus 0×, Jove 475× in the *Iliad*), so you need an alias table so users can type Greek names. |
| **Plutarch, late Republic** (35/40) | 11 Lives from one file, 674 | 225k words / ~309k tokens | Strongest overlap between books: Caesar is named in 9 of the 11 Lives. Real history reads as "enterprise-like". No Graph RAG demo uses it. | The hardest name-merging: in Antony's Life, "Caesar" usually means Octavian. Less familiar to interviewers. |
| **Pyle's King Arthur** (34/40) | 60184, 10745, 33702, 60405 | 475k words / ~637k tokens | Clear factions and family ties across all four books. Strong queries that catch the LLM answering from memory, e.g. Mordred is Arthur's nephew in Pyle, not his son. | Archaic "thee/thou" prose. |

- **What "set" means:** books from one shared story world (recommended) vs. unrelated books. Unrelated books become disconnected islands in the graph, which weakens every cross-book query. *Source: 01 §2, §3.*
- **Cost does not decide it:** one extraction pass over any of the top 3 is under $1 on GPT-4o-mini and under $5 on Haiku 4.5.

### D2. Can visitors type their own queries? What monthly budget?

| Option | Visitor experience | Your cost | Abuse risk |
|---|---|---|---|
| **Static site with ~25 queries run in advance** (recommended for v1) | Browse curated queries and replay their traces. No typing. | $0 | None |
| + **Bring your own key** | Visitors type queries. Retrieval runs in the browser; answers use the visitor's own API key. | $0 | None, but you maintain a TypeScript copy of retrieval, checked against Python by a parity test |
| + **Hosted backend with your key** | Anyone types queries. | ~$0.001–0.004 per query plus hosting. Needs a rate limit and a spending cap. | Real |

Hosting facts checked 2026-10-05:
- Hugging Face Gradio/Docker Spaces need a paid plan; free accounts get only 2 ZeroGPU Gradio Spaces.
- Streamlit Cloud apps sleep after 12 hours without traffic.
- Fly.io has no free tier.
- GitHub Models shut down on 2026-07-30.

*Source: 02 §6.*

### D3. Scope of the first version

| Option | Includes | Effort |
|---|---|---|
| **Option A** (recommended) | Local mode, Path mode, Leiden clusters, a plain-RAG baseline and a no-retrieval baseline, the trace with its three-tier funnel | MVP ~50–85 h |
| **Option A+B** | Adds cluster summaries and Global mode | One more module and one more UI mode; about $0.3–2 more at indexing time. Neither report gives an hour estimate. |

**Tension:** some of the best showcase queries are whole-corpus questions (e.g. "Which gods backed the Greeks vs. the Trojans?"). Without Global mode, Local mode can only answer them partially. If those queries matter for the demo, choose A+B. *Source: 02 §1, §3; 03 §6.*

---

## B. Where the agents agree (confirm or override)

The architecture and tooling agents reached these separately:

1. **Build the core yourself, no framework:** NetworkX, an LLM SDK and embeddings. LightRAG (`aquery_data` returns structured provenance) runs on the same books as a benchmark and is the fallback if time runs short. Microsoft GraphRAG is out: its README says it is "largely in maintenance mode" (verified).
2. **Storage in files:** Parquet/JSON plus an in-memory vector index. No graph database. Kùzu was archived on 2025-10-10 (verified).
3. **Graph view:** Sigma.js 3.0.x with graphology (`latest` = 3.0.3; v4 is still in beta, verified). Runner-up: Cytoscape.js.
4. **Frontend:** a static Vite + React site on GitHub Pages.
5. **Differentiator:** the plain "highlight the retrieved subgraph" feature is common now. Two portfolio repos with it were created on 2026-09-28 and 2026-09-30 (verified). What would set this project apart:
   - the three-tier funnel
   - a step-by-step replay that also shows dropped candidates
   - a measured comparison against plain RAG in the UI

---

## C. Smaller decisions (can wait until the execution plan)

- Which extraction model and embedding model.
- Default view: the area around the answer, or the full graph. The tooling agent recommends the area around the answer, with a toggle for the full graph.
- How rigorous the evaluation is. Famous books are already known to the LLM, so lean on citation-based checks and on "what the LLM gets wrong from memory" queries.
- React vs. Svelte.

---

## Verification log (checked by the orchestrator, not just the agents)

| Claim | Check | Result |
|---|---|---|
| Gutenberg IDs and translators for the top-3 sets | Downloaded all 8 files and read their headers | All match (Butler ×2, Mackail, Pyle ×4, Plutarch) |
| *Iliad* never mentions the wooden horse | grep | 0 hits; *Odyssey* has 3 |
| Roman names in the Butler translations | grep | Ulysses 128, Odysseus 0, Jove 475 |
| Kùzu archived | GitHub API | archived, last push 2025-10-10 |
| GraphRAG in maintenance mode | README line 4 | Confirmed |
| LightRAG `aquery_data` exists | source | `lightrag/lightrag.py:4901` |
| Hugging Face Spaces need a paid plan | HF docs | Confirmed, with the 2-ZeroGPU-Space exception |
| GitHub Models retired | changelog and press | 2026-07-30 |
| Not checked | — | "Caesar named in 9 of 11 Lives"; the effort estimates; the per-query costs |
