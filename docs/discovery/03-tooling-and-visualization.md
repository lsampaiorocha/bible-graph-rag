# 03 — Tooling and visualization discovery: Graph RAG over public-domain books

*Discovery document · 2026-10-05 · Scope: the framework and library landscape, visualization UX, prior art, frontend stack, effort, and the interview angle. Pipeline shape, the trace contract, retrieval algorithms and deployment topology are covered in [02-architecture.md](02-architecture.md).*

**How to read this.**
- Stars, release dates and licenses come from the GitHub API, PyPI and npm, all queried on 2026-10-05.
- **(unverified)** means I could not confirm the claim from a primary source.
- **(inspected)** means I read the project's source code through the GitHub API, not just its README.
- Book examples (Darcy, Wickham) are only illustrations. The corpus is still undecided.

**Terms used throughout.**
- **Provenance**: the exact entities, relationships, communities and source chunks that a given answer was built from, available in a form code can read.
- **Activated subgraph**: those elements drawn on the graph for a single query.
- **Trace**: the JSON record of one query's provenance, as defined in 02-architecture.md. Every element in it has one of three statuses:
  - `candidate`: retrieved
  - `context`: sent to the LLM
  - `cited`: referenced in the answer
- **Precomputed trace**: a trace generated offline for a curated question and shipped as a static file.
- **BYOK** ("bring your own key"): the visitor pastes their own LLM API key, and the browser calls the provider directly.
- **"Built-it" signal**: how much of the core logic an interviewer would credit to you rather than to a library.

---

## TL;DR

- **Framework: build the core yourself, and keep LightRAG as an outside baseline.** Write extraction, entity resolution, retrieval and trace output yourself with NetworkX, an LLM SDK and an embedding model. Then run **LightRAG** on the same corpus as a comparison point in the evaluation.
  - LightRAG's `aquery_data()` returns structured entities, relationships, chunks and references (inspected). That makes it the best off-the-shelf fallback if time runs short.
  - Microsoft GraphRAG exposes the richest provenance. But it is the most expensive to index and is now "largely in maintenance mode".
- **Visualization: Sigma.js 3.0.x + graphology.**
  - Why: it renders with WebGL, its node and edge reducers implement highlight-and-fade natively, and it handles 1k–10k nodes when the layout is computed offline.
  - Use the stable 3.0.x line: v4 is still in beta (4.0.0-beta.8 was published today).
  - Runner-up: **Cytoscape.js**, if you only ever show 3k nodes or fewer and want the easiest step-by-step animation plus compound nodes for communities.
- **Frontend: a static Vite + React app (using `@react-sigma`) on GitHub Pages, serving precomputed traces.**
  - Live queries are optional: run retrieval in the browser and generate answers with the visitor's own key (BYOK), or add a small backend. 02-architecture.md covers the options.
  - Python-only UIs (Streamlit, Gradio) get to an MVP fastest, but they cannot run on GitHub Pages and their graph components are stale.
  - Hugging Face now requires a paid plan to create Gradio or Docker Spaces.
- **Effort** (one experienced Python developer with an AI coding assistant):
  - MVP: **about 50–85 hours** (roughly 1.5–2 weeks full time).
  - Polished portfolio version: **about 130–190 hours** (roughly 4–5 weeks).
  - Most of that time goes into tuning extraction and entity resolution and polishing the visuals, not into writing code.
- **Distinctiveness: highlighting the retrieved subgraph is no longer unusual.** At least two public portfolio repos with exactly that feature were created in the last two weeks. To stand out, add:
  - a **retrieval replay** that steps through what happened, showing what was considered and what was dropped
  - a **candidate → context → cited funnel**
  - a **measured comparison with vanilla RAG**, including the cases where the graph loses
  - features specific to books: visible alias resolution, and a chapter timeline showing where the evidence comes from

---

## 1. Graph RAG frameworks: comparison

### 1a. Maintenance facts (as of 2026-10-05)

| Option | Latest release (date) | Last push to repo | License | Stars | Notes |
|---|---|---|---|---|---|
| **Microsoft GraphRAG** | `graphrag` 3.2.0 (2026-09-23) | 2026-10-05 | MIT | 36.2k | README: "largely in maintenance mode, and won't be accepting new PRs or implementing new features." The canonical getting-started corpus is *A Christmas Carol*. |
| **LightRAG** (HKUDS) | `lightrag-hku` 1.5.7 (2026-09-02) | 2026-10-03 | MIT | 40.0k | Very active. Ships a WebUI that uses sigma.js. Its demo corpus is also *A Christmas Carol*. |
| **nano-graphrag** | 0.0.8.2 on PyPI (2024-10-19) | 2026-01-27 (a setup.py fix) | MIT | 4.0k | Dormant. Small and readable: 18 `.py` files, about 179 KB. |
| **fast-graphrag** (Circlemind) | 0.0.5 (2025-04-04) | 2025-11-01 (README edit) | MIT | 4.0k | Stale for about 11 months. |
| **LlamaIndex PropertyGraphIndex** | `llama-index-core` 0.14.25 (2026-09-21) | 2026-10-05 | MIT | 52.4k (whole repo) | Part of the general LlamaIndex framework. |
| **LangChain LLMGraphTransformer** | Moved to `langchain-neo4j` 0.10.0 (2026-06-10) | 2026-09-18 | MIT | 41 (langchain-neo4j repo) | `langchain-experimental`, its old home, was **sunset and archived on 2026-05-22** (issue #87). |
| **neo4j-graphrag** (Python) | 1.22.0 (2026-10-01) | 2026-10-05 | Apache-2.0 (per PyPI) | 1.3k | Official Neo4j package. |
| **Graphiti** (Zep) | `graphiti-core` 0.30.2 (2026-09-08) | 2026-10-05 | Apache-2.0 | 31.4k | Temporal "context graph" for agent memory. Kuzu backend deprecated. |
| **Cognee** | 1.6.2 (2026-09-29) | 2026-10-05 | Apache-2.0 | 31.4k | Agent-memory product. Its main API is `remember`/`recall`. |
| **HippoRAG 2** (OSU) | `hipporag` 2.0.0a4 alpha (2025-06-24); GitHub release v1.0.0 (2025-02-27) | 2026-10-01 | MIT | 4.0k | Research code. Recent commits, but releases are alpha. |
| *Newer, 2025–26:* RAG-Anything | 1.4.1 (2026-09-02) | 2026-10-05 | MIT | 23.5k | Multimodal LightRAG sibling. |
| KAG (OpenSPG) | — | 2026-01-28 | Apache-2.0 | 9.1k | Logical-form reasoning. Heavy engine. |
| Youtu-GraphRAG (Tencent, ICLR 2026) | — | 2026-02-26 | NOASSERTION | 1.3k | Research. |
| LinearRAG (ICLR 2026) | — | 2026-07-05 | **GPL-3.0** | 553 | Research. Copyleft license. |
| FalkorDB GraphRAG-SDK | 1.4.0 (2026-08-10) | 2026-10-05 | Apache-2.0 | 1.0k | Requires FalkorDB. |
| AWS graphrag-toolkit | — | 2026-10-05 | Apache-2.0 | 445 | AWS-oriented. |
| kg-gen (NeurIPS '25) | 0.4.0 (2025-09-30) | 2026-03-24 | none on GitHub | 1.3k | Extraction only. |
| **Build it yourself (baseline)** | NetworkX 3.7 (2026-09-21) plus an LLM SDK and embeddings | — | — | — | NetworkX 3.7 includes `leiden_communities`/`leiden_partitions` and `forceatlas2_layout`. |

Ecosystem events that affect these choices:
- **Kuzu**, an embedded graph database, was **archived on 2025-10-10**. Graphiti marks its Kuzu driver deprecated.
- **LazyGraphRAG** is still not in the open-source GraphRAG repo: no path containing "lazy" exists on `main` (inspected). The tracking discussion #1490 has had no maintainer update since 2025.

### 1b. Fit for this project

The column that matters most is the second one: can you get provenance out in a form code can use?

| Option | Provenance for an answer | Customizing the extraction schema | LLM providers | Infrastructure | Indexing cost profile | "Built-it" signal |
|---|---|---|---|---|---|---|
| **Microsoft GraphRAG** | **Excellent** (inspected).<br>`local_search`, `global_search`, `drift_search` and `basic_search` return `(response, context_data)`. `context_data` is a dict of DataFrames (`entities`, `relationships`, `reports`, `sources`, `claims`).<br>Candidates carry an **`in_context` flag**, which maps directly onto the candidate → context statuses.<br>The prompts make the answer cite record IDs inline, e.g. `[Data: Entities (5, 7); Relationships (23)]`. This is parseable but not validated. It gives no hop or score detail. | `entity_types` in the config, plus auto and manual prompt tuning.<br>**Bring-your-own-graph** lets you supply your own entities/relationships tables and use GraphRAG only for communities and querying. | LiteLLM, so any model that returns JSON-schema structured output works. OpenAI models are the best tested. The docs warn that Ollama behind a proxy often returns malformed JSON. | Parquet files on disk. Python `<3.14`. | **Highest.** The LLM extracts entities and relations, then summarizes descriptions and community reports. Graph extraction is about 75% of indexing cost (per the docs). `--method fast` uses NLP noun phrases instead, which is cheaper but noisier. | **Low.** It is a CLI plus YAML configuration. Entities are merged on exact `(title, type)` only (inspected), so alias resolution is left to you. |
| **LightRAG** | **Excellent** (inspected).<br>`aquery_data()` returns `entities`, `relationships`, `chunks` and `references`, each with a `reference_id`. It also returns the extracted high- and low-level keywords and truncation statistics (found vs. kept after truncation).<br>`aquery_llm()` returns the same data plus the answer.<br>There is **no community layer**: its "global" mode works on relationship keywords. | An entity-type prompt override (`addon_params`, `ENTITY_TYPE_PROMPT_FILE`) and custom prompts. | OpenAI, **Anthropic**, **Ollama**, Gemini, Bedrock, Hugging Face and more (provider modules confirmed in the source). | Defaults to NetworkX plus JSON and vector files. Optional backends include Neo4j, Postgres, Qdrant and Milvus. | Moderate: LLM extraction and description merging per chunk, with no community reports. | **Low to medium.** The codebase is large: about 7 MB of `.py`, and `lightrag.py` alone is about 400 KB. |
| **nano-graphrag** | **Medium.** `only_need_context=True` returns the context as a single string of CSV sections (`-----Entities-----`, `-----Relationships-----`, `-----Sources-----`) that you must parse (inspected). | Edit the prompt dictionary. | OpenAI, Azure and Bedrock are built in. Anything else needs a custom async function. | Files (GraphML, nano-vectordb). | Similar to GraphRAG, since it also builds community reports. | **Medium.** It is small enough to read end to end, but dormant. |
| **fast-graphrag** | **Good** (inspected). `TQueryResponse.context` holds `entities`, `relations` and `chunks`, **each paired with a score** from Personalized PageRank, which is ideal for a heat-map view. | A domain description, entity types and example queries, set at initialization. | OpenAI-compatible APIs and Gemini. Anthropic support is unverified. | Files (igraph, hnswlib, pickle). | Low to moderate. No community reports. | **Low.** And it is stale. |
| **LlamaIndex PropertyGraphIndex** | **Partial.** Results come back as `NodeWithScore` items where graph paths are plain text (`subj -> rel -> obj`), optionally with source chunks attached (`include_text`). You have to map them back to graph IDs. | `SchemaLLMPathExtractor` (strict schema) and `DynamicLLMPathExtractor`. | All, through LlamaIndex integrations. | In-memory `SimplePropertyGraphStore`, or Neo4j, Nebula, TiDB or FalkorDB. | Moderate. No communities in the core library. | Low to medium. |
| **LangChain LLMGraphTransformer** | **Not applicable.** It only does extraction. Retrieval goes through Text2Cypher on Neo4j. | `allowed_nodes`, `allowed_relationships` (including typed triples) and `node_properties`. | Any LangChain chat model. | `GraphDocument` objects; Neo4j to query them. | Low to moderate. | Low. The package also just moved. |
| **neo4j-graphrag** | **Good, because you define it.** `GraphRAG.search(return_context=True)` returns `retriever_result.items`. With `VectorCypherRetriever`, you write the Cypher that returns node and relationship IDs. | `SimpleKGPipeline` with a schema of node types, relationship types and patterns. | OpenAI, Anthropic, Ollama, Gemini/Vertex, Mistral, Cohere, Bedrock (modules confirmed in the source). | **Requires Neo4j.** On Aura Free, an instance with no writes for 3 days is paused, and a paused instance is deleted after 30 days. That is a real risk for a demo link an interviewer opens weeks later. | Moderate. | Medium: you write the retrieval queries. |
| **Graphiti** | **Good** (inspected). `search()` returns `EntityEdge` facts with source and target node UUIDs and an `episodes` list, which records where each fact came from. `search_()` returns edges, nodes, episodes and communities. | Custom entity and edge types as Pydantic models. | OpenAI, Anthropic, Gemini, Groq, plus OpenAI-compatible APIs. | Neo4j, FalkorDB or Neptune. | High per episode, because each episode goes through temporal deduplication and invalidation. The exact multiplier is unverified. | Low. It is designed for agent memory, which is a poor fit for static books. |
| **Cognee** | **Good, per its docs.** `include_references=True` adds `metadata.evidence`, a list of the "chunks, graph nodes, and graph edges placed in the LLM context". `only_context=True` returns the context instead of an answer. | Custom graph models (unverified for 1.6). | Several providers (exact list unverified). | Embedded defaults. The default graph backend is unverified. | Moderate. | Low. It is product-shaped. |
| **HippoRAG 2** | **Partial** (inspected). `QuerySolution` contains `docs`, `doc_scores` and `graph_seeds`. The PPR scores for individual nodes stay internal. | Fixed OpenIE triple prompts, not driven by a schema. | OpenAI-compatible APIs, including vLLM. | Files (igraph). | Moderate: OpenIE extraction per passage. | Medium. |
| **Build it yourself** | **Complete by construction.** You emit exactly the trace the UI needs: seed scores, hop numbers, dropped candidates, and validated citation IDs. | Complete control. For example, a Pydantic schema per corpus with Character, Place, Group and Object types and a fixed relation vocabulary (see 02-architecture.md). | Any, called directly through the SDK. | NetworkX in memory, JSON/Parquet exports, numpy for vectors. | **Lowest possible.** You control the prompts, chunk size, gleaning (an optional second extraction pass per chunk), batching and caching. | **Highest.** |

**Indexing cost, back-of-envelope (an estimate, not a measurement).**

For about 500k words, which is about 665k tokens:
- **A lean extraction you build yourself** uses roughly **2–3M input and 0.4M output tokens**. That assumes 1,200-token chunks, a prompt of about 1k tokens, structured output of about 0.5k tokens, no gleaning, and light community summaries.
- **Microsoft GraphRAG on standard settings** plausibly uses **3–5 times that**, because of its few-shot extraction prompts, gleaning, description summarization and multi-level community reports.

At illustrative prices per million tokens (input/output):

| Price tier (per M tokens, in/out) | Lean, built yourself | GraphRAG standard |
|---|---|---|
| $0.15 / $0.60 | about $0.6 | about $3 |
| $1 / $5 | about $4 | about $20 |
| $3 / $15 | about $13 | about $60 |

- Batch APIs from both OpenAI and Anthropic cut these costs by 50%.
- Expect 3–5 full re-index runs while you tune prompts. Cache LLM responses by (prompt hash, chunk hash) so that unchanged chunks are never re-billed.
- One secondary source reports $6–7 to index *A Christmas Carol* (about 32k words) with GPT-4o through GraphRAG (unverified). That is the same order of magnitude as the table above.
- 02-architecture.md has per-model figures.

**Verdict.**
- **Build the core yourself.** No framework gives you hop-by-hop scores, dropped candidates and validated citations together, and those are exactly what the visualization needs to look unique. Building it also gives you the strongest interview story.
- **LightRAG is the best fallback** if you are short on time. It has the cleanest structured provenance API (`aquery_data`), supports Anthropic and Ollama, uses file-based NetworkX storage, and is actively maintained.
- **Use LightRAG as an outside baseline** in your evaluation either way: "my pipeline vs. LightRAG vs. vanilla RAG on the same 30 questions".
- **Microsoft GraphRAG has a useful middle path.** Its *bring-your-own-graph* mode lets you keep your own extraction and resolution and borrow its community reports. But it is in maintenance mode, so do not build on it.

---

## 2. Graph visualization libraries: comparison

**The single most important performance decision is to compute the layout offline.** Run ForceAtlas2 once, with NetworkX's `forceatlas2_layout` or graphology's FA2 in a build script, and ship x/y coordinates with the graph. Then the browser only renders, and every WebGL library handles 10k nodes comfortably. Running a live force simulation on 10k nodes is what makes demos stutter.

| Library | Rendering | Performance at 1k–10k nodes | Highlight a subset, fade the rest | Animation / step-through | Communities | Static hosting | License | Learning curve | Maintenance (2026-10-05) |
|---|---|---|---|---|---|---|---|---|---|
| **Sigma.js 3 + graphology** | WebGL | Official claim: "thousands of nodes and edges". In a 2022 Memgraph benchmark (layout plus render), 10k nodes and edges took 5–10 s, against 27 s for D3 on canvas. With a precomputed layout, rendering is near-instant. | **Native.** `nodeReducer`/`edgeReducer` restyle on every frame, and an official example dims everything except a node's neighborhood. | `animateNodes` utility, camera animation, and state changes followed by `refresh()`. Easy to drive from a step index. | Color by attribute. No built-in hulls (outlines drawn around a group of nodes). | Yes | MIT | Moderate: two libraries, plus the reducer pattern. | 12.2k stars. sigma 3.0.3 is stable (2026-04-30); **4.0.0-beta.8 came out today**. About 538k weekly npm downloads. Used by LightRAG's WebUI and Gephi Lite (both confirmed in their `package.json`). |
| **Cytoscape.js** | Canvas, plus a WebGL **preview** since 3.31 | Official benchmark: 1.2k nodes with 16k edges ran at 20 FPS on canvas and over 100 FPS with WebGL. 3.2k nodes with 68k edges ran at 3 and 10 FPS. | **Very easy.** CSS-like selectors and classes (`.faded`, `.active`) with style transitions. | `ele.animate()` and class toggles with transitions. The easiest step-through of any library here. | **Compound nodes** can group a community visually, at a performance cost. | Yes | MIT | **Low.** Excellent docs. | 11.2k stars. v3.34.3 (2026-09-07). The WebGL renderer has edge-style limits (no dashed edges, solid colors only). |
| **vis-network** | Canvas | **Weak.** In the 2022 Memgraph benchmark, 1k nodes and edges took 1 min 14 s with physics on, and 10k took "very long". Better with physics off and fixed positions (unverified by how much). | `DataSet.update` of colors. Workable. | Limited (focus, `moveTo`). | A built-in clustering API. | Yes | Apache-2.0 or MIT | Low | 3.6k stars. v10.1.2 (2026-08-19). |
| **react-force-graph / force-graph / 3d-force-graph** | Canvas in 2D, WebGL (three.js) in 3D | Fine at a few thousand nodes if you stop the simulation (`cooldownTicks`) or pin positions (`fx`/`fy`). Exact figures at 10k unverified. | Official "highlight nodes/links" example. | **`linkDirectionalParticles` and `emitParticle`** animate particles travelling along edges, which suits showing "retrieval flowing along edges". | Color only. | Yes | MIT | **Low.** | 3.3k / 2.1k / 6.4k stars. Releases in late September 2026. Used by graphrag-visualizer and a recent "query my CV" GraphRAG portfolio. 3D looks impressive but is harder to read. |
| **AntV G6 5** | Canvas, SVG and WebGL renderers | A 2025 study found G6 on canvas slower than D3 on canvas or WebGL for force-layout time. With a precomputed layout, fine. | Built-in element **states** (highlight, inactive). | Built-in animations. | **Hull and BubbleSets plugins** draw community outlines (plugin files confirmed in the repo). | Yes | MIT | Moderate to steep. English docs exist, but the ecosystem is mostly Chinese. | 12.3k stars. 5.1.1 (2026-05-08). |
| **Reagraph** | WebGL (three.js), React only | Unverified at 10k. | "Highlight and Selection Hook" (`actives`/`selections`). | Limited (unverified). | Built-in clustering by attribute. | Yes | Apache-2.0 | Low for React developers. | 1.1k stars. 4.32.0 (2026-06-25). About 76k weekly downloads. |
| **D3-force** | Layout only; you render (SVG or canvas) | SVG struggles beyond about 1–2k elements; canvas is fine. 10k took 27 s in the Memgraph benchmark. | You build it yourself. | Full control; you build it yourself. | You build it yourself. | Yes | ISC | **Steep** for this use case. | v3.0.0 dates from 2021. Stable and effectively frozen. |
| **Neo4j NVL** | WebGL and canvas | Good. | Yes. | Yes. | — | — | **Proprietary.** Its LICENSE.txt allows use **only with Neo4j Aura or Neo4j commercial products.** | Moderate | 2.0.0 (2026-09-15). **Not suitable** unless you adopt Aura. |
| **cosmos.gl** (`@cosmos.gl/graph`) | GPU layout and rendering | Built for 100k to 1M+ nodes. | Configuration-driven `highlightedPointIndices`/`highlightedLinkIndices`. | Limited. | Color only. | Yes | MIT. The older `@cosmograph/cosmos` package is **CC-BY-NC-4.0**, so check which one you install. | Moderate | 3.5.0 (today). Overkill at 10k nodes, and labels are weak. |
| *Python:* **pyvis** | Exports HTML using vis-network | Inherits vis-network's limits. | Static only. | None. | — | Its HTML export can be served statically. | BSD-3 | Very low | **Dormant**: 0.3.2 (2023-02), last push 2024-04. |
| *Python:* **streamlit-agraph** | Wraps react-graph-vis (vis.js) | vis-network limits. | Basic. | None. | — | Needs a Streamlit server. | MIT | Very low | **Stale**: 0.0.45 (2023-01). |
| *Python:* **st-link-analysis** | Cytoscape.js inside Streamlit | Cytoscape limits. | Yes. | Limited. | — | Needs a Streamlit server. | (unverified) | Low | 0.4.0 (2025-06). 133 stars. |
| *Python:* **Gradio custom component** | Whatever you wrap | — | — | — | — | Needs a Gradio server. | — | Moderate to high: you write a Svelte component. | I found no maintained, first-party graph component (unverified). |

**Recommendation.**
- **Use Sigma.js 3.0.x with graphology, through `@react-sigma` if you choose React.** Its reducer model is exactly "highlight a set of IDs and fade the rest", it is WebGL, it is MIT, and LightRAG's WebUI is a working reference implementation of the same stack.
- **Pin sigma to 3.0.x** until v4 is stable.
- **Choose Cytoscape.js instead** if the default view is a focused subgraph of at most a few hundred nodes and the full-graph view is secondary. Its class transitions make the step-through replay trivial, and compound nodes show communities nicely.
- **Add react-force-graph particles later**, as optional polish, only if you want animated flow along edges.

---

## 3. Visualization UX concepts

All three concepts consume the same trace (see 02-architecture.md): elements with a status of `candidate`, `context` or `cited`, plus `hop`, `score` and `step`. Precomputed and live traces look identical to the UI.

### Concept A — "Spotlight": the full graph with the activated subgraph lit up (the default view)

```
+--------------------------------------------------------------------------+
| [ Ask: "Why does Darcy change his opinion of Elizabeth?"        ] [Ask]  |
| Mode: (o) Graph  ( ) Vanilla  ( ) Both       Curated questions v         |
+-----------------------------------------------+--------------------------+
|  .  .   .    .   (full graph, faded to 10%)   | ANSWER                   |
|    .    (DARCY)=====(ELIZABETH)    .    .     | Darcy's view shifts after|
|  .   //     ||          \\                    | the refused proposal     |
|     (WICKHAM) (HUNSFORD)  (LETTER)    .   .   | [E1][R4] and his letter  |
|   .      .    \\_______//       .             | [P12].                   |
|  .  .    [ community: Pemberley visit ]   .   |                          |
|       .  .      .     .     .     .           | EVIDENCE   (click = zoom)|
|                                               | Entities 6/14  Rels 9/31 |
| Legend: (*) seed   bright = cited             | Communities 2  Chunks 5  |
|         medium = context   dim = candidate    | > P12 ch.35 "Be not      |
|         ==== cited edge    ---- context edge  |   alarmed, madam, ..."   |
+-----------------------------------------------+--------------------------+
```

- Hovering a citation like `[E1]` pulses that node. Clicking a node opens its description and the chunks that support it.
- The "6/14" counts mean 6 entities were sent to the LLM out of 14 candidates. That shows the funnel at a glance.
- **Pros:** it shows the scale of the graph and where the answer sits within it, so it looks impressive.
- **Cons:** at 10k nodes the lit region is small, so auto-zoom to its bounding box.

### Concept B — "Retrieval replay": step through each retrieval stage

```
+--------------------------------------------------------------------------+
| Step 3 of 6: Hop-1 expansion       [|<] [<] [ > Play ] [>] [>|]   1x v   |
| (1) Embed query > (2) Seeds > (3) Hop-1 > (4) Hop-2/PPR >                |
| (5) Chunks + budget cut > (6) Answer + citations                         |
+-------------------------------------+------------------------------------+
|     focused subgraph (ego view)     | THIS STEP                          |
|                                     | Expanded 3 seeds along 14 edges;   |
|        (WICKHAM)+                   | kept 9 by weight x relevance.      |
|            \                        |                                    |
|   (DARCY)*=====(ELIZABETH)*         | Seeds (cosine): DARCY .82,         |
|      |      \                       |   ELIZABETH .79, LETTER .61        |
|  (PEMBERLEY)+  (LETTER)*            | New this step (+): WICKHAM,        |
|                                     |   PEMBERLEY, ... (+6 more)         |
|  * seed   + new this step           | Dropped (grey): 5 below threshold  |
+-------------------------------------+------------------------------------+
| Context budget [###########-----] 5.2k / 8k tokens  |  cost $0.0004      |
+--------------------------------------------------------------------------+
```

- Each step adds a CSS class (Cytoscape) or changes a reducer state (Sigma) for the elements whose `step` is at most the current step.
- The "budget cut" step greys out candidates that were dropped. **No tool I surveyed shows dropped candidates** (see §4). This is the strongest differentiator, and the easiest thing to explain in an interview.
- **Pros:** it teaches the algorithm, and it is easy to record as a GIF for the README.
- **Cons:** it needs a trace with step and score detail, which a framework will not give you.

### Concept C — "Graph vs. vanilla" split screen, with a book timeline

```
+--------------------------------------------------------------------------+
| Q: "How are Wickham and Darcy connected?"                    [Run both]  |
+------------------------------------+-------------------------------------+
| VANILLA RAG (top-5 chunks)         | GRAPH RAG (activated subgraph)      |
|  #1 ch.16  .71 "...Mr. Wickham..." |  (WICKHAM)--godson of--(OLD DARCY)  |
|  #2 ch.35  .69 "...his father..."  |      \                      |       |
|  #3 ch.41  .64 ...                 |   eloped w/            father of    |
|  #4 ...                            |        \                    |       |
|                                    |  (GEORGIANA)--sister of--(DARCY)    |
| Answer: "Wickham says Darcy..."    | Answer: "Wickham was the steward's  |
|                                    |  son, raised with Darcy..."         |
| judge 2/5 | 1.1k tok | 0.8s | $.0002| judge 5/5 | 4.3k tok | 2.1s | $.0009|
+------------------------------------+-------------------------------------+
| Book timeline  ch.1 |---v-------V---------vV---------V---| ch.61         |
|                 v = vanilla chunk   V = graph chunk   vV = both          |
+--------------------------------------------------------------------------+
```

- **Pros:** this is the evaluation story made visible, including the cost and latency tradeoff. The chapter timeline is specific to books, and nobody in the prior art has one.
- **Cons:** it needs two pipelines and a judge. On a narrow screen, stack the two halves vertically.

**Recommended composition.**
- **MVP:** A, with clickable inline citations, plus a simplified B: a Prev/Next stepper over 4 stages, no budget bar.
- **Polished version:** add the full B and C.
- Make the **default view an ego subgraph**, with a "show full graph" toggle. The ego subgraph is the activated nodes plus one hop of context. It reads better on phones and at 10k nodes.
- *Alternative to consider:* an "evidence funnel" bar chart (candidates → context → cited, for each element type) sitting above the answer. It is cheap to build and makes "retrieved but unused" visible.

---

## 4. Prior art

| Project | What it shows | What it lacks for this project |
|---|---|---|
| **LightRAG WebUI** ([HKUDS/LightRAG](https://github.com/HKUDS/LightRAG), 40k stars) | A sigma.js graph explorer (search, layouts, merging and editing entities) and a separate Retrieval chat tab. | **The chat and the graph are not linked.** I inspected `features/retrieval/*`: it never references the graph viewer, so a query never lights up the graph, even though `aquery_data` returns everything needed to do so. Self-hosted only. |
| **Neo4j LLM Graph Builder** ([neo4j-labs/llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder), 5.3k stars; hosted at llm-graph-builder.neo4jlabs.com) | A chat "Details" modal with tabs for Sources, Entities, Chunks and Communities, showing "the contextual graph data (communities, entities, chunks) that went into generating the answer" (Neo4j blog). Several retrieval modes. Renders graphs with NVL (`ChatInfoModal`, `EntitiesInfo`, `CommunitiesInfo` components). | You must connect **your own Neo4j instance**, so there is no zero-setup demo. No step-through, no evaluation numbers. It is generic over documents, with nothing book-specific. NVL's license ties it to Neo4j. |
| **Microsoft GraphRAG unified-search-app** (in the GraphRAG repo) | A Streamlit app that compares global, local, DRIFT and basic search on one question, with citations, plus a Community Explorer tab. | No graph rendering of the context (per its README). Its README says it is "not supported". The project is in maintenance mode. |
| **graphrag-visualizer** ([noworneverev](https://github.com/noworneverev/graphrag-visualizer), 425 stars; GitHub Pages) | A static site: upload GraphRAG parquet files and view the graph in 2D or 3D (react-force-graph). With a local `graphrag-api` server, a search **filters the graph to the `context_data` entities and relationships** (inspected `updateGraphData`). | **The closest prior art.** Still: it requires the user's own parquet files and a local server, does not link answer text to nodes, has no step-through and no baseline or evaluation, and has had no commits since 2026-02. |
| **graphrag-workbench** ([lyon-industries](https://github.com/lyon-industries/graphrag-workbench), 736 stars) | A local 3D workbench around GraphRAG 3.1. It isolates communities, inspects entities, and fills in the graph live as indexing runs. | "Chat is intentionally absent from 2.0", so it has no query provenance at all. Local only. |
| **Kotaemon** ([Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon), 25.8k stars) | Document chat with citations highlighted in a PDF viewer, and an optional GraphRAG or nano-graphrag index with a graph preview. | A general document-QA product. Its GraphRAG integration is pinned to `graphrag<=0.3.6`. No retrieval replay. |
| **LovichiDorian/GraphRag** (created 2026-09-30, 0 stars) | A portfolio project: "query my CV". The evidence subgraph is highlighted in a 3D graph, hovering a citation shows the passage, and plan, retrieval and Cypher steps are visible. Also an MCP server. | **Evidence that this exact feature is now a common portfolio pattern.** Neo4j-backed. A single-person corpus. |
| **HuzaifaAqeel/Minimal-GraphRAG** (created 2026-09-28, 0 stars) | A Streamlit graph with the retrieved subgraph highlighted next to the answer. | Same signal: the basic highlight feature is common. |
| **graphsight** ([Kcodess2807](https://github.com/Kcodess2807/graphsight)) | Traces LangGraph agents and marks each retrieved item as "overlaps the answer" or "no lexical trace in answer". | Not a graph RAG system, but it validates the idea of **retrieved vs. used** as something worth showing. |
| **novel-graph** ([schoobani](https://github.com/schoobani/novel-graph); GitHub Pages app) | LLM-extracted character interaction graphs for novels, including *War and Peace*. | No RAG and no querying. Last push 2025-02. |
| **Memgraph ai-demos** | A GraphRAG demo on a Game of Thrones dataset. | A notebook-style demo with no provenance UI. |

**Conclusion: what would make this distinct rather than a clone.**
1. **Retrieval replay with dropped candidates.** It shows *why* each element is in the context (hop, score, step) and what was cut at the budget step. No surveyed tool shows dropped candidates or per-step expansion.
2. **A three-tier funnel, candidate → context → cited, with validated citations.** GraphRAG's `in_context` flag and inline `[Data: ...]` citations come closest, but nobody visualizes all three tiers.
3. **A measured comparison with vanilla RAG in the UI**, using the same question, judge scores, tokens, latency and cost, plus an honest "where graph loses" page. That page is backed by the literature: GraphRAG-Bench reports that "GraphRAG frequently underperforms vanilla RAG on many real-world tasks."
4. **Book-specific features.**
   - Visible **alias resolution** (e.g. "Darcy ← Mr. Darcy, Fitzwilliam Darcy"). Both major frameworks merge only exact names.
   - A **chapter timeline** showing where the evidence sits.
   - Optionally, spoiler-bounded querying ("as of chapter N").
5. **A demo that is always on and costs nothing.** Precomputed traces on GitHub Pages, with live querying optional. Neo4j-based demos break when Aura Free instances pause.
6. **Avoid *A Christmas Carol*** as the corpus. It is the getting-started corpus for both Microsoft GraphRAG and LightRAG, so it reads as a tutorial.

---

## 5. Frontend stack: comparison

| Option | Works on static GitHub Pages? | Live LLM queries | Graph visualization at 1k–10k nodes | Speed for a Python developer | Free hosting in 2026 | Interview signal |
|---|---|---|---|---|---|---|
| **Streamlit** | No. It needs a server. | Server-side. Easy. | Through components that are stale (agraph, last release 2023-01) or limited (st-link-analysis). Each interaction reruns the script. | **Fastest** | Community Cloud: apps sleep after **12 h** without traffic, with 690 MB–2.7 GB of memory. | Medium. Very common. |
| **Gradio** | No. | Server-side. | Requires writing a custom Svelte component. | Fast | **HF Spaces: Gradio and Docker Spaces now need a paid plan to create.** The exceptions are static Spaces (free) and 2 ZeroGPU Gradio Spaces per free account. | Medium |
| **NiceGUI** (16.3k stars, v3.17.1) | No. It is FastAPI-based. | Server-side. | Custom JS components can wrap sigma. | Fast | Same as any server (02-architecture.md lists the options). | Medium |
| **stlite** (Streamlit running in the browser via Pyodide) | **Yes.** | Only by BYOK from the browser. | Same component limits as Streamlit. | Fast | GitHub Pages | Low to medium. Heavy first load (download size unverified). |
| **Gradio-Lite** | Yes, in principle. | BYOK | — | — | — | **Appears unmaintained.** Last release 5.45.0 (2025-09), while Gradio is at 6.29. |
| **Vite + React + `@react-sigma`** | **Yes** (build with a GitHub Actions workflow). | Retrieval in the browser plus BYOK, or a small backend. | **Best control.** Same stack as LightRAG's WebUI. | Moderate. An AI assistant closes much of the gap. | GitHub Pages: 1 GB site limit, 100 GB/month soft bandwidth limit. | **High.** It shows full-stack ability. |
| **Vite + Svelte** | Yes. | Same. | Use sigma or Cytoscape imperatively. React has more graph wrappers. | Moderate | Same | High |
| **Plain HTML + TypeScript + sigma** | Yes. | Same. | Same engine. | Fast for a single view, then messy once you add panels, a stepper and a compare mode. | Same | Medium to high |
| **Observable Framework** | Yes. It builds a static site with data loaders that run at build time, and Python is allowed. | Precomputed only, unless BYOK. | Any JS library. | Moderate | Same | Medium. Activity is slowing (v1.13.4, 2026-03; last push 2026-05). |

**Hosting facts that shape this choice.**
- Anthropic's API accepts direct browser calls when the `anthropic-dangerous-direct-browser-access: true` header is set, which makes BYOK possible.
- `@huggingface/transformers` 4.3.0 (Transformers.js, 2026-09-16) runs small embedding models in the browser. That means **query embedding and graph retrieval can run entirely client-side**; only answer generation needs a key or a backend.
- Render's free web services spin down after 15 minutes idle and take about a minute to wake.

**Recommendation: Vite + React + `@react-sigma` + Tailwind, deployed to GitHub Pages.**
- The site ships `graph.json` (with precomputed coordinates), `chunks/*.json`, about 25 precomputed traces, and optionally an embedding matrix for live in-browser retrieval.
- **Size check (an estimate):** 10k entity embeddings × 384 dimensions × 4 bytes is about 15 MB as float32, or about 4 MB as int8. That is well within GitHub Pages limits.
- **Python stays where it belongs:** offline indexing, evaluation, and an optional `uv run` FastAPI server for local or live mode.
- If you would rather stay in Python for the MVP, build a Streamlit prototype first to tune retrieval, then port it. But do not ship Streamlit as the public demo: it sleeps, it cannot run on GitHub Pages, and its graph components are stale.

---

## 6. Effort estimate

The estimates assume one experienced Python developer working with an AI coding assistant, a hand-built pipeline, and the static React + sigma frontend. They are in focused hours. Most of the risk is in iterating on extraction quality and visual polish, not in the amount of code.

| Component | MVP (h) | Polished (h) | What MVP → polished adds |
|---|---|---|---|
| Ingestion (Gutenberg fetch, boilerplate strip, chapter split, chunking with offsets) | 3–5 | 6–8 | Multiple books, chapter metadata, tests |
| Extraction (Pydantic schema, prompt, structured output, batching, response cache, retries) | 6–10 | 14–20 | Prompt iteration against a hand-labelled sample of about 30 chunks, with measured extraction precision |
| Entity resolution (normalization rules, embedding plus string similarity, `aliases.yaml`) | 4–6 | 12–20 | LLM confirmation for ambiguous merges, a merge-audit report, the alias UI |
| Graph build (NetworkX, evidence-weighted edges, Leiden, offline ForceAtlas2 layout, JSON export) | 4–6 | 8–12 | Community summaries, a hierarchy, graph statistics page |
| Retrieval (seed linking, capped 2-hop expansion plus PPR, context budget, citation validation, trace output) | 8–12 | 16–24 | Path mode, Global mode, step and score detail, unit tests on small fixture graphs |
| Vanilla RAG baseline (top-k chunks, same LLM and prompt style) | 2–3 | 3–5 | Reranking, plus a closed-book baseline (no retrieval) |
| Evaluation set (25–40 hand-written questions with answers and gold passages; scoring script) | 6–10 | 16–24 | Question taxonomy (single-hop, multi-hop, global), LLM-as-judge calibrated against your own labels, error-analysis write-up, LightRAG as an outside baseline |
| API (FastAPI `/query` returning answer plus trace) | 0–3 (skip if fully precomputed) | 5–8 | Rate limiting, BYOK pass-through or proxy, OpenAPI docs |
| Visualization frontend (Vite, React, sigma; Concept A plus a simple stepper) | 14–22 | 35–50 | Full Concept B replay, Concept C compare plus timeline, mobile layout, accessibility, dark mode |
| Deployment (GitHub Actions to Pages; optional backend) | 2–3 | 5–8 | In-browser embeddings for live mode, budget caps, smoke test in CI |
| README, demo GIF, architecture diagram | 3–5 | 8–12 | Results table, cost and latency table, "where it breaks" section, a 2-minute video |
| **Total** | **≈ 52–85 h** (about 1.5–2 weeks full time, or 4–6 weekends) | **≈ 128–191 h** (about 4–5 weeks full time) | |

**Ways to shrink the MVP:**
- One book of about 100–150k words, instead of the full 300k–1M words.
- Local mode only.
- No communities.
- Precomputed traces only.

**If you use LightRAG instead of building it yourself:**
- It saves about 15–25 h across extraction and retrieval.
- It adds about 4–8 h of adapter work to map `aquery_data` and the GraphML output into the trace format.
- You lose step and score detail and dropped candidates, so Concept B shrinks to "here is what was used".

---

## 7. The interview angle: what makes a RAG portfolio project stand out in 2026

These are practitioner and blog sources, not hiring studies, so treat them as consensus opinion.

1. **Evaluation beats features.**
   - Hamel Husain argues that unsuccessful AI products "almost always share a common root cause: a failure to create robust evaluation systems". He advises starting with error analysis, meaning categorizing failures, before automating checks ([hamel.dev](https://hamel.dev/blog/posts/evals/)).
   - Dataquest (2026-07-04) suggests "an eval set of 20–30 real queries with expected answers, written by you, not generated", and a README section titled "Where this breaks and why" ([dataquest.io](https://www.dataquest.io/blog/ai-projects/)).
   - Sundeep Teki (2026-06-18): "A RAG app you have measured is what signals seniority" ([sundeepteki.org](https://www.sundeepteki.org/advice/the-4-portfolio-projects-every-ai-forward-deployed-engineer-should-build-in-2026)).
2. **Discuss honestly when the graph helps.**
   - GraphRAG-Bench (arXiv 2506.05690): "GraphRAG frequently underperforms vanilla RAG on many real-world tasks."
   - A systematic evaluation (arXiv 2502.11371) finds RAG and GraphRAG have "distinct strengths" depending on the task.
   - A side-by-side comparison with a "graph lost here" section is far more credible than claiming the graph is better.
3. **Cost and latency numbers.** Dataquest recommends listing token counts, latency and "approximate dollars per 1,000 requests". Put indexing cost, cost per query and p50/p95 latency for each mode in the README.
4. **A README that works within seconds.**
   - letsdatascience says hiring managers spend about 40 seconds on a first pass, so the problem, approach and results should fit in the first 200 words.
   - A live demo "signals a completely different level of seriousness".
   - Include RAGAS-style metrics with real numbers ([letsdatascience.com](https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026)).
   - Put a GIF of the retrieval replay at the very top.
5. **Tests and clean commit history.** The same source says adding "even two tests" signals engineering discipline. Unit-test entity resolution and expansion on small fixture graphs; they are deterministic and fast.
6. **Show that you own the internals.** Interviewers probe the following, so each should have a short design note:
   - the extraction schema
   - how entity resolution decides a merge
   - why PPR was used instead of plain BFS (breadth-first expansion)
   - how citations are validated
   - what happens when the context budget is exceeded

   "I called LightRAG" invites the question "so what did you build?" Building the core yourself and using LightRAG as a comparison answers it in advance.
7. **An architecture diagram** in the README, using the Mermaid diagram already drafted in 02-architecture.md.
8. **Avoid the tutorial look.** "One generic RAG demo, lifted from a course, unevaluated" is recognizable "in about ten seconds" (Dataquest, as surfaced in search; wording unverified). This is another reason to avoid *A Christmas Carol* and to lead with the replay and the evaluation rather than the 3D graph.

---

## 8. Open decisions for the user

1. **Build it yourself, or use LightRAG?** Recommended: build it, with LightRAG as an outside baseline. If you are time-boxed to about 1 week, use LightRAG and accept a lighter Concept B.
2. **Demo interactivity:**
   - (a) precomputed traces only, which is zero cost and always on
   - (b) plus in-browser retrieval with BYOK answers
   - (c) plus a hosted backend proxy, which brings a cost-abuse risk and needs a budget cap

   Recommended: (a) for the MVP, then (b).
3. **Corpus:** one book for the MVP, or a multi-book "universe" (shared characters across books make the graph more interesting). Avoid *A Christmas Carol*. Entity density matters more than word count.
4. **Frontend framework:** React (more graph wrappers, and AI assistants are fluent in it), Svelte (leaner), or plain TypeScript (fastest single view).
5. **Visualization library:** Sigma.js 3.0.x (recommended) or Cytoscape.js (easier animation, smaller graphs). Also, will you upgrade to sigma v4 once it is stable?
6. **Default view:** an ego subgraph (recommended for readability and mobile) or the full graph.
7. **Communities and Global mode:** include them in the MVP, or defer them to v2 as 02-architecture.md suggests?
8. **LLM providers and budget:** which model extracts and which answers, and a hard spending cap for indexing iterations (02-architecture.md suggests $10–30 in total).
9. **Book-specific extras:** a chapter timeline, alias display, spoiler-bounded querying. Pick at most one for the MVP.

---

## Sources

All checked on 2026-10-05 unless noted.

**Frameworks**
- Microsoft GraphRAG repo, README (maintenance-mode notice), `api/query.py`, `local_search/mixed_context.py`, `local_search_system_prompt.py`, `extract_graph.py`: https://github.com/microsoft/graphrag
- GraphRAG docs, indexing methods (Standard vs. FastGraphRAG, the 75% figure): https://github.com/microsoft/graphrag/blob/main/docs/index/methods.md
- GraphRAG docs, models (LiteLLM, Ollama caveat): https://github.com/microsoft/graphrag/blob/main/docs/config/models.md
- GraphRAG docs, bring your own graph: https://github.com/microsoft/graphrag/blob/main/docs/index/byog.md
- GraphRAG get-started guide (*A Christmas Carol*): https://github.com/microsoft/graphrag/blob/main/docs/get_started.md
- GraphRAG unified-search-app README: https://github.com/microsoft/graphrag/tree/main/unified-search-app
- LazyGraphRAG discussion #1490: https://github.com/microsoft/graphrag/discussions/1490
- PyPI `graphrag`: https://pypi.org/project/graphrag/
- LightRAG repo, `lightrag/lightrag.py` (`aquery_data`, `aquery_llm`), `lightrag/llm/*`, `lightrag/kg/*`, WebUI source and `package.json`: https://github.com/HKUDS/LightRAG
- PyPI `lightrag-hku`: https://pypi.org/project/lightrag-hku/
- nano-graphrag `_op.py`: https://github.com/gusye1234/nano-graphrag
- fast-graphrag `_types.py`: https://github.com/circlemind-ai/fast-graphrag
- LlamaIndex PropertyGraphIndex guide: https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/
- langchain-experimental sunset, issue #87: https://github.com/langchain-ai/langchain-experimental/issues/87
- langchain-neo4j CHANGELOG (LLMGraphTransformer migration): https://github.com/langchain-ai/langchain-neo4j
- neo4j-graphrag-python, `generation/graphrag.py`, `llm/*`: https://github.com/neo4j/neo4j-graphrag-python
- Graphiti README and `edges.py`/`graphiti.py`: https://github.com/getzep/graphiti
- Cognee README: https://github.com/topoteretes/cognee
- Cognee recall docs: https://docs.cognee.ai/core-concepts/main-operations/recall
- HippoRAG `utils/misc_utils.py` (`QuerySolution`): https://github.com/OSU-NLP-Group/HippoRAG
- Kuzu (archived): https://github.com/kuzudb/kuzu
- Other repos: KAG https://github.com/OpenSPG/KAG · Youtu-GraphRAG https://github.com/TencentCloudADP/youtu-graphrag · LinearRAG https://github.com/DEEP-PolyU/LinearRAG · RAG-Anything https://github.com/HKUDS/RAG-Anything · GraphRAG-SDK https://github.com/FalkorDB/GraphRAG-SDK · graphrag-toolkit https://github.com/awslabs/graphrag-toolkit · kg-gen https://github.com/stair-lab/kg-gen
- NetworkX community algorithms (3.7): https://networkx.org/documentation/stable/reference/algorithms/community.html
- NetworkX `forceatlas2_layout`: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.forceatlas2_layout.html
- Neo4j Aura Free limits (secondary summary of Neo4j pages): https://neo4j.com/free-graph-database/ · https://support.neo4j.com/s/article/16094506528787-Support-resources-and-FAQ-for-Aura-Free-Tier
- Batch API discounts (secondary): https://www.finout.io/blog/anthropic-api-pricing · https://community.openai.com/t/batch-api-is-now-available/718416
- GraphRAG cost for *A Christmas Carol* (secondary, unverified): https://learnopencv.com/lightrag/

**Visualization**
- Sigma.js site: https://www.sigmajs.org/
- Sigma.js repo and npm `sigma`: https://github.com/jacomyal/sigma.js
- Cytoscape.js WebGL preview benchmarks: https://blog.js.cytoscape.org/2025/01/13/webgl-preview/
- Cytoscape.js 3.31 release: https://blog.js.cytoscape.org/2025/01/13/3.31.0-release/
- Memgraph graph visualization benchmark (2022): https://memgraph.com/blog/you-want-a-fast-easy-to-use-and-popular-graph-visualization-tool
- "Graph visualization efficiency of popular web-based libraries" (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12061801/
- vis-network: https://github.com/visjs/vis-network
- react-force-graph README: https://github.com/vasturiano/react-force-graph
- AntV G6 (hull and bubble-sets plugins): https://github.com/antvis/G6
- Reagraph: https://github.com/reaviz/reagraph
- d3-force: https://github.com/d3/d3-force
- cosmos.gl: https://github.com/cosmosgl/graph
- Neo4j NVL license: https://unpkg.com/@neo4j-nvl/base@2.0.0/LICENSE.txt
- pyvis: https://github.com/WestHealth/pyvis
- streamlit-agraph: https://github.com/ChrisDelClea/streamlit-agraph
- st-link-analysis: https://github.com/AlrasheedA/st-link-analysis
- Gephi Lite (uses sigma 3): https://github.com/gephi/gephi-lite

**Prior art**
- graphrag-visualizer: https://github.com/noworneverev/graphrag-visualizer
- graphrag-workbench: https://github.com/lyon-industries/graphrag-workbench
- Neo4j LLM Graph Builder: https://github.com/neo4j-labs/llm-graph-builder · https://neo4j.com/blog/developer/llm-knowledge-graph-builder-release/
- Kotaemon: https://github.com/Cinnamon/kotaemon
- LovichiDorian/GraphRag: https://github.com/LovichiDorian/GraphRag
- Minimal-GraphRAG: https://github.com/HuzaifaAqeel/Minimal-GraphRAG
- graphsight: https://github.com/Kcodess2807/graphsight
- novel-graph: https://github.com/schoobani/novel-graph
- Memgraph ai-demos: https://github.com/memgraph/ai-demos

**Frontend and hosting**
- GitHub Pages limits: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- Hugging Face Spaces overview (paid plan for Gradio/Docker; free tier hardware): https://huggingface.co/docs/hub/spaces-overview
- Streamlit Community Cloud (12 h sleep, memory): https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- Render free tier: https://render.com/docs/free
- Anthropic browser access (CORS header): https://simonwillison.net/2024/Aug/23/anthropic-dangerous-direct-browser-access/
- Transformers.js: https://github.com/huggingface/transformers.js
- stlite: https://github.com/whitphx/stlite
- NiceGUI: https://github.com/zauberzeug/nicegui
- Observable Framework: https://github.com/observablehq/framework

**Interview angle and evaluation literature**
- Hamel Husain, "Your AI Product Needs Evals": https://hamel.dev/blog/posts/evals/
- Dataquest, "Best AI Projects to Build in 2026": https://www.dataquest.io/blog/ai-projects/
- letsdatascience, "The ML Portfolio That Actually Gets You Hired in 2026": https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026
- Sundeep Teki, "4 Portfolio Projects Every AI FDE Should Build in 2026": https://www.sundeepteki.org/advice/the-4-portfolio-projects-every-ai-forward-deployed-engineer-should-build-in-2026
- GraphRAG-Bench, "When to use Graphs in RAG": https://arxiv.org/abs/2506.05690
- GraphRAG-Bench repo: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- "RAG vs. GraphRAG: A Systematic Evaluation and Key Insights": https://arxiv.org/abs/2502.11371
