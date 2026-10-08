# 01 — Book-set candidates for the Graph RAG demo

Discovery notes, checked 2026-10-05. Nothing is decided yet: the user picks the corpus.

**How to read this.** Word and token counts are **measured**. I downloaded each Gutenberg plain-text file, removed the Gutenberg header and footer, counted words by splitting on whitespace, and counted tokens with `tiktoken` `o200k_base` (the GPT-4o tokenizer). Claude's tokenizer gives somewhat different counts; I did not measure by how much. The cross-book mention counts are also measured: I counted how often each name appears in each book. **(unverified)** marks anything I could not confirm from a source or from the text itself.

---

## TL;DR

| Rank | Set | One-line reason |
|---|---|---|
| **1** | **Trojan War cycle**: *Iliad* + *Odyssey* (Samuel Butler's prose translations) + *Aeneid* (J. W. Mackail's prose translation) | Every interviewer knows Achilles, Helen, Odysseus and the Trojan Horse. The books share one story world but come from different authors. The gods form ready-made factions, the heroes have explicit family trees, and the set is a moderate ~535k tokens. Nolan's *The Odyssey* (July 2026) makes it topical. |
| **2** | **Plutarch, Lives of the late Roman Republic**: 11 biographies, Dryden translation revised by Clough | It has the strongest cross-document overlap of any candidate: Caesar is named in 9 of the 11 Lives and Pompey in 10. It is real history, so it reads as a professional, "enterprise-like" corpus, and I found no Graph RAG demo that uses it. The risks are name ambiguity and an older prose style. |
| **3** | **Howard Pyle's four King Arthur books** | Recognizable core names, clear factions (Gawaine's Orkney clan against Launcelot's kin), kinship chains and quests that run across all 4 volumes. The archaic "thee/thou" prose is the main cost. |

- **Runners-up.** **Oz (Baum, 14 books)** is the safest and easiest to build, but readers only know book 1. **Sherlock Holmes** is the best-known set of all, but it is already a Graph RAG demo corpus and contains a few dated portrayals, one of them a religious group cast as villains.
- **Key tradeoff: recognizability vs. graph value vs. extraction difficulty.** The sets that make the best graphs (Plutarch, Shakespeare's histories) have the hardest names to resolve. The sets that are easiest to extract (Sherlock, Austen) have little entity overlap between books.
- **Cost does not decide this.** A single extraction pass over any top-3 set costs **under $1 with GPT-4o-mini and under $5 with Claude Haiku 4.5** (assumptions in §2). Even 10–20 re-indexing runs while tuning prompts stay under about $100.
- **Recommendation on "a set of different books":** use a **shared-universe** set, meaning books whose characters and places recur (§5). A set of unrelated books turns into N disconnected islands. That throws away the multi-hop, cross-book paths that make "show the activated graph elements" look impressive.

---

## 1. Scoring scale (definitions)

Each criterion is scored 1–5, where 5 is best.

- **Entity density**: how many named characters, places, groups and events there are, and how richly they are linked (family, alliances, rivalries).
- **Cross-book connectivity**: how much the same entities recur across books. 5 means the main cast appears in nearly every book. 1 means no shared characters at all.
- **Recognizability**: whether an interviewer who has not read the books can follow a demo query and judge whether the answer is plausible.
- **Graph RAG advantage**: how naturally the text produces questions that plain vector RAG answers poorly. These come in three kinds:
  - **multi-hop**: the answer needs a chain of facts spread over passages that never appear together;
  - **global or aggregation**: "what are the factions", "summarize across all books";
  - **timeline**: events ordered across books.
- **Size fit**: 5 = 350k–700k tokens, big enough to look substantial (~300–650 chunks) and small enough to re-index in minutes. 4 = 250–350k or 700k–1M. 3 = under 250k or 1–1.5M. 2 = over 1.5M.
- **Distinctiveness**: 5 = I found no Graph RAG or knowledge-graph demo using it. 1 = a default tutorial corpus.
- **Neutrality**: 5 = nothing likely to be awkward in an interview. Lower scores mean dated racial or religious portrayals, or content you would not want on screen.
- **Extraction ease** (extra column, added because the user asked for "not overcomplicated"): how cleanly an LLM can pull entities out of the text. It covers prose vs. verse, archaic diction, one name used for several people, and markup noise.

---

## 2. Scored comparison table

| # | Set (book count) | Words | Tokens (o200k) | Density | Cross-book | Recogn. | GraphRAG adv. | Size fit | Distinct. | Neutral. | Ease | **Total /40** | 1-pass cost: 4o-mini / Haiku 4.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Trojan War cycle (3) | 381,651 | 534,661 | 5 | 4 | 5 | 5 | 5 | 4 | 4 | 4 | **36** | $0.46 / $3.43 |
| 2 | Plutarch, late-Republic Lives (11 Lives, 1 file) | 224,644 | 308,993 | 5 | 5 | 4 | 5 | 4 | 5 | 5 | 2 | **35** | $0.26 / $1.98 |
| 3 | Pyle's King Arthur (4) | 474,530 | 637,322 | 5 | 5 | 4 | 4 | 5 | 4 | 4 | 3 | **34** | $0.54 / $4.09 |
| 4 | Oz series, Baum (14; first 6 in brackets) | 626,337 (255,281) | 848,356 (346,974) | 3 | 5 | 3 | 4 | 4 | 3 | 5 | 5 | **32** | $0.72 / $5.44 ($0.30 / $2.23) |
| 5 | Shakespeare histories, *Richard II* to *Richard III* (8) | 211,963 | 312,444 | 5 | 5 | 3 | 5 | 4 | 5 | 3 | 1 | **31** | $0.27 / $2.01 |
| 6 | Sherlock Holmes canon (9) | 657,553 | 903,220 | 3 | 3 | 5 | 4 | 4 | 2 | 3 | 5 | **29** | $0.77 / $5.80 |
| 7 | Jane Austen novels (6) | 724,049 | 967,961 | 4 | 1 | 4 | 3 | 4 | 2 | 5 | 5 | **28** | $0.82 / $6.20 |
| 8 | Gothic classics mix, an unrelated-books set (6) | 411,353 | 547,452 | 3 | 1 | 5 | 3 | 5 | 3 | 3 | 4 | **27** | $0.47 / $3.51 |

**Rationale for the lower rows, briefly.**
- **Oz.** Dorothy appears in 14 of 14 books and Ozma in 13. But most people know only book 1, and that book is already a Graph RAG benchmark text [S9][S10].
- **Shakespeare.** It is the best "faction graph" here, being the Wars of the Roses, often called the real-life *Game of Thrones*. It is also verse with speaker tags, and the same title names different people across the plays: "Gloucester", "York" and "Somerset" each refer to several men.
- **Sherlock.** Each case has its own cast, so the graph is hub-and-spoke: Holmes and Watson sit in the middle, with ~60 separate case clusters around them. Moriarty appears meaningfully in only 3 of the 9 books.
- **Austen.** No character appears in more than one novel.
- **Gothic mix.** The books share only generic places such as London.

**Cost assumptions** for the "1-pass" column, meaning a single LLM extraction pass with no gleaning, summarization or community reports:
- **Chunks.** 1,200-token chunks with 100-token overlap (Microsoft GraphRAG's defaults [S4]). That gives `tokens / 1,100` chunks.
- **Input per chunk.** 1,200 chunk tokens plus 1,850 prompt tokens. 1,850 is the measured size of Microsoft GraphRAG's `extract_graph.py` prompt file [S5].
- **Output per chunk.** 800 tokens, which is **my assumption**.
- **Prices**, standard tier, per 1M input / output tokens:
  - GPT-4o-mini: $0.15 / $0.60 [S7]
  - Claude Haiku 4.5: $1 / $5 [S6]
  - GPT-5-nano: $0.05 / $0.40 [S7]
  - GPT-5-mini: $0.25 / $2.00 [S7]
- **Discounts.** Batch APIs halve every figure [S6][S7].
- **Trojan set on the other models.** $0.23 with GPT-5-nano and $1.15 with GPT-5-mini.
- **Smaller chunks cost more.** `02-architecture.md` proposes ~600-token chunks. That roughly doubles the number of calls, so expect about 1.5–1.8× these figures (estimate).
- **A full GraphRAG pipeline costs far more than one pass.** Microsoft's cost write-up indexed *The Wonderful Wizard of Oz*, which I measured at ~51k tokens. It reports **4,058,128 tokens consumed** [S9], about 80× the size of the corpus. The article does not state its chunking settings, so treat this as a rough upper bound. At 80× (assuming an 85/15 input/output split), the Trojan set would cost about $9 with 4o-mini and about $68 with Haiku 4.5 per full run.

---

## 3. Top-3 detail

### 3.1 #1 — Trojan War cycle (Iliad + Odyssey + Aeneid)

**Book list**

| Book | Gutenberg ID | Translation | Words / tokens |
|---|---|---|---|
| Homer, *The Iliad* | **2199** | Samuel Butler, prose, 1898 | 152,757 / 218,835 |
| Homer, *The Odyssey* | **1727** | Samuel Butler, prose, 1900. The Gutenberg file includes a December 1921 preface by Henry Festing Jones (died 1928) | 129,571 / 173,132 |
| Virgil, *The Aeneid* | **22456** | J. W. Mackail, prose, 1885 (the date is in the file) | 99,323 / 142,694 |
| *Optional:* Apollonius, *Argonautica* | 830 | R. C. Seaton, 1912 | 61,281 / 84,105 |

- **Public-domain status.**
  - Butler died in 1902, so his translations are public domain worldwide.
  - Mackail died in 1945, so his is public domain in the US (published 1885) and in life+70 and life+80 countries.
  - Seaton's translation (1912) is public domain in the US. I did not verify his death year (unverified).
- **Why Butler.** His translations are plain modern prose. I also checked the alternatives:
  - The Lang-Leaf-Myers *Iliad* (Gutenberg 3059) spells names unfamiliarly: "Aineias" 79×, "Aias" 154×, "Patroklos" 151×, "Menelaos" 128×.
  - The verse translations (Pope 6130, Cowper 16452, Derby 6150) are harder to extract from.

**Why it works**
- **It is a shared universe with different authors.** In my counts:
  - Menelaus (66), Nestor (40), Agamemnon (38) and Helen (26) all recur in the *Odyssey*.
  - Aeneas appears 86 times in the *Iliad* before becoming the *Aeneid*'s hero.
  - Priam (44), Achilles (23) and Hector (22) recur in the *Aeneid*.

  This set fits **both** readings of "a set of different books" (§5).
- **It has a clear faction structure.** The gods take sides in the war. That gives Graph RAG's community detection (automatic clustering of tightly linked entities) something obvious and easy to check.
- **It has explicit family trees in the text.** In *Iliad* Book XX, Aeneas recites his descent ("Dardanus was the son of Jove") to Achilles, and Neptune then rescues him. Both passages are verified in 2199.
- **It has research precedent to cite.** A network-science paper analysed the *Iliad* as a social network of 716 characters (Mac Carron & Kenna, 2012 [S16][S17]). That is a good interview talking point.
- **It is topical.** Nolan's *The Odyssey* opened in the US on 2026-07-17; Wikipedia lists its box office at $1.768 billion [S18].

**Risks**
- **Butler and Mackail use Roman names.** In 2199: "Jove" 475×, "Minerva" 162×, "Ulysses" 128×, against "Zeus" 1× and "Odysseus" 0×.
  - The good news: the three books are consistent with each other, so they need no cross-translation alias merging.
  - The bad news: an interviewer will type "Odysseus" or "Zeus". You need a small alias table (Ulysses = Odysseus, Jove = Zeus, Minerva = Athena) and should show it off as an entity-resolution feature.
  - Adding the Seaton *Argonautica* brings Greek names (Zeus 92×, Hera 41×), so the alias work becomes mandatory.
- **Some names and epithets are ambiguous.**
  - "Pallas" means Minerva *and* Pallas, son of Evander, in the *Aeneid* (53 hits in total).
  - Achilles is often called the "descendant of Aeacus" (19 "Aeacus" hits in the *Iliad*).
  - Paris also goes by "Alexandrus": 49 hits for "Alexandrus" vs. 23 for "Paris" in 2199.
- **The text includes translator noise.** Butler's prefaces and footnotes should be stripped.
- **Some content is grim.** There is graphic battle violence, and the central quarrel is over a captive woman (Briseis). The Odyssey's Book 22 includes the hanging of the maidservants. All of this is standard school reading, but keep the demo queries away from it.
- **LLMs have memorized these stories.** A closed-book LLM answers many questions about Homer correctly. Prefer queries whose answers depend on the text (see query 3).

**Sample demo queries**

| # | Query | Graph feature exercised | Why vanilla RAG struggles |
|---|---|---|---|
| 1 | "How is Aeneas connected to Achilles?" | **Path search**: duel in Book XX → Neptune rescues Aeneas → Aeneas's descent from Jove via Dardanus; Aeneas → *Aeneid* | The links sit in different passages and books. Top-k chunks for "Aeneas Achilles" return only the duel. |
| 2 | "Which gods backed the Greeks and which the Trojans? Does Juno's hostility carry into the *Aeneid*?" | **Community detection plus a global answer across two books** | This is an aggregation over hundreds of scattered mentions. No single chunk lists the sides. |
| 3 | "Where does the wooden horse appear, and who was inside it?" | **Entity provenance across books** (the source passages behind a single entity), plus **negative evidence** | Verified: the *Odyssey* has it ("Epeus made with the assistance of Minerva… Ulysses got [it] into the fort of Troy"). The *Aeneid* has it ("by Pallas' divine craft"; Sinon frees the Greeks). The *Iliad* never mentions it (0 hits). This is a strong check of grounding against model memory. |
| 4 | "What happened to Agamemnon after the war, and who avenged him?" | **Event chain / timeline**: *Iliad* (commander) → *Odyssey* (Aegisthus kills him; Orestes kills Aegisthus) | The answer crosses books, and the *Odyssey* tells it in fragments across several books. |
| 5 | "Who are the most connected characters across all three epics, and who bridges the Greek and Trojan sides?" | **Graph analytics**: degree and betweenness centrality | This is a structural question, so vector search has nothing to retrieve. |

### 3.2 #2 — Plutarch's late Roman Republic Lives

**Book list.** All 11 Lives are in one Gutenberg file, **674** (*Plutarch: Lives of the Noble Grecians and Romans*, Dryden translation, edited by Arthur Hugh Clough). Each Life starts on an all-caps heading line, so splitting the file by Life is a short script (tested). Measured sizes:

| Life | Words | Life | Words |
|---|---|---|---|
| Caius Marius | 17,997 | Caesar | 24,082 |
| Sylla (Sulla) | 16,060 | Cato the Younger | 24,699 |
| Crassus | 14,428 | Cicero | 16,915 |
| Sertorius | 11,067 | Antony | 27,915 |
| Lucullus | 18,179 | Marcus Brutus | 19,496 |
| Pompey | 33,806 | **Total** | **224,644 words / 308,993 tokens** |

- **Optional additions.**
  - *Tiberius* and *Caius Gracchus* bring the total to 240,939 words.
  - Shakespeare's *Julius Caesar* (Gutenberg **1522**) and *Antony and Cleopatra* (**1534**), which Shakespeare based on Plutarch, would add a cross-genre bridge.
- **Alternative translation.** George Long and Aubrey Stewart's 4-volume translation (IDs 14033, 14114, 14140, 44315) is 1880s prose. I did not measure it.
- **Public-domain status.** Dryden died in 1700 and Clough in 1861, so the text is public domain worldwide.

**Why it works**
- **Each Life is a separate document about overlapping people and events.** This is exactly the cross-document entity-merging problem that enterprise Graph RAG exists to solve, so it is a strong interview story. Measured mentions:

  | Life | Caesar | Pompey | Cicero | Other notable |
  |---|---|---|---|---|
  | Pompey | 109 | 397 | | Cato 37 |
  | Cato the Younger | 90 | 100 | | |
  | Antony | 187 | | | Cleopatra 85 |
  | Cicero | | | 190 | Catiline 27, Clodius 31, Antony 29 |
  | Sylla | | | | Marius 41 |

- **The same events are told from several sides.** The Ides of March, Pharsalus and the Catiline conspiracy each appear in multiple Lives. That suits "compare the accounts" global queries.
- **Many names are familiar.** Caesar, Pompey, Cicero, Brutus, Antony, Cleopatra and Spartacus (16 mentions, mostly in *Crassus*) are well known from Shakespeare, HBO's *Rome* and school.
- **It is very distinctive.** I found no Graph RAG demo built on Plutarch.
- **It is neutral.** The violence is ordinary ancient history.

**Risks**
- **This is the hardest set for entity resolution.**
  - In *Antony*, "Caesar" (187×) mostly means **Octavian**, and "Octavius" appears only once.
  - "Cato" can mean the Elder or the Younger; "Brutus" can mean Marcus or Decimus; "Marius" can mean the father or the son.
  - Naive name-based merging will fuse these people. This is either your best interview story ("here's how I handled Caesar vs. Octavian") or a quality trap.
- **The prose is 17th-century in style**, with long sentences and Latinized spellings ("Sylla", "Caius"). Source passages shown in the UI are less pleasant to read.
- **The corpus is only ~309k tokens.** That is enough, but it is the smallest of the top 3.
- **It is one file, not "books".** You have to present the Lives as separate documents.

**Sample demo queries**

| # | Query | Graph feature exercised |
|---|---|---|
| 1 | "How did the rivalry between Marius and Sylla shape Pompey's rise?" | **Multi-hop across documents**: the *Marius*, *Sylla* and *Pompey* Lives |
| 2 | "Compare the accounts of Caesar's assassination in the Lives of Caesar, Brutus, Antony and Cicero." | **One event with sources from several documents, plus a global comparison** |
| 3 | "How were Caesar and Pompey tied by marriage and politics, and how did the tie break?" | **Typed relationship subgraph plus time ordering**. Verified passage: Caesar had "a daughter by Cornelia… whom he afterwards married to Pompey the Great" |
| 4 | "Who were Cicero's enemies, and what became of each?" | **Neighborhood expansion (pulling in an entity's direct links) plus aggregation**: Catiline, Clodius, Antony |
| 5 | "What were the main factions of the late Republic, and who belonged to each?" | **Community detection / global search**. Plutarch himself writes "There being two factions in the city, one that of Sylla…" (verified) |

### 3.3 #3 — Howard Pyle's King Arthur books

**Book list.** Pyle died in 1911; the books were published 1903–1910, so they are public domain worldwide.

| Book | Gutenberg ID | Words / tokens |
|---|---|---|
| *The Story of King Arthur and His Knights* | **60184** | 115,509 / 154,444 |
| *The Story of the Champions of the Round Table* | **10745** | 125,742 / 169,360 |
| *The Story of Sir Launcelot and His Companions* | **33702** | 135,196 / 180,755 |
| *The Story of the Grail and the Passing of Arthur* | **60405** | 98,083 / 132,763 |

The alternative is Malory's *Le Morte d'Arthur* (Gutenberg 1251 and 1252; 362,068 words / 496,222 tokens). It is more complete, but its prose is much more archaic.

**Why it works**
- **It is a true series universe.** Measured mentions per volume:

  | Character | Vol 1 | Vol 2 | Vol 3 | Vol 4 |
  |---|---|---|---|---|
  | Arthur | 798 | 262 | 193 | 369 |
  | Launcelot | 5 | 766 | 1,016 | 636 |
  | Gawaine | 383 | 32 | 259 | 276 |

  Mordred becomes central in volume 4 (129 mentions), and Galahad there too (277).
- **It has dense kinship and faction structure:**
  - the Orkney brothers against Launcelot's kin;
  - "Galahad was the true son of Sir Launcelot of the Lake";
  - Bors as Launcelot's "kinsman" (both verified in 60405).
- **The famous core is recognizable:** Arthur, Merlin, Launcelot, Guinevere, Excalibur, Camelot and the Grail.
- **It is distinctive.** I found no Graph RAG demo built on Arthurian texts.

**Risks**
- **The prose is pseudo-archaic.** It is full of "quoth", "thee" and "Messire", plus `[Sidenote: …]` markers that must be stripped.
- **Name variants and a large cast.** Many knights have "Sir X of Y" names, and spellings vary (Morgana le Fay / la Fay).
- **Pyle contradicts himself.** Volume 1 says "Margaise was wedded to King Urien of Gore, and Morgana le Fay was wedded to King Lot", then later calls Ewaine "the son of King Uriens and of Queen Morgana la Fay". The graph will show conflicting edges. That is either a contradiction-detection demo or a data-quality headache.
- **Pyle differs from the popular legend.** He calls Mordred Arthur's "nephew" (verified), not his son. That makes a good grounding demo, but an interviewer may think the answer is wrong.
- **The Grail quest is Christian-themed.** It is literature, not scripture, so the risk is mild.

**Sample demo queries**

| # | Query | Graph feature exercised |
|---|---|---|
| 1 | "How is Mordred related to Arthur in this telling?" | **A kinship edge with its source passage**, and grounding against model memory (Pyle says nephew; the popular legend says son) |
| 2 | "What chain of events led from Launcelot and Guinevere to the fall of the Round Table?" | **Causal chain across volumes 3–4** (Mordred and Agravaine accuse Launcelot of treason, and so on) |
| 3 | "Which knights are kin to Launcelot, and what part did they play in the Grail quest?" | **Kinship subgraph plus event participation** (Galahad, Bors) |
| 4 | "What are the main factions among the knights, and where do they clash?" | **Community detection / global search** |
| 5 | "Trace Excalibur from its first appearance to its last." | **One object's timeline across books**: volume 1 (68 mentions) to volume 4 ("how Excalibur was cast away", Bedivere) |

### 3.4 Runners-up in one paragraph each

- **Oz** (Gutenberg 55, 54, 486, 420, 485, 517, 955, 956, 957, 958, 959, 960, 419, 961; all public domain worldwide, since Baum died in 1919; the US status is confirmed in [S19]).
  - **Strengths:** the easiest prose, the most neutral content, and real series continuity. Tip turns out to be Ozma, and the Nome King returns across several books.
  - **A neat grounding demo:** the shoes are *silver* in the book and ruby only in the 1939 MGM film, which is still under copyright [S19].
  - **Why it is not top 3:** interviewers know only book 1, and that book is already a Graph RAG benchmark [S9][S10].
- **Sherlock Holmes** (Gutenberg 244, 2097, 1661, 834, 2852, 108, 3289, 2350, 69700; all public domain in the US since 2023-01-01 [S11]; Doyle died in 1930, so also public domain in life+70 countries).
  - **Strengths:** the best-known set and very clean prose. "Communities = cases" makes a readable global demo.
  - **Weaknesses:**
    - Cross-book links run mostly through the Holmes/Watson hub.
    - It is already used for the same purpose: a public repo uses the full 60-story canon as its knowledge-graph corpus [S12], and there are DZone and Neo4j talks [S13][S14].
    - *A Study in Scarlet* casts Mormons (the Latter-day Saints, a real religious group) as villains [S15], and *The Sign of the Four* has a racist portrayal of Tonga [S15b]. That cuts against the user's neutrality goal.

---

## 4. Also considered and set aside

| Candidate | Size (measured unless noted) | Why set aside |
|---|---|---|
| Dumas, d'Artagnan novels (1257, 1259; all six: + 2609, 2681, 2710, 2759) | 2 books: 471k words / 676k tokens. 6 books: 1.19M words / 1.71M tokens | Strong cast and factions with real historical figures, but only *The Three Musketeers* is widely known, and all six books are too big. Anonymous 19th-century translations (public domain). A good fallback. |
| Montgomery, *Anne of Green Gables* series (45, 47, 51, 544, 5343, 3796) | 534k words / 733k tokens | Neutral, with a recurring cast, but a low-conflict, domestic graph. Later books (*Windy Poplars* 1936, *Ingleside* 1939) are not yet US public domain. |
| Burroughs, Barsoom (62, 64, 68, 72, 1153) | 343k words / 450k tokens | Rich factions, but the hero is a Confederate veteran, and the color-coded Martian races carry racialized traits (my reading; not source-checked). Burroughs, Inc. holds trademarks (unverified). |
| Verne, Nemo trilogy (2083, 164, 1268) | 454k words / 625k tokens | A genuine linked set: Captain Nemo links *Twenty Thousand Leagues* to *The Mysterious Island*, and Ayrton links *In Search of the Castaways* to *The Mysterious Island*. But the Victorian translations are poor and the link is little known. |
| Conan Doyle, Professor Challenger (139, 126, 68903) | 182k words / 250k tokens | Too small. *The Lost World* has dated racial depictions (unverified), and *The Land of Mist* promotes spiritualism. |
| Lovecraft (68283, 70652, 50133, 73181, …) | not measured | Mostly US public domain; no copyright renewals have been found for the post-1929 works [S20]. Well-known racism makes it a poor interview choice. |
| *Arabian Nights* (Lang, 128) | 112k words / 146k tokens | Small, with nested unrelated tales and Orientalism concerns. |
| Grimm (2591) / Andersen | Grimm: 101k words / 132k tokens | The generic entities ("the king", "the princess") break entity resolution, and the tales do not connect. |
| *War and Peace* (2600, Maude) / *Les Misérables* (135, Hapgood) | 563k / 566k words | Single books, not a set. The *Les Misérables* character co-occurrence graph is already a classic network-science dataset (Knuth's Stanford GraphBase; not re-verified here). |
| Suetonius, *Twelve Caesars* (6400, Thomson) | 233k words / 342k tokens | Great family-tree graph, but it has explicit sexual content (Tiberius, Caligula) you would not want on an interview screen. |
| Bulfinch, *Age of Fable* (3327) | 146k words / 209k tokens | A retelling and compilation, not primary books. Useful only as an optional "glue" text for the Trojan set. |
| Shakespeare histories | (in the table) | Kept in the table as #5; too complex for "not overcomplicated". |
| Dickens, *A Christmas Carol* | — | It is **the** default Graph RAG tutorial text: Microsoft GraphRAG [S1], LightRAG [S2], fast-graphrag [S3]. Avoid it. |
| Wodehouse (Jeeves), early Christie | — | Public domain in the US only: Wodehouse died in 1975 and Christie in 1976, so both are under copyright in life+70 countries. A public repo would be exposed. |

---

## 5. Shared universe vs. unrelated books

The phrase "a set of different books" can mean two things.

**(a) Shared universe**: the same characters, places and events recur across books (a series, a cycle, or overlapping histories).
- **What the graph looks like.** One large connected component. Entities are merged across books, so a single node such as "Aeneas" carries source passages from the *Iliad* and the *Aeneid*. Communities cross book boundaries; for example, a "Greek camp" community draws on both Homeric books.
- **What the demo shows.** Multi-hop paths whose hops come from different books. This is the clearest case where Graph RAG beats vector RAG: the facts never sit in the same chunk, so top-k similarity search does not retrieve them together. The UI can colour each activated node or edge by its source book, so viewers literally see the answer stitched across books.
- **Cost.** Entity resolution becomes the core engineering problem (Caesar vs. Octavian, Pallas vs. Pallas, Ulysses vs. Odysseus), and mistakes are visible on screen. It is also the most interview-worthy problem to talk about.

**(b) Unrelated books across genres** (for example, *Frankenstein*, *Dracula*, *Jekyll and Hyde*, *Dorian Gray*, *Carmilla* and *The Turn of the Screw*: Gutenberg 84, 345, 43, 174, 10007, 209).
- **What the graph looks like.** N islands. The only bridges are generic nodes like "London" or "England", which many pipelines filter out as noise anyway. Visually it reads as "N separate graphs".
- **What the demo shows.** Graph RAG's value shifts to **global/thematic** questions answered from community summaries ("How do these novels treat science and transgression?"). Microsoft GraphRAG was designed for exactly this, but the answers are subjective and hard to evaluate. For famous books, a closed-book LLM (one answering from memory with no retrieval) answers them well too, which weakens the comparison.
- **Making connections up front.** You can add schema-level types (Theme, Setting, Era, Narrative device, such as letters and diaries in *Dracula* and *Frankenstein*). But those edges are LLM-inferred rather than text-grounded, and designing the ontology adds complexity.

**Middle ground.** "Different books" can still mean different authors and genres writing about the **same** entities.
- The Trojan cycle is Homer and Virgil, ~700 years apart, Greek vs. Roman, sharing one story world.
- Plutarch plus Shakespeare's *Julius Caesar* and *Antony and Cleopatra* is history and drama covering the same people.

This satisfies both readings and keeps the graph connected.

**Recommendation.** Choose (a) or the middle ground. The core feature, "see which entities, relationships, communities and passages were activated", only looks impressive when activation spreads across a connected graph.

---

## 6. Open questions for the user

1. **Which reading of "a set of different books"?** Shared universe (recommended), unrelated books, or the middle ground (§5).
2. **Myth/fiction or history?** The Trojan cycle is more recognizable and easier. Plutarch is more distinctive, more "enterprise-like", and harder.
3. **Translation choice for the Trojan set.**
   - Butler/Mackail: consistent Roman names (Ulysses, Jove), so the only alias work is for user queries.
   - Mixing in Greek-name translations (Butcher-Lang *Odyssey* **1728**: "Odysseus" 718×, "Zeus" 272×, "Athene" 181×), or adding the *Argonautica*: this makes cross-translation entity resolution a deliberate showcase, at the cost of extra work.
4. **Corpus size target.** Small and fast to iterate (~300–550k tokens: the Trojan set, Plutarch, Oz books 1–6), or "big = impressive" (~850k–1M tokens: all 14 Oz books, Sherlock, Austen)? Cost is negligible either way. The difference is iteration time and how readable the graph visualization stays.
5. **How to handle the LLM's memory of famous books.** `02-architecture.md` already proposes a closed-book baseline. Do you want demo queries chosen so the closed-book baseline visibly fails? Examples: "Is the horse in the *Iliad*?", "Pyle's Mordred is a nephew", "silver shoes in Oz".
6. **Cleaning scope.** Should translator prefaces, footnotes and sidenotes be stripped (recommended), or kept as extra documents?
7. **Repo hygiene.** Strip all "Project Gutenberg" references before committing texts. Gutenberg's license says a stripped public-domain text is "unrestricted by U.S. intellectual property law" [S8]. Otherwise you must keep the full Gutenberg license with each file.

---

## Sources

- [S1] Microsoft GraphRAG, Getting Started (uses *A Christmas Carol*, Gutenberg 24022): https://microsoft.github.io/graphrag/get_started/
- [S2] LightRAG README (demo `book.txt` = *A Christmas Carol*, `./dickens` working directory): https://github.com/hkuds/lightrag
- [S3] fast-graphrag README (quickstart uses *A Christmas Carol*): https://github.com/circlemind-ai/fast-graphrag
- [S4] Microsoft GraphRAG defaults (chunk size 1200, overlap 100, `max_gleanings` 1): https://raw.githubusercontent.com/microsoft/graphrag/main/packages/graphrag/graphrag/config/defaults.py
- [S5] Microsoft GraphRAG extraction prompt (measured at 1,853 o200k tokens for the file): https://raw.githubusercontent.com/microsoft/graphrag/main/packages/graphrag/graphrag/prompts/index/extract_graph.py
- [S6] Anthropic pricing (Haiku 4.5 $1 / $5; batch $0.50 / $2.50; cache hits 0.1×): https://platform.claude.com/docs/en/about-claude/pricing
- [S7] OpenAI pricing (gpt-4o-mini $0.15 / $0.60; gpt-5-mini $0.25 / $2.00; gpt-5-nano $0.05 / $0.40): https://developers.openai.com/api/docs/pricing
- [S8] Project Gutenberg license (stripping PG references leaves an unrestricted text): https://www.gutenberg.org/policy/license.html
- [S9] Microsoft Tech Community, "GraphRAG Costs Explained" (*Wizard of Oz*, 4,058,128 tokens), read via mirror: https://argonsys.com/microsoft-cloud/library/graphrag-costs-explained-what-you-need-to-know/ (original: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978)
- [S10] fast-graphrag benchmark sentence ("Using *The Wizard of Oz*, fast-graphrag costs $0.08 vs. graphrag $0.48"): https://github.com/circlemind-ai/fast-graphrag
- [S11] Last Sherlock Holmes stories entered US public domain 2023-01-01: https://en.wikipedia.org/wiki/The_Case-Book_of_Sherlock_Holmes and https://teleread.org/2023/01/05/sherlock-holmes-case-book-virginia-woolf-classic-others-added-to-public-domain-in-u-s/
- [S12] knowledge-management-sherlock (full 60-story canon as a knowledge-graph corpus): https://github.com/MattATurner/knowledge-management-sherlock
- [S13] DZone, Graph RAG vs. RAG on *A Study in Scarlet*: https://dzone.com/articles/how-knowledge-graphs-supercharge-rag
- [S14] Neo4j NODES 2025 talk, "How to Find a Killer Using Graphs" (Sherlock): https://neo4j.com/nodes-2025/agenda/how-to-find-a-killer-using-graphs-what-sherlock-holmes-was-really-doing-with-that-red-string/
- [S15] *A Study in Scarlet*, Mormon portrayal: https://en.wikipedia.org/wiki/A_Study_in_Scarlet and https://archive.sltrib.com/article.php?id=4170094&itype=CMSID ; [S15b] *The Sign of the Four*, Tonga portrayal: https://www.litcharts.com/lit/the-sign-of-the-four/themes/empire-and-imperialism
- [S16] Mac Carron & Kenna, "Universal properties of mythological networks", EPL 2012: https://iopscience.iop.org/article/10.1209/0295-5075/99/28002
- [S17] ScienceDaily summary (716 *Iliad* characters): https://www.sciencedaily.com/releases/2012/07/120725090933.htm
- [S18] *The Odyssey* (2026 film), US release 2026-07-17, box office $1.768B: https://en.wikipedia.org/wiki/The_Odyssey_(2026_film)
- [S19] Copyright status of the Oz books in the US (all 14 Baum books public domain; MGM film elements protected): https://en.wikipedia.org/wiki/Copyright_status_of_The_Wonderful_Wizard_of_Oz_and_related_works_in_the_United_States
- [S20] Lovecraft copyright status: https://lovecraft.fandom.com/wiki/Copyright_status_of_works_by_H._P._Lovecraft
- [S21] Other knowledge-graph demos found: MythologyGraphRAG (Greek myth; source texts not specified) https://github.com/dev-it-with-me/MythologyGraphRAG ; GraphAware, Austen entity relations https://graphaware.com/blog/ere-jane-austen/ ; Memgraph *Game of Thrones* GraphRAG demo https://github.com/memgraph/ai-demos
- [S22] Shakespeare sensitivities: Joan of Arc in *1 Henry VI* https://www.folger.edu/explore/shakespeares-works/henry-vi-part-1/appendix-joan-la-pucelle-or-joan-of-arc/ ; *Richard III* and disability https://www.folger.edu/blogs/shakespeare-and-beyond/richard-iii-disability-excerpt-unfixable-forms-katherine-schaap-williams/
- Gutenberg texts: every ID above was resolved and counted directly from `https://www.gutenberg.org/cache/epub/<ID>/pg<ID>.txt` on 2026-10-05.
