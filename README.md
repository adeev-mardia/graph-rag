# graph-rag

A graph-based retrieval-augmented generation (GraphRAG) system, built from
scratch and benchmarked head-to-head against a plain vector-retrieval
baseline over the same corpus.

The whole pipeline is intentionally offline and dependency-light: there is no
pretrained NER model, no embedding model download, and no external API call
anywhere. Entity extraction is rule-based and statistical, relation
extraction is co-occurrence-and-context based, and the embeddings used by
the vector baseline are classic TF-IDF + truncated SVD (LSA). That makes the
whole thing reproducible with pip install and nothing else - no model
weights to fetch, nothing to go stale.

## Why graph retrieval, and why benchmark it honestly

Plain vector retrieval ranks passages by how semantically similar they are
to a query. That works well when the answer sits in one passage. It
struggles on multi-hop questions, where answering requires connecting
two facts that live in different passages, joined only by a shared entity
(for example: "what connects the rocket engineer who designed the V-2 to the
rocket used on Apollo 11?" - the answer requires combining a sentence about
Wernher von Braun and the V-2 with a separate sentence about Wernher von
Braun and the Saturn V).

GraphRAG builds an explicit knowledge graph of entities and relations, so it
can traverse from a query's entities to connected entities a few hops
away, surfacing passages that a lexical or embedding match alone would never
connect.

This repository does not assume that story is true and stop there - it
measures it, on a hand-labeled evaluation set, and reports what actually
happened, including the parts where the simpler baseline won. See the
Benchmark results section below.

## Architecture

```
data/corpus/*.txt            15 documents on the history of space exploration
        |
        v
graphrag.corpus              sentence/passage segmentation
        |
        v
graphrag.entities            rule-based entity extraction + salience + alias merge
        |
        v
graphrag.relations           co-occurrence + context relation extraction
        |
        v
graphrag.graph                networkx knowledge graph (nodes = entities, edges = relations)
        |
        v
graphrag.communities          greedy-modularity community detection + extractive summaries
        |
   +----+-------------------------------+
   v                                    v
graphrag.retrieval_graph      graphrag.retrieval_vector + graphrag.embeddings
(BFS traversal from query          (TF-IDF + SVD / LSA cosine similarity,
 entities, confidence-weighted      no graph at all)
 by edge, scored by hop)
   |                                    |
   +------------------+-----------------+
                      v
           graphrag.benchmark    precision@k / recall@k / MRR over a
                                  hand-labeled eval set, single-hop vs
                                  multi-hop
```

### Entity extraction (graphrag/entities.py)

No pretrained NER model is used (there's no network path to download one
anyway, which is exactly why this is built offline-first). Instead:

1. Proper-noun phrase detection - maximal runs of capitalized tokens,
   with a small set of lowercase "glue" words allowed inside a run (of,
   the, and, ...) so names like "Jet Propulsion Laboratory" or "National
   Aeronautics and Space Administration" come out whole. Commas, semicolons
   and similar punctuation always end a run, so a list like "Neil Armstrong,
   Buzz Aldrin, and Michael Collins" is read as three names, not one. An
   interior bare "and" is treated both ways - the joined phrase and its
   two halves are kept as separate candidates - so a real multi-word name
   that happens to contain "and" survives, and so does a list of two
   unrelated names joined by "and".
2. Frequency / TF-IDF salience filtering - a candidate that only ever
   shows up once across the whole corpus is dropped as noise rather than
   kept as an entity.
3. Coreference-lite alias merging - an ALL-CAPS acronym (NASA, ISRO,
   ISS) is merged into the spelled-out entity whose initials match it; a
   short form that is a strict prefix of a longer, more specific entity name
   ("Saturn V" vs. "Saturn V rocket") is folded into the longer one; a
   possessive 's is stripped so NASA's lines up with NASA.

This is coarse on purpose, and it has real, visible limits - see the
Limitations section below.

### Relation extraction (graphrag/relations.py)

For every pair of entities that co-occur in the same sentence, the text
connecting their two mentions is pulled out and cleaned into a relation
label (long spans are trimmed down to the words right around the
connective). The same (entity_a, entity_b, label) triple is usually
observed more than once across the corpus, and each repeat increases that
edge's confidence score. This is the same co-occurrence-and-context
technique used by several real open-source GraphRAG-style systems - coarse,
but genuinely pulled from the text rather than invented.

### Knowledge graph + communities (graphrag/graph.py, graphrag/communities.py)

An undirected, weighted networkx graph: nodes carry mention counts and the
passages they appear in; edges carry the relation label, co-occurrence
weight, and confidence. Communities are found with networkx's greedy
modularity maximization, and each community gets an extractive summary -
the real corpus sentences that mention the most entities from that
community, not generated text.

### Retrieval

- Graph retrieval (graphrag/retrieval_graph.py): lexically match the
  query against known entity names, then BFS outward through the graph for
  a few hops, decaying relevance by hop distance and by edge confidence
  along the way. Hub entities that appear almost everywhere (NASA, Moon,
  Apollo) are down-weighted with an IDF-style specificity factor so a
  rare, precisely-matched entity isn't drowned out. Passages are then
  ranked by the entities they mention, together with the relations
  traversed to reach them and the relevant community summary.
- Vector retrieval (graphrag/retrieval_vector.py): the same corpus,
  embedded with TF-IDF + truncated SVD (graphrag/embeddings.py), ranked by
  cosine similarity to the query. No graph involved - this is the baseline
  graph retrieval is measured against.

### Benchmark (graphrag/benchmark.py, data/eval_set.json)

28 hand-written questions over the shipped corpus, each with its own
ground-truth set of relevant passage ids, split into 12 single-hop and 16
multi-hop questions. Both retrieval pipelines are run over every question
and scored with precision@k, recall@k and MRR.

## The corpus

data/corpus/ holds 15 short documents (99 sentences total) on the history
of space exploration - Sputnik, Vostok, Mercury/Gemini, Apollo, Skylab/Mir/
ISS, SpaceX, Artemis, the Mars rovers, Voyager, Hubble/JWST, Rosetta/
Cassini-Huygens, Chandrayaan/ISRO, the Cold War backdrop, and Wernher von
Braun's career - written so that entities recur across documents (the same
rocket engineer behind Sputnik's launch vehicle also designed the Vostok
program; the same laboratory built Voyager, the Mars rovers, and managed
Cassini) and multi-hop questions have a genuine answer that spans more than
one document.

## Usage

```bash
pip install -e .

# Ingest the corpus and print graph statistics
python -m graphrag.cli build

# Run a single query through both pipelines
python -m graphrag.cli query "How does President Kennedy's 1961 challenge to Congress connect to the first Moon landing?"

# Run the full benchmark
python -m graphrag.cli benchmark
```

### Example query

```
$ python -m graphrag.cli query "How does President Kennedy's 1961 challenge to Congress connect to the first Moon landing?"

=== Graph retrieval ===
Seed entities: ['Congress', 'Moon', 'President John F. Kennedy', 'President Dwight Eisenhower']
Relations traversed:
  Congress --[triggered what historians ... United States, prompting]--> Sputnik (confidence=0.333)
  President John F. Kennedy --[announced before]--> Congress (confidence=0.333)
  Apollo --[program was NASA's effort ... 1961 challenge to]--> Congress (confidence=0.333)
Top passages:
  [1.735] (02_gagarin_vostok#4) Shortly after Gagarin's flight, President John F. Kennedy announced before
          Congress that the United States would commit to landing a man on the Moon before the end of the decade.
  [1.735] (04_apollo_program#0) The Apollo program was NASA's effort to land astronauts on the Moon,
          launched in response to President John F. Kennedy's 1961 challenge to Congress.
  [1.427] (05_apollo11#5) Apollo 11 fulfilled the goal that President John F. Kennedy had set before
          Congress in 1961, ending the Space Race's lunar phase with an American victory.
  ...

=== Vector retrieval (baseline) ===
  [0.861] (04_apollo_program#0) The Apollo program was NASA's effort to land astronauts on the Moon, ...
  [0.709] (02_gagarin_vostok#4) Shortly after Gagarin's flight, President John F. Kennedy announced ...
  ...
```

## Benchmark results

Run with python -m graphrag.cli benchmark, k=5, over all 28 questions
(12 single-hop, 16 multi-hop). These are the real, measured numbers from an
actual run against the shipped corpus - nothing here is invented or
hand-tuned to make graph retrieval win.

| metric | graph (overall) | vector (overall) |
|---|---|---|
| precision@5 | 0.221 | 0.321 |
| recall@5 | 0.613 | 0.917 |
| MRR | 0.549 | 0.911 |

| metric | graph (single-hop) | vector (single-hop) |
|---|---|---|
| precision@5 | 0.117 | 0.217 |
| recall@5 | 0.583 | 1.000 |
| MRR | 0.288 | 0.958 |

| metric | graph (multi-hop) | vector (multi-hop) |
|---|---|---|
| precision@5 | 0.300 | 0.400 |
| recall@5 | 0.635 | 0.854 |
| MRR | 0.745 | 0.875 |

Honest takeaway: plain TF-IDF/LSA vector retrieval wins on every metric
in this benchmark, including on the multi-hop questions. That's a real
result, not a bug, and it's worth explaining rather than hiding:

- The corpus is small (99 sentences) and densely cross-referenced by
  design - most multi-hop questions were written around a named bridging
  entity ("the Jet Propulsion Laboratory", "the European Space Agency",
  "Wernher von Braun"), and that entity's name tends to appear in the
  surface text of both relevant passages. A lexical/TF-IDF match can find
  both passages directly from shared vocabulary, without needing graph
  traversal at all, which is exactly why multi-hop questions don't give
  graph retrieval as much of an edge here as the premise might suggest.
- The gap between graph and vector retrieval is visibly smaller on
  multi-hop questions (MRR 0.745 vs. 0.875, a 0.13 gap) than on single-hop
  questions (MRR 0.288 vs. 0.958, a 0.67 gap) - consistent with the graph
  path being comparatively more useful exactly where the premise says it
  should be, even though it does not overtake the baseline here.
- Graph retrieval's weakest spot is single-hop precision: its passage
  ranking rewards any passage that mentions several matched entities, so
  a direct one-fact answer sometimes gets outranked by a passage mentioning
  more (less relevant) entities. A sparser scoring function tuned
  specifically for single-hop recall would likely close part of this gap.
- Seed matching is purely lexical/string-based. A query like "the Soviet
  space program's chief engineer" never literally names "Korolev", so
  graph retrieval's seed step fails to anchor on him at all, while TF-IDF
  still catches the shared words "chief", "engineer" and "Soviet" in the
  matching passage. This is a real limitation of string-based entity
  resolution, not a scoring issue.

On a larger, less densely-named corpus, or on multi-hop questions where the
bridging entity is referenced indirectly in the query but shared only via
graph structure (not surface vocabulary), the gap would be expected to look
different. This benchmark is honest about what it actually measured on the
corpus it shipped with.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

26 tests cover sentence segmentation edge cases, entity extraction
(salience filtering, acronym merging, noise rejection), relation extraction
(real co-occurring pairs vs. pairs that never co-occur), graph construction,
community detection, and two end-to-end retrieval tests against the real
shipped corpus. All 26 pass.

## Limitations

- Coreference is lexical, not semantic. "Korolev" and "Sergei Korolev",
  or "Kennedy" and "President John F. Kennedy", only merge when one is a
  strict prefix of the other or an acronym of it. A bare surname that
  isn't a prefix of the full name (most surnames) stays a separate,
  smaller entity. This is the single biggest source of missed graph
  connections in the benchmark above.
- Relation labels are coarse. The label on an edge is the raw
  connecting text between two entity mentions, lightly cleaned - it is not
  a normalized predicate like founded_by or located_in.
- Seed matching is string-based. Graph retrieval can only anchor on an
  entity the query names closely enough in words; it has no semantic
  fallback the way the TF-IDF/LSA baseline implicitly does through shared
  vocabulary.
- The corpus is small. 99 sentences is enough to build a real,
  non-trivial graph and run a genuine benchmark, but it's far smaller than
  a production corpus, and some of the numbers above (especially anything
  near 0 or 1) would be noisy at this scale.
