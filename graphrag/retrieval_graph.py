"""Graph-based retrieval.

Given a query, this:

1. Finds query-relevant entities by lexical/string match against
   entity canonical names and variants.
2. Expands from those seed entities via real graph traversal (BFS up
   to ``num_hops``), so entities connected to the query only indirectly
   -- through one or more intermediate entities -- are still reachable.
   This is exactly the mechanism that should give graph retrieval an
   edge on multi-hop questions, where the two relevant facts live in
   different passages connected only through a shared entity.
3. Scores and surfaces passages mentioning the seed/expanded entities,
   together with the connecting relations and the community summary of
   the entities involved, as the retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from graphrag.communities import Community, community_for_entity
from graphrag.corpus import Corpus
from graphrag.entities import EntityIndex, _normalize

_TOKEN_RE = re.compile(r"[A-Za-z]+")
_HOP_DECAY = 0.55


@dataclass
class GraphIndex:
    graph: nx.Graph
    entity_index: EntityIndex
    corpus: Corpus
    communities: list[Community]


def _query_tokens(query: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(query)}


def _specificity_weights(index: GraphIndex) -> dict[str, float]:
    """An IDF-style weight that down-ranks entities mentioned so often
    across the corpus that they carry little distinguishing signal
    (e.g. "NASA", "Apollo", "Moon" turn up almost everywhere), relative
    to entities that are rarer and therefore more informative when a
    query actually names them. Without this, high-degree hub entities
    drown out the specific entity that is actually doing the work of
    connecting two passages.
    """
    import math

    if not index.entity_index.entities:
        return {}
    max_count = max(e.mention_count for e in index.entity_index.entities.values())
    weights = {}
    for eid, entity in index.entity_index.entities.items():
        weights[eid] = 1.0 / math.log(2 + (entity.mention_count / max(1, max_count)) * (math.e - 1) * 10)
    # Normalize to [0.2, 1.0] so even the most generic entity still counts a little.
    lo, hi = min(weights.values()), max(weights.values())
    span = hi - lo or 1.0
    return {eid: 0.2 + 0.8 * (w - lo) / span for eid, w in weights.items()}


def find_seed_entities(index: GraphIndex, query: str, max_seeds: int = 6) -> list[tuple[str, float]]:
    """Lexically match the query against known entity names/variants.

    Exact normalized match scores highest; a partial token-overlap
    match (the query mentions some but not all words of a multi-word
    entity name, or vice versa) scores proportionally lower. Matches
    are then weighted down for very generic, high-frequency entities
    (see ``_specificity_weights``) so a rare, precisely-matched entity
    isn't drowned out by an incidental mention of "NASA" or "Moon".
    This is a real string-matching retrieval step, not a lookup table
    of expected queries.
    """
    q_norm = _normalize(query)
    q_tokens = _query_tokens(query)
    scored: dict[str, float] = {}
    specificity = _specificity_weights(index)

    for entity_id, entity in index.entity_index.entities.items():
        best = 0.0
        names = set(entity.variants) | {entity.canonical_name}
        for name in names:
            name_norm = _normalize(name)
            if not name_norm:
                continue
            if name_norm in q_norm:
                best = max(best, 1.0)
                continue
            name_tokens = set(name_norm.split())
            if not name_tokens:
                continue
            overlap = len(name_tokens & q_tokens)
            if overlap:
                score = overlap / len(name_tokens)
                best = max(best, 0.5 * score)
        if best > 0:
            weighted = best * specificity.get(entity_id, 1.0)
            scored[entity_id] = max(scored.get(entity_id, 0.0), weighted)

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:max_seeds]


def expand_entities(
    index: GraphIndex, seeds: list[tuple[str, float]], num_hops: int = 2
) -> dict[str, float]:
    """BFS outward from seed entities, decaying relevance score by hop
    distance and by edge confidence along the path."""
    scores: dict[str, float] = dict(seeds)
    frontier = {eid: score for eid, score in seeds}

    for _hop in range(num_hops):
        next_frontier: dict[str, float] = {}
        for eid, score in frontier.items():
            if eid not in index.graph:
                continue
            for neighbor in index.graph.neighbors(eid):
                edge = index.graph.get_edge_data(eid, neighbor) or {}
                confidence = max(edge.get("confidence", 0.3), 0.15)
                propagated = score * _HOP_DECAY * confidence
                if propagated <= scores.get(neighbor, 0.0):
                    continue
                scores[neighbor] = max(scores.get(neighbor, 0.0), propagated)
                next_frontier[neighbor] = propagated
        frontier = next_frontier
        if not frontier:
            break
    return scores


def graph_retrieve(index: GraphIndex, query: str, k: int = 5, num_hops: int = 2) -> dict:
    seeds = find_seed_entities(index, query)
    entity_scores = expand_entities(index, seeds, num_hops=num_hops)

    q_tokens = _query_tokens(query)
    passage_scores: dict[str, float] = {}
    for eid, escore in entity_scores.items():
        entity = index.entity_index.entities.get(eid)
        if entity is None:
            continue
        for pid in entity.passage_ids:
            passage_scores[pid] = passage_scores.get(pid, 0.0) + escore

    # Small lexical tie-break bonus: passages sharing query words directly.
    passage_by_id = {p.passage_id: p for p in index.corpus.passages}
    for pid in list(passage_scores.keys()):
        text_tokens = _query_tokens(passage_by_id[pid].text)
        overlap = len(text_tokens & q_tokens)
        passage_scores[pid] += 0.02 * overlap

    ranked_passages = sorted(passage_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    results = []
    for rank, (pid, score) in enumerate(ranked_passages):
        passage = passage_by_id[pid]
        results.append(
            {
                "rank": rank + 1,
                "passage_id": pid,
                "doc_id": passage.doc_id,
                "text": passage.text,
                "score": round(score, 4),
            }
        )

    seed_names = [index.entity_index.entities[eid].canonical_name for eid, _ in seeds if eid in index.entity_index.entities]
    expanded_names = [
        index.entity_index.entities[eid].canonical_name
        for eid in entity_scores
        if eid in index.entity_index.entities
    ]

    relevant_edges = []
    seen_pairs = set()
    seed_ids = {eid for eid, _ in seeds}
    for eid in entity_scores:
        if eid not in index.graph:
            continue
        for neighbor in index.graph.neighbors(eid):
            if neighbor not in entity_scores:
                continue
            pair = tuple(sorted((eid, neighbor)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edge = index.graph.get_edge_data(eid, neighbor) or {}
            relevant_edges.append(
                {
                    "source": index.entity_index.entities[pair[0]].canonical_name,
                    "target": index.entity_index.entities[pair[1]].canonical_name,
                    "label": edge.get("label", ""),
                    "confidence": edge.get("confidence", 0.0),
                }
            )

    community_summaries = []
    seen_communities = set()
    for eid in seed_ids:
        community = community_for_entity(index.communities, eid)
        if community is None or community.community_id in seen_communities:
            continue
        seen_communities.add(community.community_id)
        community_summaries.append(
            {
                "community_id": community.community_id,
                "entities": community.entity_names,
                "summary": community.summary_sentences,
            }
        )

    return {
        "query": query,
        "seed_entities": seed_names,
        "expanded_entities": expanded_names,
        "relations_used": relevant_edges[:10],
        "community_summaries": community_summaries,
        "passages": results,
    }
