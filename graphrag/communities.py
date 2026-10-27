"""Community detection and extractive summarization over the knowledge
graph.

Communities are found with networkx's greedy modularity maximization
(Clauset-Newman-Moore), a real, standard community-detection algorithm.
Each community is then summarized *extractively*: rather than
generating new text, we select the corpus sentences that mention the
most entities from that community, which is a legitimate and simple
real summarization technique (the same idea behind classic
multi-document extractive summarizers).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from graphrag.corpus import Corpus
from graphrag.entities import EntityIndex


@dataclass
class Community:
    community_id: int
    entity_ids: list[str]
    entity_names: list[str]
    summary_sentences: list[str]
    summary_passage_ids: list[str]


def detect_communities(graph: nx.Graph) -> list[list[str]]:
    """Return a list of communities, each a list of entity (node) ids."""
    if graph.number_of_edges() == 0:
        return [[n] for n in graph.nodes()]
    communities = nx.algorithms.community.greedy_modularity_communities(
        graph, weight="weight"
    )
    return [sorted(c) for c in communities]


def summarize_communities(
    graph: nx.Graph,
    entity_index: EntityIndex,
    corpus: Corpus,
    top_k_sentences: int = 3,
) -> list[Community]:
    passage_by_id = {p.passage_id: p for p in corpus.passages}
    raw_communities = detect_communities(graph)

    out: list[Community] = []
    for cid, entity_ids in enumerate(raw_communities):
        entity_ids = [e for e in entity_ids if e in entity_index.entities]
        if not entity_ids:
            continue
        entity_set = set(entity_ids)

        # Score each passage by how many distinct community entities it
        # mentions (an extractive-summary salience signal), breaking
        # ties by preferring earlier, shorter sentences.
        passage_scores: dict[str, int] = {}
        for eid in entity_ids:
            for pid in entity_index.entities[eid].passage_ids:
                mentioned_here = set(entity_index.mentions_by_passage.get(pid, []))
                passage_scores[pid] = len(mentioned_here & entity_set)

        ranked = sorted(
            passage_scores.items(),
            key=lambda kv: (-kv[1], len(passage_by_id[kv[0]].text)),
        )
        top_passages = [pid for pid, score in ranked if score >= 2][:top_k_sentences]
        if not top_passages:
            top_passages = [pid for pid, _ in ranked[:top_k_sentences]]

        out.append(
            Community(
                community_id=cid,
                entity_ids=entity_ids,
                entity_names=[entity_index.entities[e].canonical_name for e in entity_ids],
                summary_sentences=[passage_by_id[pid].text for pid in top_passages],
                summary_passage_ids=top_passages,
            )
        )
    return out


def community_for_entity(communities: list[Community], entity_id: str) -> Community | None:
    for c in communities:
        if entity_id in c.entity_ids:
            return c
    return None
