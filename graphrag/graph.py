"""Knowledge graph construction with networkx."""

from __future__ import annotations

import networkx as nx

from graphrag.corpus import Corpus
from graphrag.entities import EntityIndex
from graphrag.relations import RelationEdge, extract_relations


def build_graph(corpus: Corpus, entity_index: EntityIndex) -> nx.Graph:
    """Build an undirected weighted graph from extracted entities and
    relations.

    Node attributes: ``name`` (canonical name), ``mention_count``,
    ``doc_ids``, ``passage_ids``.
    Edge attributes: ``label`` (best relation phrase), ``weight``
    (co-occurrence frequency), ``confidence``, ``all_labels``,
    ``example_passages``.
    """
    graph = nx.Graph()
    for entity_id, entity in entity_index.entities.items():
        graph.add_node(
            entity_id,
            name=entity.canonical_name,
            variants=sorted(entity.variants),
            mention_count=entity.mention_count,
            doc_ids=sorted(entity.doc_ids),
            passage_ids=sorted(entity.passage_ids),
        )

    edges = extract_relations(corpus, entity_index)
    for (a, b), edge in edges.items():
        if a not in graph or b not in graph:
            continue
        graph.add_edge(
            a,
            b,
            label=edge.best_label(),
            weight=edge.weight,
            confidence=edge.confidence(),
            all_labels=edge.labels,
            example_passages=edge.to_dict()["example_passages"],
        )

    return graph


def graph_summary(graph: nx.Graph) -> dict:
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "top_entities_by_degree": [
            {"entity_id": eid, "name": graph.nodes[eid]["name"], "degree": deg}
            for eid, deg in top_nodes
        ],
    }
