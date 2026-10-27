from graphrag.corpus import Corpus, Passage
from graphrag.entities import extract_entities
from graphrag.graph import build_graph, graph_summary
from graphrag.communities import detect_communities, summarize_communities


def _make_corpus(sentences):
    passages = [
        Passage(f"doc#{i}", "doc", i, text) for i, text in enumerate(sentences)
    ]
    return Corpus(passages)


_SAMPLE = [
    "Marie Curie discovered polonium with Pierre Curie in 1898.",
    "Marie Curie and Pierre Curie later discovered radium together.",
    "Pierre Curie died in a street accident in 1906.",
    "Irene Joliot-Curie was the daughter of Marie Curie.",
    "Irene Joliot-Curie won her own Nobel Prize in Chemistry in 1935.",
    "Albert Einstein published his theory of relativity in 1905.",
    "Albert Einstein won the Nobel Prize in Physics in 1921.",
    "Albert Einstein corresponded with Marie Curie about physics.",
]


def test_graph_has_nodes_and_edges():
    corpus = _make_corpus(_SAMPLE)
    index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, index)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


def test_graph_node_attributes_present():
    corpus = _make_corpus(_SAMPLE)
    index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, index)
    marie_id = index.resolve("Marie Curie")
    assert marie_id in graph
    node = graph.nodes[marie_id]
    assert node["mention_count"] > 0
    assert "passage_ids" in node


def test_graph_edge_has_label_and_confidence():
    corpus = _make_corpus(_SAMPLE)
    index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, index)
    marie_id = index.resolve("Marie Curie")
    pierre_id = index.resolve("Pierre Curie")
    assert graph.has_edge(marie_id, pierre_id)
    edge = graph.get_edge_data(marie_id, pierre_id)
    assert edge["label"]
    assert 0 <= edge["confidence"] <= 1


def test_graph_summary_reports_consistent_counts():
    corpus = _make_corpus(_SAMPLE)
    index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, index)
    summary = graph_summary(graph)
    assert summary["num_nodes"] == graph.number_of_nodes()
    assert summary["num_edges"] == graph.number_of_edges()
    assert len(summary["top_entities_by_degree"]) <= 10


def test_community_detection_covers_all_nodes():
    corpus = _make_corpus(_SAMPLE)
    index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, index)
    communities = detect_communities(graph)
    covered = {n for community in communities for n in community}
    assert covered == set(graph.nodes())


def test_community_summaries_use_real_corpus_sentences():
    corpus = _make_corpus(_SAMPLE)
    index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, index)
    communities = summarize_communities(graph, index, corpus)
    assert len(communities) > 0
    corpus_texts = {p.text for p in corpus.passages}
    for community in communities:
        for sentence in community.summary_sentences:
            assert sentence in corpus_texts
