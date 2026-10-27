"""End-to-end retrieval test over the real shipped corpus."""

from graphrag.communities import summarize_communities
from graphrag.corpus import load_corpus
from graphrag.entities import extract_entities
from graphrag.graph import build_graph
from graphrag.retrieval_graph import GraphIndex, graph_retrieve
from graphrag.retrieval_vector import build_vector_index, vector_retrieve

CORPUS_DIR = "data/corpus"


def _build():
    corpus = load_corpus(CORPUS_DIR)
    entity_index = extract_entities(corpus, min_mentions=2)
    graph = build_graph(corpus, entity_index)
    communities = summarize_communities(graph, entity_index, corpus)
    graph_index = GraphIndex(
        graph=graph, entity_index=entity_index, corpus=corpus, communities=communities
    )
    vector_index = build_vector_index(corpus)
    return graph_index, vector_index


def test_corpus_loads_and_is_nontrivial():
    corpus = load_corpus(CORPUS_DIR)
    assert len(corpus) > 50


def test_graph_retrieval_finds_direct_fact():
    graph_index, _ = _build()
    result = graph_retrieve(graph_index, "Who was the first human to walk on the Moon?", k=5)
    ids = [p["passage_id"] for p in result["passages"]]
    assert "05_apollo11#2" in ids


def test_graph_retrieval_multi_hop_reaches_connected_passage():
    graph_index, _ = _build()
    result = graph_retrieve(
        graph_index,
        "How does President Kennedy's 1961 challenge to Congress connect "
        "to the first Moon landing?",
        k=5,
        num_hops=2,
    )
    ids = {p["passage_id"] for p in result["passages"]}
    # Kennedy's challenge (in the Gagarin/Vostok document) and the
    # Apollo 11 landing itself (in a different document) are only
    # connected through the shared Apollo-program entity; a working
    # multi-hop traversal should surface passages from both documents.
    assert any(pid.startswith("02_gagarin_vostok") for pid in ids)
    assert any(pid.startswith("05_apollo11") for pid in ids)


def test_vector_retrieval_finds_direct_fact():
    _, vector_index = _build()
    results = vector_retrieve(vector_index, "Who was the first human to walk on the Moon?", k=5)
    ids = [r["passage_id"] for r in results]
    assert "05_apollo11#2" in ids


def test_graph_retrieval_returns_scored_ranked_results():
    graph_index, _ = _build()
    result = graph_retrieve(graph_index, "What did the Jet Propulsion Laboratory build?", k=5)
    scores = [p["score"] for p in result["passages"]]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
