from graphrag.corpus import Corpus, Passage
from graphrag.entities import extract_entities
from graphrag.relations import extract_relations


def _make_corpus(sentences):
    passages = [
        Passage(f"doc#{i}", "doc", i, text) for i, text in enumerate(sentences)
    ]
    return Corpus(passages)


def test_extracts_relation_between_cooccurring_entities():
    corpus = _make_corpus(
        [
            "Marie Curie discovered polonium with Pierre Curie in 1898.",
            "Marie Curie later worked with Pierre Curie on further research.",
            "Pierre Curie died in 1906.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    edges = extract_relations(corpus, index)
    marie_id = index.resolve("Marie Curie")
    pierre_id = index.resolve("Pierre Curie")
    key = tuple(sorted((marie_id, pierre_id)))
    assert key in edges
    edge = edges[key]
    assert edge.weight >= 2
    assert edge.confidence() > 0


def test_no_relation_for_entities_never_cooccurring():
    corpus = _make_corpus(
        [
            "Marie Curie discovered polonium in 1898.",
            "Marie Curie later discovered radium as well.",
            "Albert Einstein published his theory in 1905.",
            "Albert Einstein won the Nobel Prize in 1921.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    edges = extract_relations(corpus, index)
    marie_id = index.resolve("Marie Curie")
    einstein_id = index.resolve("Albert Einstein")
    key = tuple(sorted((marie_id, einstein_id)))
    assert key not in edges


def test_relation_label_is_derived_from_connecting_text():
    corpus = _make_corpus(
        [
            "Marie Curie collaborated closely with Pierre Curie on radioactivity.",
            "Marie Curie and Pierre Curie shared the Nobel Prize.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    edges = extract_relations(corpus, index)
    marie_id = index.resolve("Marie Curie")
    pierre_id = index.resolve("Pierre Curie")
    key = tuple(sorted((marie_id, pierre_id)))
    edge = edges[key]
    # The label should be real text pulled from the sentence, not a
    # placeholder, and should contain at least one real word.
    assert edge.best_label() != ""
    assert any(c.isalpha() for c in edge.best_label())
