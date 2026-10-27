from graphrag.corpus import Corpus, Passage
from graphrag.entities import extract_entities


def _make_corpus(sentences):
    passages = [
        Passage(f"doc#{i}", "doc", i, text) for i, text in enumerate(sentences)
    ]
    return Corpus(passages)


def test_extracts_repeated_proper_noun_phrase():
    corpus = _make_corpus(
        [
            "Marie Curie discovered polonium in 1898.",
            "Marie Curie later discovered radium as well.",
            "Her work with Marie Curie's techniques influenced chemistry.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    names = {e.canonical_name for e in index.entities.values()}
    assert "Marie Curie" in names


def test_drops_single_mention_noise():
    corpus = _make_corpus(
        [
            "Marie Curie discovered polonium in 1898.",
            "Marie Curie later discovered radium as well.",
            "Weird Zebra Thing appears exactly once here.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    names = {e.canonical_name for e in index.entities.values()}
    assert "Weird Zebra Thing" not in names
    assert "Marie Curie" in names


def test_acronym_merges_with_spelled_out_form():
    corpus = _make_corpus(
        [
            "The World Health Organization issued new guidance today.",
            "The World Health Organization also published a report last year.",
            "WHO officials praised the new guidance.",
            "WHO later clarified the report's findings.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    who_entities = [
        e for e in index.entities.values()
        if e.canonical_name == "World Health Organization"
    ]
    assert len(who_entities) == 1
    assert "WHO" in who_entities[0].variants


def test_mentions_by_passage_resolves_to_merged_entity():
    corpus = _make_corpus(
        [
            "NASA launched a new satellite today.",
            "NASA confirmed the satellite reached orbit.",
            "The National Aeronautics and Space Administration released photos.",
            "The National Aeronautics and Space Administration plans another launch.",
        ]
    )
    index = extract_entities(corpus, min_mentions=2)
    eid_nasa = index.resolve("NASA")
    eid_full = index.resolve("National Aeronautics and Space Administration")
    assert eid_nasa is not None
    assert eid_nasa == eid_full
    entity = index.entities[eid_nasa]
    assert entity.mention_count == 4
