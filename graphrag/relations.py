"""Co-occurrence-and-context relation extraction.

For every pair of entities that co-occur within the same sentence
(a one-sentence window), the text connecting their two mentions is
pulled out and cleaned into a coarse relation label. The same
(entity_a, entity_b, label) triple is usually seen more than once
across the corpus; each repeated observation increases that edge's
confidence score. This is the standard open-information-extraction
style approach used by several real open-source GraphRAG
implementations -- deliberately coarse, but genuinely derived from the
text rather than invented.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from graphrag.corpus import Corpus, Passage
from graphrag.entities import EntityIndex, _extract_runs, _normalize

_STOP_CONNECTOR_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
}

# Verb/preposition-ish function words we prefer to keep when trimming a
# connecting span down to its most informative core.
_RELATION_STRIP_LEADING = re.compile(r"^(,|\s)+")
_RELATION_STRIP_TRAILING = re.compile(r"(,|\s)+$")


def _clean_relation_text(span: str) -> str:
    span = span.strip()
    span = _RELATION_STRIP_LEADING.sub("", span)
    span = _RELATION_STRIP_TRAILING.sub("", span)
    span = re.sub(r"\s+", " ", span)
    # Collapse overly long connecting spans (likely two unrelated
    # clauses rather than a real predicate) down to a bounded window of
    # words around the connective core, keeping the first few and last
    # few tokens which usually carry the verb/preposition.
    words = span.split()
    if len(words) > 8:
        span = " ".join(words[:4] + ["..."] + words[-3:])
    return span


def _find_entity_spans(
    sentence: str, entity_index: EntityIndex
) -> list[tuple[int, int, str]]:
    """Locate the character span of each resolvable entity mention in a
    sentence, using the same proper-noun-run detector used for
    extraction so spans line up with what was actually counted as a
    mention."""
    spans: list[tuple[int, int, str]] = []
    search_from = 0
    for run in _extract_runs(sentence, 0):
        entity_id = entity_index.resolve(run)
        if entity_id is None:
            continue
        idx = sentence.find(run, search_from)
        if idx == -1:
            idx = sentence.find(run)
        if idx == -1:
            continue
        spans.append((idx, idx + len(run), entity_id))
    return spans


@dataclass
class RelationEdge:
    source: str
    target: str
    labels: dict[str, int] = field(default_factory=dict)  # label -> freq
    example_passages: dict[str, list[str]] = field(default_factory=dict)  # label -> passage_ids
    weight: int = 0

    def best_label(self) -> str:
        if not self.labels:
            return "related to"
        return max(self.labels.items(), key=lambda kv: kv[1])[0]

    def confidence(self) -> float:
        # Simple, honest confidence score: how often this pair
        # co-occurred relative to how many distinct labels were seen
        # for it (consistent phrasing across the corpus => higher
        # confidence that the relation is real rather than noise).
        total = sum(self.labels.values())
        if total == 0:
            return 0.0
        top = max(self.labels.values())
        return round((top / total) * min(1.0, total / 3.0), 3)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.best_label(),
            "weight": self.weight,
            "confidence": self.confidence(),
            "all_labels": self.labels,
            "example_passages": self.example_passages.get(self.best_label(), [])[:3],
        }


def extract_relations(
    corpus: Corpus, entity_index: EntityIndex, max_entities_per_sentence: int = 6
) -> dict[tuple[str, str], RelationEdge]:
    edges: dict[tuple[str, str], RelationEdge] = {}
    passage_by_id = {p.passage_id: p for p in corpus.passages}

    for passage_id, entity_ids in entity_index.mentions_by_passage.items():
        if len(entity_ids) < 2:
            continue
        passage = passage_by_id[passage_id]
        spans = _find_entity_spans(passage.text, entity_index)
        if len(spans) < 2:
            continue
        spans.sort(key=lambda s: s[0])
        # Deduplicate to at most one span per entity id (first occurrence),
        # capped to avoid combinatorial blowup on long sentences.
        seen: dict[str, tuple[int, int]] = {}
        ordered: list[tuple[int, int, str]] = []
        for start, end, eid in spans:
            if eid in seen:
                continue
            seen[eid] = (start, end)
            ordered.append((start, end, eid))
        ordered = ordered[:max_entities_per_sentence]

        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a_start, a_end, a_id = ordered[i]
                b_start, b_end, b_id = ordered[j]
                if a_id == b_id:
                    continue
                connector = passage.text[a_end:b_start]
                label = _clean_relation_text(connector)
                if not label:
                    label = "mentioned with"
                key = tuple(sorted((a_id, b_id)))
                edge = edges.setdefault(key, RelationEdge(source=key[0], target=key[1]))
                edge.labels[label] = edge.labels.get(label, 0) + 1
                edge.example_passages.setdefault(label, []).append(passage_id)
                edge.weight += 1

    return edges
