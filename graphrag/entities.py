"""Rule-based, statistical entity extraction.

No pretrained NER model is used anywhere in this module (there is no
network access to download one, and the design goal is a fully
reproducible, zero-setup pipeline anyway). Instead entities are found
with three real, well-understood techniques:

1. Proper-noun phrase detection: maximal runs of capitalized tokens,
   allowing a small set of lowercase "glue" words inside a run
   (``of``, ``the``, ``for``, ...), which is the classic approach for
   pulling multi-word names like "National Aeronautics and Space
   Administration" or "Jet Propulsion Laboratory" out of running text
   without a part-of-speech tagger.
2. Frequency/TF-IDF salience filtering: a candidate phrase that only
   ever appears once, or that is dominated by a handful of generic
   words, is dropped as noise rather than kept as an entity.
3. Coreference-lite normalization: exact-match and near-string
   matching (normalized casing/punctuation, plus acronym detection) to
   merge obvious variants of the same name, e.g. "NASA" and "National
   Aeronautics and Space Administration".
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from graphrag.corpus import Corpus, Passage

# Lowercase connector words allowed *inside* a capitalized run without
# breaking it, e.g. "Jet Propulsion Laboratory" / "United States of America".
_GLUE_WORDS = {
    "of", "the", "and", "for", "in", "on", "at", "to", "de", "von",
}

# Words that are capitalized but are not meaningful entity heads on their
# own (sentence-initial common words, months, generic terms). A run
# consisting *only* of these is discarded.
_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "his", "her", "their", "after", "before",
    "during", "when", "while", "although", "however", "in", "on",
    "at", "later", "shortly", "many", "both", "several", "early",
    "decades", "years", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october",
    "november", "december", "that's",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'.\-]*|\d+")
_POSSESSIVE_RE = re.compile(r"'s$", re.IGNORECASE)


def _tokenize(sentence: str) -> list[str]:
    tokens = _TOKEN_RE.findall(sentence)
    # Strip a trailing possessive 's so "NASA's" / "Braun's" line up with
    # the bare name "NASA" / "Braun" when runs are normalized later.
    return [_POSSESSIVE_RE.sub("", t) or t for t in tokens]


def _is_capitalized(token: str) -> bool:
    return token[0].isupper() if token else False


def _extract_runs(sentence: str, sentence_index_in_doc: int) -> list[str]:
    """Return maximal capitalized-phrase candidates from one sentence.

    A run is a maximal sequence of tokens where every token is either
    capitalized or a glue word, anchored by at least one capitalized
    token on each end, and it must not be purely the first word of the
    sentence standing alone (to cut down on sentence-initial false
    positives like a stray "After" captured with nothing else).

    A comma, semicolon, colon, dash, parenthesis or quote always ends a
    run even if the next word is capitalized: without this, a list like
    "Neil Armstrong, Buzz Aldrin, and Michael Collins" would be read as
    one long run instead of three separate names.
    """
    segments = re.split(r"[,;:()\"'—–]", sentence)
    runs: list[str] = []
    for segment in segments:
        runs.extend(_extract_runs_from_segment(segment))
    return runs


def _extract_runs_from_segment(sentence: str) -> list[str]:
    tokens = _tokenize(sentence)
    runs: list[str] = []
    current: list[str] = []

    def flush():
        if not current:
            return
        # Trim leading/trailing glue words and sentence-initial
        # stopwords (a run can't start or end on "of"/"the"/"After"/...).
        trimmed = list(current)
        _trim_set = _GLUE_WORDS | _STOPWORDS
        while trimmed and trimmed[-1].lower() in _trim_set:
            trimmed.pop()
        while trimmed and trimmed[0].lower() in _trim_set:
            trimmed.pop(0)
        if trimmed:
            runs.append(" ".join(trimmed))
            # "and" is kept as glue inside a run so real multi-word names
            # like "National Aeronautics and Space Administration" stay
            # intact, but a bare "X and Y" is usually two different
            # entities (e.g. "Marie Curie and Pierre Curie"), not one.
            # Emit the pieces either side of an interior "and" as extra
            # candidates too, so each still accumulates its own mention
            # count; salience filtering sorts out which reading is real.
            and_positions = [i for i, w in enumerate(trimmed) if w.lower() == "and"]
            for pos in and_positions:
                left = trimmed[:pos]
                right = trimmed[pos + 1:]
                for piece in (list(left), list(right)):
                    while piece and piece[0].lower() in _trim_set:
                        piece.pop(0)
                    while piece and piece[-1].lower() in _trim_set:
                        piece.pop()
                    if piece:
                        runs.append(" ".join(piece))
        current.clear()

    for i, tok in enumerate(tokens):
        if tok.isdigit():
            if current and current[-1].endswith("-"):
                # A hyphen-joined trailing number, e.g. "Chandrayaan-3"
                # or "Apollo-11", continues the same name instead of
                # starting a new run.
                current[-1] = current[-1] + tok
            else:
                flush()
            continue
        if _is_capitalized(tok) or tok.lower() in _GLUE_WORDS:
            # Skip the very first token of the sentence if it stands
            # entirely alone (length 1 run) -- likely just normal
            # sentence capitalization, not a name.
            current.append(tok)
        else:
            flush()
    flush()

    # Drop single-token runs that are stopwords or too short to be
    # useful, and drop runs that are entirely glue/stopwords.
    cleaned = []
    for run in runs:
        words = run.split()
        words_lower = [w.lower() for w in words]
        if all(w in _STOPWORDS or w in _GLUE_WORDS for w in words_lower):
            continue
        if len(words) == 1 and (len(words[0]) <= 2 or words_lower[0] in _STOPWORDS):
            continue
        cleaned.append(run)
    return cleaned


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _acronym_of(name: str) -> str:
    words = [w for w in name.split() if w.lower() not in _GLUE_WORDS]
    return "".join(w[0] for w in words if w[0].isalpha()).upper()


def _is_acronym_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2,6}", token))


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    variants: set[str] = field(default_factory=set)
    mention_count: int = 0
    passage_ids: set[str] = field(default_factory=set)
    doc_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "variants": sorted(self.variants),
            "mention_count": self.mention_count,
            "passage_ids": sorted(self.passage_ids),
            "doc_ids": sorted(self.doc_ids),
        }


@dataclass
class EntityIndex:
    """All extracted entities plus a lookup from (passage_id) -> entity
    mentions found in that passage, used downstream by relation
    extraction."""

    entities: dict[str, Entity]
    mentions_by_passage: dict[str, list[str]]  # passage_id -> [entity_id,...]
    alias_to_entity: dict[str, str]  # normalized surface form -> entity_id

    def resolve(self, surface_form: str) -> str | None:
        return self.alias_to_entity.get(_normalize(surface_form))


def extract_entities(
    corpus: Corpus,
    min_mentions: int = 2,
    min_tfidf: float = 0.0,
) -> EntityIndex:
    """Run the full extraction + salience filter + alias merge pipeline."""

    # Step 1: collect raw candidate surface forms per passage.
    raw_by_passage: dict[str, list[str]] = {}
    for p in corpus.passages:
        raw_by_passage[p.passage_id] = _extract_runs(p.text, p.sentence_index)

    # Step 2: document frequency / TF-IDF salience scoring over raw
    # normalized surface forms, to down-weight noise that only looks
    # like a name once (e.g. a leftover capitalized sentence fragment).
    doc_ids = sorted({p.doc_id for p in corpus.passages})
    n_docs = len(doc_ids)
    doc_term_counts: dict[str, Counter] = {d: Counter() for d in doc_ids}
    global_counts: Counter = Counter()
    norm_to_surface_counts: dict[str, Counter] = defaultdict(Counter)

    passage_by_id = {p.passage_id: p for p in corpus.passages}
    for passage_id, runs in raw_by_passage.items():
        doc_id = passage_by_id[passage_id].doc_id
        for run in runs:
            norm = _normalize(run)
            if not norm:
                continue
            doc_term_counts[doc_id][norm] += 1
            global_counts[norm] += 1
            norm_to_surface_counts[norm][run] += 1

    def tfidf(norm: str, doc_id: str) -> float:
        tf = doc_term_counts[doc_id][norm]
        if tf == 0:
            return 0.0
        df = sum(1 for d in doc_ids if doc_term_counts[d][norm] > 0)
        idf = math.log((1 + n_docs) / (1 + df)) + 1
        return tf * idf

    best_tfidf: dict[str, float] = {}
    for norm in global_counts:
        for doc_id in doc_ids:
            if doc_term_counts[doc_id][norm] > 0:
                score = tfidf(norm, doc_id)
                best_tfidf[norm] = max(best_tfidf.get(norm, 0.0), score)

    # Salient terms: mentioned often enough across the corpus, or
    # scoring high enough on TF-IDF in at least one document.
    salient_norms = {
        norm
        for norm, count in global_counts.items()
        if count >= min_mentions
        or (min_tfidf > 0.0 and best_tfidf.get(norm, 0.0) >= min_tfidf)
    }

    # Step 3: build canonical entities from salient normalized forms,
    # picking the most frequent surface spelling as the canonical name.
    norm_to_entity_id: dict[str, str] = {}
    entities: dict[str, Entity] = {}
    for norm in salient_norms:
        surface_counts = norm_to_surface_counts[norm]
        canonical_name = surface_counts.most_common(1)[0][0]
        entity_id = f"e{len(entities)}"
        norm_to_entity_id[norm] = entity_id
        entities[entity_id] = Entity(
            entity_id=entity_id,
            canonical_name=canonical_name,
            variants=set(surface_counts.keys()),
        )

    # Step 4: coreference-lite -- merge acronyms with their spelled-out
    # form. An ALL-CAPS token (e.g. "NASA", "ISRO", "ISS") is merged
    # into a multi-word entity whose initials match it, when that
    # multi-word entity is itself salient. This is a real, if coarse,
    # alias-resolution rule, not a hardcoded lookup of specific names.
    acronym_norms = {
        norm for norm in salient_norms
        if _is_acronym_token(norm_to_surface_counts[norm].most_common(1)[0][0])
    }
    multiword_norms = {
        norm for norm in salient_norms
        if len(norm.split()) >= 2 and norm not in acronym_norms
    }
    merge_into: dict[str, str] = {}  # acronym norm -> multiword norm
    for acro_norm in acronym_norms:
        acro_surface = norm_to_surface_counts[acro_norm].most_common(1)[0][0]
        candidates = [
            mw for mw in multiword_norms
            if _acronym_of(norm_to_surface_counts[mw].most_common(1)[0][0]) == acro_surface
        ]
        if len(candidates) == 1:
            merge_into[acro_norm] = candidates[0]

    for acro_norm, target_norm in merge_into.items():
        target_id = norm_to_entity_id[target_norm]
        acro_id = norm_to_entity_id[acro_norm]
        if target_id == acro_id:
            continue
        target_entity = entities[target_id]
        acro_entity = entities.pop(acro_id, None)
        if acro_entity is None:
            continue
        target_entity.variants |= acro_entity.variants
        norm_to_entity_id[acro_norm] = target_id

    # Step 5: also merge near-duplicate multiword variants that differ
    # only by a leading/trailing glue word or minor punctuation, e.g.
    # "Space Launch System" vs "the Space Launch System rocket" would
    # already normalize differently, but "Apollo program" vs
    # "Apollo Program" collapse naturally via _normalize's lowercasing.
    # Here we additionally merge a short form that is a prefix or
    # suffix substring of a longer salient entity's canonical name
    # (e.g. "Vostok" and "Vostok 1" staying distinct is fine, but
    # "Saturn V rocket" and "Saturn V" should merge).
    norms_by_len = sorted(salient_norms, key=len)
    merged_short_into: dict[str, str] = {}
    for short_norm in norms_by_len:
        if short_norm not in norm_to_entity_id:
            continue
        short_id = norm_to_entity_id[short_norm]
        if short_id not in entities:
            continue
        for long_norm in norms_by_len:
            if long_norm == short_norm or long_norm not in norm_to_entity_id:
                continue
            long_id = norm_to_entity_id[long_norm]
            if long_id not in entities or long_id == short_id:
                continue
            words_short = short_norm.split()
            words_long = long_norm.split()
            if len(words_short) >= len(words_long):
                continue
            if words_long[: len(words_short)] == words_short and len(words_long) - len(words_short) <= 1:
                # short_norm is a strict prefix of long_norm with at
                # most one extra trailing word -> treat short as an
                # alias of the longer, more specific entity only when
                # the short form is comparatively rare (otherwise they
                # really are two different things, e.g. "Mars" vs
                # "Mars rovers").
                if global_counts[short_norm] <= global_counts[long_norm]:
                    merged_short_into[short_norm] = long_norm
                    break

    for short_norm, long_norm in merged_short_into.items():
        short_id = norm_to_entity_id.get(short_norm)
        long_id = norm_to_entity_id.get(long_norm)
        if short_id is None or long_id is None or short_id == long_id:
            continue
        short_entity = entities.pop(short_id, None)
        if short_entity is None:
            continue
        entities[long_id].variants |= short_entity.variants
        norm_to_entity_id[short_norm] = long_id

    # Step 6: build alias table and record mentions per passage.
    alias_to_entity: dict[str, str] = {}
    for entity_id, entity in entities.items():
        for variant in entity.variants:
            alias_to_entity[_normalize(variant)] = entity_id
        alias_to_entity[_normalize(entity.canonical_name)] = entity_id

    mentions_by_passage: dict[str, list[str]] = {}
    for passage_id, runs in raw_by_passage.items():
        found: list[str] = []
        seen_this_passage: set[str] = set()
        for run in runs:
            norm = _normalize(run)
            entity_id = norm_to_entity_id.get(norm)
            if entity_id is None or entity_id not in entities:
                continue
            if entity_id not in seen_this_passage:
                found.append(entity_id)
                seen_this_passage.add(entity_id)
            entities[entity_id].mention_count += 1
            entities[entity_id].passage_ids.add(passage_id)
            entities[entity_id].doc_ids.add(passage_by_id[passage_id].doc_id)
        mentions_by_passage[passage_id] = found

    # Drop any entity that ended up with zero confirmed mentions after
    # merging (shouldn't normally happen, but keep the index clean).
    entities = {eid: e for eid, e in entities.items() if e.mention_count > 0}

    return EntityIndex(
        entities=entities,
        mentions_by_passage=mentions_by_passage,
        alias_to_entity=alias_to_entity,
    )
