"""Plain vector retrieval baseline (no graph): TF-IDF+SVD embeddings over
passages, ranked by cosine similarity to the query. This is the
head-to-head baseline that graph retrieval is benchmarked against.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.corpus import Corpus
from graphrag.embeddings import LSAEmbedder, fit_lsa


@dataclass
class VectorIndex:
    corpus: Corpus
    embedder: LSAEmbedder


def build_vector_index(corpus: Corpus, n_components: int = 64) -> VectorIndex:
    texts = [p.text for p in corpus.passages]
    embedder = fit_lsa(texts, n_components=n_components)
    return VectorIndex(corpus=corpus, embedder=embedder)


def vector_retrieve(index: VectorIndex, query: str, k: int = 5) -> list[dict]:
    sims = index.embedder.similarities(query)
    order = sims.argsort()[::-1][:k]
    results = []
    for rank, idx in enumerate(order):
        passage = index.corpus.passages[idx]
        results.append(
            {
                "rank": rank + 1,
                "passage_id": passage.passage_id,
                "doc_id": passage.doc_id,
                "text": passage.text,
                "score": float(sims[idx]),
            }
        )
    return results
