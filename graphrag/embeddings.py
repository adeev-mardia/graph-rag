"""Local, fully offline embeddings via classic TF-IDF + truncated SVD
(latent semantic analysis).

This is a real, well-established embedding technique -- LSA predates
neural embeddings and is still used as a lightweight, zero-download
baseline in retrieval research and in other local-first RAG projects.
It needs no model weights and no network access: the "model" is fit
directly from the corpus at index time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class LSAEmbedder:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    doc_vectors: np.ndarray  # (n_docs, n_components), L2-normalized

    def embed_query(self, text: str) -> np.ndarray:
        tfidf = self.vectorizer.transform([text])
        vec = self.svd.transform(tfidf)[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def similarities(self, text: str) -> np.ndarray:
        q = self.embed_query(text)
        return self.doc_vectors @ q


def fit_lsa(texts: list[str], n_components: int = 64) -> LSAEmbedder:
    n_components = max(2, min(n_components, len(texts) - 1, 128))
    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=13)
    doc_vectors = svd.fit_transform(tfidf_matrix)
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_vectors = doc_vectors / norms
    return LSAEmbedder(vectorizer=vectorizer, svd=svd, doc_vectors=doc_vectors)
