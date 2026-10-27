"""graphrag: a dependency-light graph retrieval-augmented generation system.

Everything in this package runs fully offline: entity and relation
extraction are rule-based and statistical rather than relying on a
downloaded NER model, and the vector baseline uses classic TF-IDF/LSA
embeddings computed locally. That keeps the whole pipeline reproducible
with nothing to download and nothing that goes stale.
"""

__version__ = "0.1.0"
