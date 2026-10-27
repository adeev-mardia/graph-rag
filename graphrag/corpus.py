"""Corpus loading and sentence/passage segmentation.

A "passage" here is a single sentence, tagged with the id of the source
document it came from. Sentences are the unit both retrieval pipelines
operate over: fine enough to give precise evidence, coarse enough to
carry a full relation between two entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Sentence boundary heuristic: split on '.', '!', '?' followed by
# whitespace and a capital letter or end of string, but guard the most
# common false positives (abbreviations, decimal numbers, initials).
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e",
    "no", "fig", "u.s", "u.k", "jr", "sr",
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    """Split a block of text into sentences using punctuation + case cues.

    This is a heuristic splitter, not a full sentence tokenizer: it looks
    for a sentence-ending punctuation mark followed by whitespace and a
    capital letter or digit, then filters out splits that landed right
    after a common abbreviation (which would otherwise be treated as a
    sentence boundary).
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    raw_splits = _SENT_SPLIT_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for chunk in raw_splits:
        if buffer:
            buffer = buffer + " " + chunk
        else:
            buffer = chunk
        last_token_match = re.search(r"([A-Za-z]+)\.?$", buffer)
        last_token = last_token_match.group(1) if last_token_match else ""
        trailing = last_token.lower()
        is_middle_initial = len(last_token) == 1 and last_token.isupper()
        if trailing in _ABBREVIATIONS or is_middle_initial:
            # Likely a false boundary (e.g. "Dr. Smith", or a middle
            # initial like "John F. Kennedy"); keep accumulating.
            continue
        sentences.append(buffer.strip())
        buffer = ""
    if buffer:
        sentences.append(buffer.strip())
    return [s for s in sentences if len(s) > 1]


@dataclass(frozen=True)
class Passage:
    """A single retrievable unit: one sentence plus its provenance."""

    passage_id: str
    doc_id: str
    sentence_index: int
    text: str


@dataclass
class Corpus:
    passages: list[Passage]

    def __len__(self) -> int:
        return len(self.passages)

    def by_id(self, passage_id: str) -> Passage:
        for p in self.passages:
            if p.passage_id == passage_id:
                return p
        raise KeyError(passage_id)


def load_corpus(corpus_dir: str | Path) -> Corpus:
    """Load every ``*.txt`` file in ``corpus_dir`` into a flat list of
    sentence-level passages, each tagged with its document id.
    """
    corpus_dir = Path(corpus_dir)
    passages: list[Passage] = []
    for file_path in sorted(corpus_dir.glob("*.txt")):
        doc_id = file_path.stem
        text = file_path.read_text(encoding="utf-8")
        for idx, sentence in enumerate(split_sentences(text)):
            passage_id = f"{doc_id}#{idx}"
            passages.append(Passage(passage_id, doc_id, idx, sentence))
    return Corpus(passages)
