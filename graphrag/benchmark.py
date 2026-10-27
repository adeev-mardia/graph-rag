"""Benchmark harness: run both retrieval pipelines over a hand-labeled
evaluation set and measure real information-retrieval metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RetrieveFn = Callable[[str, int], list[str]]  # query, k -> ranked passage_ids


@dataclass
class EvalQuestion:
    qid: str
    query: str
    multi_hop: bool
    relevant: set[str]


def load_eval_set(path: str | Path) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalQuestion(
            qid=item["id"],
            query=item["query"],
            multi_hop=bool(item.get("multi_hop", False)),
            relevant=set(item["relevant_passage_ids"]),
        )
        for item in data
    ]


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(top_k)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, pid in enumerate(ranked, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


@dataclass
class QuestionResult:
    qid: str
    query: str
    multi_hop: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ranked_passage_ids: list[str]


@dataclass
class BenchmarkReport:
    k: int
    per_question: list[QuestionResult]

    def _subset(self, multi_hop: bool | None):
        if multi_hop is None:
            return self.per_question
        return [r for r in self.per_question if r.multi_hop == multi_hop]

    def aggregate(self, multi_hop: bool | None = None) -> dict:
        subset = self._subset(multi_hop)
        n = len(subset)
        if n == 0:
            return {"n": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
        return {
            "n": n,
            "precision_at_k": round(sum(r.precision_at_k for r in subset) / n, 4),
            "recall_at_k": round(sum(r.recall_at_k for r in subset) / n, 4),
            "mrr": round(sum(r.reciprocal_rank for r in subset) / n, 4),
        }

    def summary(self) -> dict:
        return {
            "overall": self.aggregate(None),
            "single_hop": self.aggregate(False),
            "multi_hop": self.aggregate(True),
        }


def run_benchmark(
    questions: list[EvalQuestion], retrieve_fn: RetrieveFn, k: int = 5
) -> BenchmarkReport:
    results = []
    for q in questions:
        ranked = retrieve_fn(q.query, k)
        results.append(
            QuestionResult(
                qid=q.qid,
                query=q.query,
                multi_hop=q.multi_hop,
                precision_at_k=precision_at_k(ranked, q.relevant, k),
                recall_at_k=recall_at_k(ranked, q.relevant, k),
                reciprocal_rank=reciprocal_rank(ranked, q.relevant),
                ranked_passage_ids=ranked,
            )
        )
    return BenchmarkReport(k=k, per_question=results)
