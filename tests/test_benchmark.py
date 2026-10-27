from graphrag.benchmark import (
    load_eval_set,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_benchmark,
)


def test_precision_recall_mrr_basic():
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d"}
    assert precision_at_k(ranked, relevant, 5) == 2 / 5
    assert recall_at_k(ranked, relevant, 5) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5  # "b" is rank 2


def test_recall_at_k_partial():
    ranked = ["a", "b", "c"]
    relevant = {"b", "z"}
    assert recall_at_k(ranked, relevant, 3) == 0.5


def test_load_eval_set_has_expected_shape():
    questions = load_eval_set("data/eval_set.json")
    assert len(questions) >= 20
    assert any(q.multi_hop for q in questions)
    assert any(not q.multi_hop for q in questions)
    for q in questions:
        assert q.relevant


def test_run_benchmark_perfect_retriever_scores_well():
    questions = load_eval_set("data/eval_set.json")

    def oracle(query: str, k: int):
        for q in questions:
            if q.query == query:
                return list(q.relevant)[:k]
        return []

    report = run_benchmark(questions, oracle, k=5)
    summary = report.aggregate()
    assert summary["recall_at_k"] > 0.9
    assert summary["mrr"] > 0.9
