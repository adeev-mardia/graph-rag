"""Command-line interface for the graph-rag pipeline.

Subcommands:
    build       ingest a corpus directory and report graph statistics
    query       run graph retrieval and vector retrieval for one query
    benchmark   run both pipelines over the hand-labeled eval set
"""

from __future__ import annotations

import argparse
import json
import sys

from graphrag.benchmark import load_eval_set, run_benchmark
from graphrag.communities import summarize_communities
from graphrag.corpus import load_corpus
from graphrag.entities import extract_entities
from graphrag.graph import build_graph, graph_summary
from graphrag.retrieval_graph import GraphIndex, graph_retrieve
from graphrag.retrieval_vector import build_vector_index, vector_retrieve


def _build_all(corpus_dir: str, min_mentions: int = 2):
    corpus = load_corpus(corpus_dir)
    entity_index = extract_entities(corpus, min_mentions=min_mentions)
    graph = build_graph(corpus, entity_index)
    communities = summarize_communities(graph, entity_index, corpus)
    graph_index = GraphIndex(
        graph=graph, entity_index=entity_index, corpus=corpus, communities=communities
    )
    vector_index = build_vector_index(corpus)
    return corpus, entity_index, graph, communities, graph_index, vector_index


def cmd_build(args):
    corpus, entity_index, graph, communities, _, _ = _build_all(
        args.corpus_dir, args.min_mentions
    )
    summary = graph_summary(graph)
    print(f"Loaded {len(corpus)} passages from {args.corpus_dir}")
    print(f"Extracted {len(entity_index.entities)} entities")
    print(f"Graph: {summary['num_nodes']} nodes, {summary['num_edges']} edges")
    print(f"Detected {len(communities)} communities")
    print("Top entities by degree:")
    for item in summary["top_entities_by_degree"]:
        print(f"  {item['name']}  (degree={item['degree']})")


def cmd_query(args):
    _, _, _, _, graph_index, vector_index = _build_all(args.corpus_dir, args.min_mentions)

    print(f"Query: {args.query}\n")
    print("=== Graph retrieval ===")
    graph_result = graph_retrieve(graph_index, args.query, k=args.k, num_hops=args.hops)
    print(f"Seed entities: {graph_result['seed_entities']}")
    print(f"Expanded entities ({len(graph_result['expanded_entities'])}): "
          f"{graph_result['expanded_entities'][:12]}")
    if graph_result["relations_used"]:
        print("Relations traversed:")
        for rel in graph_result["relations_used"][:6]:
            print(f"  {rel['source']} --[{rel['label']}]--> {rel['target']} "
                  f"(confidence={rel['confidence']})")
    print("Top passages:")
    for p in graph_result["passages"]:
        print(f"  [{p['score']}] ({p['passage_id']}) {p['text']}")
    if graph_result["community_summaries"]:
        print("\nRelated community summary:")
        for cs in graph_result["community_summaries"][:1]:
            for s in cs["summary"]:
                print(f"  - {s}")

    print("\n=== Vector retrieval (baseline) ===")
    for r in vector_retrieve(vector_index, args.query, k=args.k):
        print(f"  [{round(r['score'], 4)}] ({r['passage_id']}) {r['text']}")

    if args.json:
        print("\n=== JSON ===")
        print(json.dumps(graph_result, indent=2))


def cmd_benchmark(args):
    _, _, _, _, graph_index, vector_index = _build_all(args.corpus_dir, args.min_mentions)
    questions = load_eval_set(args.eval_set)

    def graph_fn(query: str, k: int) -> list[str]:
        result = graph_retrieve(graph_index, query, k=k, num_hops=args.hops)
        return [p["passage_id"] for p in result["passages"]]

    def vector_fn(query: str, k: int) -> list[str]:
        result = vector_retrieve(vector_index, query, k=k)
        return [r["passage_id"] for r in result]

    graph_report = run_benchmark(questions, graph_fn, k=args.k)
    vector_report = run_benchmark(questions, vector_fn, k=args.k)

    print(f"Benchmark over {len(questions)} questions, k={args.k}\n")
    print(f"{'metric':<16}{'graph (overall)':<18}{'vector (overall)':<18}")
    g_sum = graph_report.summary()
    v_sum = vector_report.summary()
    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        print(f"{metric:<16}{g_sum['overall'][metric]:<18}{v_sum['overall'][metric]:<18}")

    print("\n-- single-hop questions --")
    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        print(f"{metric:<16}{g_sum['single_hop'][metric]:<18}{v_sum['single_hop'][metric]:<18}")

    print("\n-- multi-hop questions --")
    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        print(f"{metric:<16}{g_sum['multi_hop'][metric]:<18}{v_sum['multi_hop'][metric]:<18}")

    if args.json:
        out = {
            "graph": g_sum,
            "vector": v_sum,
            "k": args.k,
            "n_questions": len(questions),
        }
        print("\n=== JSON ===")
        print(json.dumps(out, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="graphrag", description=__doc__)
    parser.add_argument("--corpus-dir", default="data/corpus", help="directory of .txt documents")
    parser.add_argument("--min-mentions", type=int, default=2, help="entity salience threshold")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="ingest corpus and print graph stats")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="run a query through both retrieval pipelines")
    p_query.add_argument("query", help="natural language query")
    p_query.add_argument("-k", type=int, default=5)
    p_query.add_argument("--hops", type=int, default=2)
    p_query.add_argument("--json", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_bench = sub.add_parser("benchmark", help="run the IR benchmark")
    p_bench.add_argument("--eval-set", default="data/eval_set.json")
    p_bench.add_argument("-k", type=int, default=5)
    p_bench.add_argument("--hops", type=int, default=2)
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
