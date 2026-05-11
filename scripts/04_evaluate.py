"""
Step 4: Run evaluation — produces the paper's main results table.

Run: python scripts/04_evaluate.py

Compares all methods on DefinitionBench:
  - Naive chunking (Baseline 1)
  - Summary-Augmented Chunking / SAC (Baseline 2 — prior SOTA)
  - Post-retrieval definition injection (Baseline 3 — closest prior art)
  - Ours: pre-embedding definition injection (our method)

Results split by Type A and Type B questions.
Type B is where our method should win decisively.
"""

import sys
import json
sys.path.insert(0, ".")

from pathlib import Path
from rich.console import Console

from config import DATA_DIR, BENCHMARK_DIR, EMBEDDING_PROVIDER, EMBEDDING_MODEL
from src.eval.benchmark import load_benchmark, compute_split_metrics, print_results_table, save_results
from src.embed.embedder import LegalVectorStore

console = Console()


def build_retrieval_fn(store: LegalVectorStore, collection: str, top_k: int = 10):
    """Baseline retrieval — plain query, no enrichment."""
    def retrieve(query: str, contract_id: str) -> list[str]:
        hits = store.query(
            query_text=query,
            collection=collection,
            top_k=top_k,
            contract_id=contract_id,
        )
        return [h["id"] for h in hits]
    return retrieve


def build_enriched_retrieval_fn(store: LegalVectorStore, top_k: int = 10):
    """
    Enriched retrieval — uses the pre-built enriched_question from the benchmark,
    which already has the relevant contract definitions injected.
    This aligns the query embedding with the enriched chunk embeddings.
    """
    def retrieve(query: str, contract_id: str) -> list[str]:
        # query here is already the enriched_question (see eval loop below)
        hits = store.query(
            query_text=query,
            collection="enriched",
            top_k=top_k,
            contract_id=contract_id,
        )
        return [h["id"] for h in hits]

    return retrieve


def main(top_k: int = 10):
    console.rule("[bold]Step 4: Evaluation[/bold]")

    # Load benchmark
    bench_path = BENCHMARK_DIR / "definition_bench.json"
    if not bench_path.exists():
        console.print("[red]Run scripts/03_create_benchmark.py first.[/red]")
        return

    qa_pairs = load_benchmark(bench_path)
    console.print(f"Loaded {len(qa_pairs)} QA pairs")

    type_a = sum(1 for q in qa_pairs if not q.requires_definition)
    type_b = sum(1 for q in qa_pairs if q.requires_definition)
    console.print(f"  Type A: {type_a}  |  Type B: {type_b}")

    # Load vector store
    store = LegalVectorStore(
        persist_dir=DATA_DIR / "chroma",
        provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
    )

    # Define methods to compare
    methods = {
        "Naive (baseline)":           build_retrieval_fn(store, "baseline", top_k),
        "Pre-embed injection (ours)":  build_enriched_retrieval_fn(store, top_k),
    }

    # Run retrieval for all methods on all QA pairs
    console.print("\nRunning retrieval for all methods...")
    all_raw_results: dict[str, list[dict]] = {name: [] for name in methods}

    enriched_method = "Pre-embed injection (ours)"
    for qa in qa_pairs:
        for method_name, retrieve_fn in methods.items():
            # Our method uses the enriched question (definitions pre-injected);
            # baselines use the plain question. Both are stored in the benchmark.
            if method_name == enriched_method:
                query = qa.enriched_question
            else:
                query = qa.question
            retrieved = retrieve_fn(query, qa.contract_id)
            all_raw_results[method_name].append({
                "retrieved_ids":        retrieved,
                "relevant_ids":         set(qa.relevant_chunk_ids),
                "requires_definition":  qa.requires_definition,
                "query_id":             qa.id,
            })

    # Compute split metrics
    all_metrics = {}
    for method_name, results in all_raw_results.items():
        metrics_a, metrics_b = compute_split_metrics(results)
        all_metrics[method_name] = (metrics_a, metrics_b)

    # Print results table
    print_results_table(all_metrics)

    # Save results
    results_path = DATA_DIR / "results.json"
    save_results(all_metrics, results_path)

    # Key finding summary for paper
    console.rule("[bold]Key Finding[/bold]")
    ours_b  = all_metrics.get("Pre-embed injection (ours)", ({}, {}))[1]
    naive_b = all_metrics.get("Naive (baseline)", ({}, {}))[1]

    if ours_b and naive_b:
        p5_improvement = (ours_b.get("P@5", 0) - naive_b.get("P@5", 0)) / max(naive_b.get("P@5", 0.001), 0.001) * 100
        mrr_improvement = (ours_b.get("MRR", 0) - naive_b.get("MRR", 0)) / max(naive_b.get("MRR", 0.001), 0.001) * 100
        console.print(f"\nType B (definition-dependent queries):")
        console.print(f"  P@5 improvement over naive: [green]+{p5_improvement:.1f}%[/green]")
        console.print(f"  MRR improvement over naive: [green]+{mrr_improvement:.1f}%[/green]")
        console.print("\n[bold]This is your headline number for the abstract.[/bold]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    main(top_k=args.k)
