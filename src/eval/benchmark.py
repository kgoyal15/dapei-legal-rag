"""
Benchmark runner.

Runs all methods against the same QA pairs and computes metrics.
Produces the main results table for the paper.

Two question types (critical for the paper's argument):
  Type A: answerable without knowing defined terms (control group)
  Type B: requires correct definition resolution (treatment group)

Our method should dominate on Type B while staying competitive on Type A.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table

from src.eval.metrics import (
    precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k,
    token_f1, exact_match, compute_retrieval_metrics,
)

console = Console()


@dataclass
class QAPair:
    id: str
    contract_id: str
    question: str
    enriched_question: str          # question + injected definitions (for our method)
    answer: str
    relevant_chunk_ids: list[str]   # chunk IDs that contain the answer
    requires_definition: bool       # Type A (False) or Type B (True)
    defined_terms_needed: list[str] # which defined terms are needed


@dataclass
class MethodResult:
    method_name: str
    query_id: str
    retrieved_ids: list[str]
    answer: str
    relevant_ids: set


def load_benchmark(benchmark_path: Path) -> list[QAPair]:
    """Load benchmark QA pairs from JSON."""
    data = json.loads(benchmark_path.read_text())
    return [QAPair(**item) for item in data]


def run_benchmark(
    qa_pairs: list[QAPair],
    methods: dict,   # {"method_name": callable(query, contract_id) -> list[chunk_ids]}
    top_k: int = 10,
) -> dict:
    """
    Run all methods against all QA pairs.

    methods: dict of {name: retrieval_function}
    Each retrieval function takes (query: str, contract_id: str) and
    returns list of chunk IDs in ranked order.
    """
    results_by_method = {name: [] for name in methods}

    for qa in qa_pairs:
        for method_name, retrieve_fn in methods.items():
            retrieved = retrieve_fn(qa.question, qa.contract_id)[:top_k]
            results_by_method[method_name].append({
                "retrieved_ids": retrieved,
                "relevant_ids":  set(qa.relevant_chunk_ids),
                "requires_definition": qa.requires_definition,
                "query_id": qa.id,
            })

    return results_by_method


def compute_split_metrics(
    results: list[dict],
    k_values: list[int] = [1, 3, 5, 10],
) -> tuple[dict, dict]:
    """Separate results into Type A and Type B, compute metrics for each."""
    type_a = [r for r in results if not r["requires_definition"]]
    type_b = [r for r in results if r["requires_definition"]]

    metrics_a = compute_retrieval_metrics(type_a, k_values) if type_a else {}
    metrics_b = compute_retrieval_metrics(type_b, k_values) if type_b else {}
    return metrics_a, metrics_b


def print_results_table(all_metrics: dict[str, tuple[dict, dict]]) -> None:
    """Print the paper's main results table to terminal."""
    table = Table(title="Retrieval Results (Type A = no def needed, Type B = def needed)")
    table.add_column("Method", style="bold")
    table.add_column("Type A P@5")
    table.add_column("Type A MRR")
    table.add_column("Type B P@5", style="green")
    table.add_column("Type B MRR", style="green")
    table.add_column("Type B R@10")

    for method, (ma, mb) in all_metrics.items():
        table.add_row(
            method,
            f"{ma.get('P@5', 0):.3f}",
            f"{ma.get('MRR', 0):.3f}",
            f"{mb.get('P@5', 0):.3f}",
            f"{mb.get('MRR', 0):.3f}",
            f"{mb.get('R@10', 0):.3f}",
        )

    console.print(table)


def save_results(all_metrics: dict, output_path: Path) -> None:
    """Save full results to JSON for further analysis."""
    serializable = {
        method: {"type_a": ma, "type_b": mb}
        for method, (ma, mb) in all_metrics.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2))
    console.print(f"[green]Results saved to {output_path}[/green]")
