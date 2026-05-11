"""
Diagnostic — run this to find out why evaluation shows all zeros.
Run: python diagnose.py
"""
import sys, json
sys.path.insert(0, ".")

from pathlib import Path
from rich.console import Console
from config import DATA_DIR, BENCHMARK_DIR, RAW_DIR, PROCESSED_DIR, EMBEDDING_PROVIDER, EMBEDDING_MODEL

console = Console()

console.rule("[bold cyan]Diagnostic[/bold cyan]")

# 1. Vector store counts
console.print("\n[cyan]1. Vector store counts[/cyan]")
try:
    from src.embed.embedder import LegalVectorStore
    store = LegalVectorStore(
        persist_dir=DATA_DIR / "chroma",
        provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
    )
    e = store.count("enriched")
    b = store.count("baseline")
    console.print(f"  enriched collection : {e} chunks")
    console.print(f"  baseline collection : {b} chunks")
    if e == 0:
        console.print("[red]  !! Vector store is EMPTY — run scripts/02_build_pipeline.py first[/red]")
except Exception as ex:
    console.print(f"[red]  Error loading store: {ex}[/red]")
    e = 0

# 2. Processed contracts
console.print("\n[cyan]2. Processed contracts[/cyan]")
processed_files = list(PROCESSED_DIR.glob("cuad_*.json")) if PROCESSED_DIR.exists() else []
console.print(f"  Processed JSON files: {len(processed_files)}")
if processed_files:
    sample = json.loads(processed_files[0].read_text())
    console.print(f"  Sample: {sample['id']} — {sample['chunk_count']} chunks, {sample['resolved_count']} resolved defs")
    if sample["chunks"]:
        console.print(f"  Sample chunk ID: {sample['chunks'][0]['id']}")

# 3. CUAD contracts + annotation coverage
console.print("\n[cyan]3. CUAD annotation coverage[/cyan]")
from src.data.cuad_loader import load_saved_cuad
contracts = load_saved_cuad(RAW_DIR / "cuad")
console.print(f"  CUAD contracts on disk: {len(contracts)}")
has_annotations = sum(1 for c in contracts if c.get("annotations"))
console.print(f"  Contracts WITH annotations: {has_annotations}/{len(contracts)}")
if has_annotations == 0:
    console.print("[red]  !! No annotations — run: python scripts/05_repair_annotations.py[/red]")

# 4. Benchmark file
console.print("\n[cyan]4. Benchmark[/cyan]")
bench = BENCHMARK_DIR / "definition_bench.json"
if not bench.exists():
    console.print("[red]  !! definition_bench.json not found — run scripts/03_create_benchmark.py[/red]")
else:
    pairs = json.loads(bench.read_text())
    console.print(f"  QA pairs total: {len(pairs)}")
    type_a = sum(1 for p in pairs if not p["requires_definition"])
    type_b = sum(1 for p in pairs if p["requires_definition"])
    console.print(f"  Type A: {type_a}  |  Type B: {type_b}")
    if pairs:
        p = pairs[0]
        console.print(f"  Sample pair: {p['id']} contract={p['contract_id']}")
        console.print(f"  relevant_chunk_ids: {p['relevant_chunk_ids']}")

# 5. Sample retrieval test
console.print("\n[cyan]5. Sample retrieval (first contract)[/cyan]")
if e > 0 and processed_files:
    sample = json.loads(processed_files[0].read_text())
    cid = sample["id"]
    hits = store.query("termination clause material adverse effect", collection="enriched",
                       top_k=3, contract_id=cid)
    console.print(f"  Query against {cid}: {len(hits)} hits")
    for h in hits:
        console.print(f"    id={h['id']}  score={h['score']:.3f}")
elif e == 0:
    console.print("  [yellow]Skipped — vector store empty[/yellow]")

console.rule("[bold]Done[/bold]")
