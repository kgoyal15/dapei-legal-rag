"""
Quickstart — run the full pipeline on 3 sample contracts to verify everything works.

Run: python quickstart.py

No API key needed (uses local embeddings).
Downloads CUAD and processes 3 contracts end-to-end.
"""

import sys
sys.path.insert(0, ".")

from rich.console import Console
console = Console()


def main():
    console.rule("[bold cyan]Legal RAG Pipeline — Quickstart[/bold cyan]")

    # ── 1. Load 3 CUAD contracts ─────────────────────────────────────────────
    console.print("\n[cyan]1. Loading CUAD contracts...[/cyan]")
    from config import RAW_DIR, DATA_DIR
    from src.data.cuad_loader import load_cuad, load_saved_cuad

    cuad_dir = RAW_DIR / "cuad"
    if len(list(cuad_dir.glob("cuad_*.json"))) < 3:
        contracts = load_cuad(cuad_dir, max_contracts=3)
    else:
        contracts = load_saved_cuad(cuad_dir)[:3]
    console.print(f"  Loaded {len(contracts)} contracts")

    # ── 2. Parse ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]2. Parsing contracts...[/cyan]")
    from src.data.contract_parser import parse_cuad_contract
    parsed = [parse_cuad_contract(c) for c in contracts]

    for p in parsed:
        console.print(f"  {p['id'][:50]:50s} | {len(p['sections'])} sections | {len(p['raw_text'])} chars")

    # ── 3. Extract definitions ────────────────────────────────────────────────
    console.print("\n[cyan]3. Extracting definitions (regex only, no LLM)...[/cyan]")
    from src.extract.definition_extractor import extract_definitions, definitions_to_dict

    all_defs = []
    for p in parsed:
        defs_obj = extract_definitions(p, use_llm=False)
        defs     = definitions_to_dict(defs_obj)
        all_defs.append(defs)
        console.print(f"  {p['id'][:50]:50s} | {len(defs)} definitions found")
        for term, defn in list(defs.items())[:3]:
            console.print(f"    [green]{term}[/green]: {defn[:80]}...")

    # ── 4. Resolve dependency graph ───────────────────────────────────────────
    console.print("\n[cyan]4. Resolving definition dependencies...[/cyan]")
    from src.graph.dependency_resolver import resolve_definitions

    for i, defs in enumerate(all_defs):
        resolved = resolve_definitions(defs)
        nested   = [r for r in resolved.values() if r.depth > 0]
        console.print(f"  Contract {i+1}: {len(resolved)} resolved | {len(nested)} nested")
        for r in list(nested)[:2]:
            console.print(f"    [yellow]{r.term}[/yellow] (depth={r.depth}) deps: {r.dependencies}")

    # ── 5. Chunk + enrich ─────────────────────────────────────────────────────
    console.print("\n[cyan]5. Chunking + injecting definitions before embedding...[/cyan]")
    from src.pipeline.chunker import chunk_contract
    from src.pipeline.enricher import enrich_chunks

    all_chunks = []
    for i, (p, defs) in enumerate(zip(parsed, all_defs)):
        resolved  = resolve_definitions(defs)
        chunks    = chunk_contract(p)
        enriched  = enrich_chunks(chunks, resolved)
        n_enriched = sum(1 for c in enriched if c.metadata.get("enriched"))
        all_chunks.extend(enriched)
        console.print(f"  Contract {i+1}: {len(enriched)} chunks | {n_enriched} enriched")

        # Show one enriched chunk
        for c in enriched:
            if c.metadata.get("enriched"):
                console.print(f"\n  [bold]Sample enriched chunk:[/bold]")
                console.print(f"  Section: {c.section_heading}")
                console.print(f"  Injected: {c.metadata['injected_terms']}")
                preview = c.enriched_text[:400].replace("\n", " ")
                console.print(f"  [dim]{preview}...[/dim]")
                break

    # ── 6. Embed ──────────────────────────────────────────────────────────────
    console.print("\n[cyan]6. Embedding chunks (local model, no API key needed)...[/cyan]")
    from src.embed.embedder import LegalVectorStore

    store = LegalVectorStore(
        persist_dir=DATA_DIR / "chroma_quickstart",
        provider="local",
    )
    store.add_chunks(all_chunks, collection="enriched")
    store.add_chunks(all_chunks, collection="baseline")
    console.print(f"  Stored: enriched={store.count('enriched')} | baseline={store.count('baseline')}")

    # ── 7. Test retrieval ─────────────────────────────────────────────────────
    console.print("\n[cyan]7. Test retrieval — definition-dependent query...[/cyan]")

    test_query = "What are the obligations of the Indemnified Party following a Material Adverse Effect?"
    console.print(f"  Query: [italic]{test_query}[/italic]\n")

    for collection in ["enriched", "baseline"]:
        hits = store.query(test_query, collection=collection, top_k=3)
        console.print(f"  [{collection.upper()}] Top result:")
        if hits:
            console.print(f"    Section: {hits[0]['metadata'].get('section_heading', 'N/A')}")
            console.print(f"    Score:   {hits[0]['score']:.4f}")
            console.print(f"    Text:    {hits[0]['text'][:150]}...")
        console.print()

    console.rule("[bold green]Quickstart complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  python scripts/01_download_data.py   # download all contracts")
    console.print("  python scripts/02_build_pipeline.py  # process all contracts")
    console.print("  python scripts/03_create_benchmark.py # build DefinitionBench")
    console.print("  python scripts/04_evaluate.py        # run evaluation + get paper numbers")


if __name__ == "__main__":
    main()
