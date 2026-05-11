"""
Step 2: Parse contracts → extract definitions → enrich chunks → embed.

Run: python scripts/02_build_pipeline.py

Processes all contracts in data/raw/ through the full pipeline:
  1. Parse contract text + detect sections
  2. Extract defined terms (3-layer: regex → NLP → LLM)
  3. Build dependency graph + resolve nested terms
  4. Chunk by structure
  5. Inject definitions into chunks BEFORE embedding
  6. Embed enriched chunks into ChromaDB (enriched collection)
  7. Also embed plain chunks (baseline collection) for comparison
"""

import sys
import json
sys.path.insert(0, ".")

from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from config import RAW_DIR, PROCESSED_DIR, DATA_DIR, EMBEDDING_PROVIDER, EMBEDDING_MODEL
from src.data.cuad_loader import load_saved_cuad
from src.data.contract_parser import parse_text_file, parse_cuad_contract
from src.extract.definition_extractor import extract_definitions, definitions_to_dict
from src.graph.dependency_resolver import resolve_definitions
from src.pipeline.chunker import chunk_contract
from src.pipeline.enricher import enrich_chunks
from src.embed.embedder import LegalVectorStore

console = Console()


def process_contract(parsed: dict, use_llm: bool = False) -> dict:
    """Full pipeline for one contract. Returns processing summary."""
    contract_id = parsed["id"]

    # Extract definitions
    raw_defs_obj = extract_definitions(parsed, use_llm=use_llm)
    raw_defs     = definitions_to_dict(raw_defs_obj)

    # Resolve nested dependencies
    resolved = resolve_definitions(raw_defs)

    # Chunk
    chunks = chunk_contract(parsed)

    # Enrich (inject definitions before embedding)
    chunks = enrich_chunks(chunks, resolved)

    # Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "id":            contract_id,
        "source":        parsed["source"],
        "definitions":   raw_defs,
        "resolved_count": len(resolved),
        "chunk_count":   len(chunks),
        "enriched_count": sum(1 for c in chunks if c.metadata.get("enriched")),
        "chunks": [
            {
                "id":             c.id,
                "text":           c.text,
                "enriched_text":  c.enriched_text,
                "section_heading": c.section_heading,
                "injected_terms": c.metadata.get("injected_terms", []),
            }
            for c in chunks
        ],
    }
    (PROCESSED_DIR / f"{contract_id[:80]}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    return out, chunks, resolved


def main(max_contracts: int = 100, use_llm: bool = False):
    console.rule("[bold]Step 2: Building pipeline[/bold]")

    store = LegalVectorStore(
        persist_dir=DATA_DIR / "chroma",
        provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
    )

    # Load CUAD contracts
    cuad_contracts = load_saved_cuad(RAW_DIR / "cuad")
    console.print(f"Found {len(cuad_contracts)} CUAD contracts")

    processed = 0
    errors    = 0

    for raw_contract in tqdm(cuad_contracts[:max_contracts], desc="Processing"):
        try:
            parsed = parse_cuad_contract(raw_contract)
            summary, chunks, resolved = process_contract(parsed, use_llm=use_llm)

            # Add to vector store — both enriched and baseline collections
            store.add_chunks(chunks, collection="enriched")
            store.add_chunks(chunks, collection="baseline")

            console.print(
                f"  {parsed['id'][:40]:40s} | "
                f"defs={len(summary['definitions']):3d} | "
                f"chunks={summary['chunk_count']:4d} | "
                f"enriched={summary['enriched_count']:4d}"
            )
            processed += 1
        except Exception as e:
            console.print(f"[red]Error processing {raw_contract.get('id', '?')}: {e}[/red]")
            errors += 1

    console.print(f"\n[green]Processed: {processed} | Errors: {errors}[/green]")
    console.print(f"Vector store: enriched={store.count('enriched')} baseline={store.count('baseline')} chunks")
    console.rule("[bold green]Done. Run scripts/03_create_benchmark.py next.[/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=100, help="Max contracts to process")
    parser.add_argument("--llm", action="store_true", help="Use LLM for definition extraction fallback")
    args = parser.parse_args()
    main(max_contracts=args.max, use_llm=args.llm)
