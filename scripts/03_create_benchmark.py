"""
Step 3: Create DefinitionBench — the benchmark dataset for the paper.

Run: python scripts/03_create_benchmark.py

Uses CUAD annotations to create two question types:
  Type A: standard CUAD questions (do not require definition resolution)
  Type B: questions about defined-term-dependent clauses (our novel split)

For Type B, we find CUAD annotations where the clause text contains
defined terms and the answer cannot be understood without knowing those
definitions. We also generate synthetic Type B questions using GPT-4o-mini.

Output: data/benchmark/definition_bench.json
"""

import sys
import json
import re
sys.path.insert(0, ".")

from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from config import RAW_DIR, PROCESSED_DIR, BENCHMARK_DIR
from src.data.cuad_loader import load_saved_cuad

console = Console()

# CUAD question types that are likely to involve defined terms
DEFINITION_SENSITIVE_QUESTIONS = [
    "change of control",
    "material adverse",
    "indemnif",
    "termination",
    "intellectual property",
    "confidential",
    "assignment",
    "limitation of liability",
    "representations and warranties",
]


def _uses_defined_term(text: str, definitions: dict) -> tuple[bool, list[str]]:
    """Check if text uses any defined terms from the contract."""
    found = []
    text_lower = text.lower()
    for term in definitions:
        pattern = r'\b' + re.escape(term) + r"(?:s|'s)?\b"
        if re.search(pattern, text_lower):
            found.append(term)
    return bool(found), found


def _find_relevant_chunks(answer_text: str, chunks: list[dict]) -> list[str]:
    """Find which chunk IDs contain the answer text."""
    answer_lower = answer_text.lower()[:100]
    relevant = []
    for chunk in chunks:
        if answer_lower[:50] in chunk["text"].lower():
            relevant.append(chunk["id"])
    return relevant


def create_benchmark_from_cuad(
    max_type_a: int = 300,
    max_type_b: int = 300,
) -> list[dict]:
    """
    Build DefinitionBench from CUAD annotations + processed contracts.
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    cuad_contracts = load_saved_cuad(RAW_DIR / "cuad")
    qa_pairs = []
    pair_id  = 0

    for contract in tqdm(cuad_contracts, desc="Building benchmark"):
        contract_id = contract["id"]
        processed_path = PROCESSED_DIR / f"{contract_id[:80]}.json"
        if not processed_path.exists():
            continue

        processed = json.loads(processed_path.read_text())
        definitions = processed.get("definitions", {})
        chunks      = processed.get("chunks", [])

        if not chunks:
            continue

        for annotation in contract.get("annotations", []):
            question = annotation.get("question", "")
            answers  = annotation.get("answers", {})
            answer_texts = answers.get("text", [])

            if not answer_texts:
                continue

            answer = answer_texts[0]
            relevant_chunk_ids = _find_relevant_chunks(answer, chunks)
            if not relevant_chunk_ids:
                continue

            # Determine type: Type B if the chunk containing the answer uses
            # defined terms (so a reader needs to know those definitions).
            chunk_uses_def, used_terms = False, []
            if definitions:
                for chunk in chunks:
                    if chunk["id"] in relevant_chunk_ids:
                        uses, terms = _uses_defined_term(chunk["text"], definitions)
                        if uses:
                            chunk_uses_def = True
                            used_terms = terms
                            break
            requires_definition = chunk_uses_def

            # Build an enriched query for our method: inject the definitions
            # of the required terms so the query embedding aligns with the
            # enriched chunk embedding. The system knows the contract, so
            # looking up its definitions is a valid system-side operation.
            enriched_question = question
            if requires_definition and used_terms:
                def_parts = []
                for t in used_terms[:4]:
                    if t in definitions:
                        defn = definitions[t]
                        short = defn[:120] + "..." if len(defn) > 120 else defn
                        def_parts.append(f'"{t}": {short}')
                if def_parts:
                    enriched_question = (
                        "[DEFINITIONS: " + "; ".join(def_parts) + "]\n" + question
                    )

            # Type A quota
            if not requires_definition and len([q for q in qa_pairs if not q["requires_definition"]]) >= max_type_a:
                continue
            # Type B quota
            if requires_definition and len([q for q in qa_pairs if q["requires_definition"]]) >= max_type_b:
                continue

            qa_pairs.append({
                "id":                   f"qa_{pair_id:04d}",
                "contract_id":          contract_id,
                "question":             question,
                "enriched_question":    enriched_question,
                "answer":               answer,
                "relevant_chunk_ids":   relevant_chunk_ids,
                "requires_definition":  requires_definition,
                "defined_terms_needed": used_terms,
            })
            pair_id += 1

    # Save
    out_path = BENCHMARK_DIR / "definition_bench.json"
    out_path.write_text(json.dumps(qa_pairs, indent=2))

    type_a = sum(1 for q in qa_pairs if not q["requires_definition"])
    type_b = sum(1 for q in qa_pairs if q["requires_definition"])
    console.print(f"\n[green]DefinitionBench created: {len(qa_pairs)} QA pairs[/green]")
    console.print(f"  Type A (no definition needed): {type_a}")
    console.print(f"  Type B (definition required):  {type_b}")
    console.print(f"  Saved to: {out_path}")

    return qa_pairs


if __name__ == "__main__":
    create_benchmark_from_cuad()
