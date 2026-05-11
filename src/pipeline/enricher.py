"""
Chunk Enricher — pre-embedding definition injection.

THIS IS THE CORE NOVEL CONTRIBUTION.

For each chunk, before it is embedded:
  1. Scan chunk text for defined terms
  2. Look up their fully-resolved definitions (with nested expansions)
  3. Prepend a "DEFINITIONS CONTEXT" block to the chunk text
  4. The embedding is then computed on the enriched text

This means the vector representation carries semantic signal about what
defined terms mean — not just that they appear.

Example:
  Original chunk:  "The Indemnified Party shall notify within 30 days of
                    a Material Adverse Effect..."

  Enriched chunk:  "[DEFINITIONS: Indemnified Party: the party designated
                    to receive indemnification under Section 8.1;
                    Material Adverse Effect: any event, change or condition
                    that has a material adverse effect on the Business
                    [where: "Business": the operations and assets of the
                    Company and its Subsidiaries]]
                    The Indemnified Party shall notify within 30 days of
                    a Material Adverse Effect..."

The enriched text is used ONLY for embedding — the original chunk text
is stored separately and returned to the LLM at query time.
"""

import re
from src.pipeline.chunker import Chunk
from src.graph.dependency_resolver import ResolvedDefinition, get_terms_in_text


def enrich_chunks(
    chunks: list[Chunk],
    resolved_defs: dict[str, ResolvedDefinition],
    max_enrichment_tokens: int = 256,
) -> list[Chunk]:
    """
    Inject definitions into each chunk before embedding.

    Args:
        chunks: output of chunker.py
        resolved_defs: output of dependency_resolver.resolve_definitions()
        max_enrichment_tokens: approx word budget for injected definitions
                               (keeps chunks from growing too large for embedding)

    Returns:
        Same chunks with enriched_text populated.
    """
    known_terms = set(resolved_defs.keys())

    for chunk in chunks:
        relevant_terms = get_terms_in_text(chunk.text, known_terms)

        if not relevant_terms:
            chunk.enriched_text = chunk.text
            chunk.metadata["enriched"] = False
            chunk.metadata["injected_terms"] = []
            continue

        # Build definitions block — prioritize by relevance:
        # terms that appear multiple times in the chunk come first
        term_counts = {}
        text_lower = chunk.text.lower()
        for term in relevant_terms:
            pattern = r'\b' + re.escape(term) + r"(?:s|'s|ies)?\b"
            count = len(re.findall(pattern, text_lower))
            term_counts[term] = count

        sorted_terms = sorted(relevant_terms, key=lambda t: -term_counts.get(t, 0))

        # Build the definitions block within token budget
        def_parts = []
        words_used = 0

        for term in sorted_terms:
            if term not in resolved_defs:
                continue
            rd = resolved_defs[term]
            # Use resolved definition (includes nested expansions)
            def_text = rd.resolved_definition
            def_entry = f'"{rd.term}": {def_text}'
            entry_words = len(def_entry.split())

            if words_used + entry_words > max_enrichment_tokens:
                # Still include the raw definition if resolved is too long
                short_def = rd.raw_definition[:150] + "..." if len(rd.raw_definition) > 150 else rd.raw_definition
                short_entry = f'"{rd.term}": {short_def}'
                if words_used + len(short_entry.split()) <= max_enrichment_tokens:
                    def_parts.append(short_entry)
                    words_used += len(short_entry.split())
                break

            def_parts.append(def_entry)
            words_used += entry_words

        if def_parts:
            definitions_block = "[DEFINITIONS: " + "; ".join(def_parts) + "]\n"
            chunk.enriched_text = definitions_block + chunk.text
        else:
            chunk.enriched_text = chunk.text

        chunk.metadata["enriched"] = bool(def_parts)
        chunk.metadata["injected_terms"] = [t for t in sorted_terms if t in resolved_defs]

    return chunks


def enrich_query(
    query: str,
    resolved_defs: dict[str, ResolvedDefinition],
    max_enrichment_tokens: int = 128,
) -> str:
    """
    Also enrich the query at retrieval time.
    If the query contains defined terms, prepend their definitions.
    This aligns the query embedding with enriched chunk embeddings.
    """
    known_terms = set(resolved_defs.keys())
    relevant_terms = get_terms_in_text(query, known_terms)

    if not relevant_terms:
        return query

    def_parts = []
    words_used = 0
    for term in relevant_terms:
        if term not in resolved_defs:
            continue
        rd = resolved_defs[term]
        short_def = rd.raw_definition[:100] + "..." if len(rd.raw_definition) > 100 else rd.raw_definition
        entry = f'"{rd.term}": {short_def}'
        if words_used + len(entry.split()) > max_enrichment_tokens:
            break
        def_parts.append(entry)
        words_used += len(entry.split())

    if def_parts:
        return "[DEFINITIONS: " + "; ".join(def_parts) + "]\n" + query
    return query
