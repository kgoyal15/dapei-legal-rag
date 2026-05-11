"""
Baseline 4: Post-retrieval definition injection.

This is the closest prior art (the Medium prototype approach).
Chunks are embedded WITHOUT definitions (standard chunking).
Definitions are injected AFTER retrieval, before the LLM call.

We compare this to our approach (pre-embedding injection) to prove
that injecting BEFORE embedding produces better retrieval quality.

The hypothesis: post-retrieval injection helps the LLM read correctly
but does NOT fix the retrieval step — you still retrieve the wrong chunks
because their embeddings were computed without definition context.
"""

from src.pipeline.chunker import Chunk, chunk_contract
from src.graph.dependency_resolver import ResolvedDefinition, get_terms_in_text


def post_retrieval_chunk(parsed_contract: dict, chunk_size: int = 512) -> list[Chunk]:
    """Standard chunking — no enrichment at embedding time."""
    chunks = chunk_contract(parsed_contract, chunk_size=chunk_size)
    for chunk in chunks:
        chunk.enriched_text = chunk.text  # embed plain text
    return chunks


def inject_definitions_post_retrieval(
    retrieved_chunks: list[dict],
    resolved_defs: dict[str, ResolvedDefinition],
    max_words: int = 300,
) -> list[dict]:
    """
    Called after retrieval — inject definitions into the retrieved chunks
    before passing to the LLM.
    This is what the existing prototype does.
    """
    known_terms = set(resolved_defs.keys())
    enriched = []

    for chunk in retrieved_chunks:
        text = chunk["text"]
        relevant_terms = get_terms_in_text(text, known_terms)

        if not relevant_terms:
            enriched.append({**chunk, "context_text": text})
            continue

        def_parts = []
        words_used = 0
        for term in relevant_terms:
            if term not in resolved_defs:
                continue
            rd = resolved_defs[term]
            entry = f'"{rd.term}": {rd.raw_definition[:150]}'
            if words_used + len(entry.split()) > max_words:
                break
            def_parts.append(entry)
            words_used += len(entry.split())

        if def_parts:
            context = "[DEFINITIONS: " + "; ".join(def_parts) + "]\n" + text
        else:
            context = text

        enriched.append({**chunk, "context_text": context})

    return enriched
