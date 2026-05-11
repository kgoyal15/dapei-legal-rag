"""
Baseline 3: Summary-Augmented Chunking (SAC).

From arxiv:2510.06999 — prepend a short document-level summary to each chunk.
This is the current best-published approach for legal RAG.
We beat this with our definition injection.
"""

from src.pipeline.chunker import Chunk, chunk_contract


def _generate_summary(text: str, llm_model: str = "gpt-4o-mini") -> str:
    """Generate a short document summary using LLM."""
    try:
        from openai import OpenAI
        from config import OPENAI_API_KEY
        client = OpenAI(api_key=OPENAI_API_KEY)
        snippet = text[:3000]
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{
                "role": "user",
                "content": f"Write a 2-sentence summary of this legal contract. Be factual:\n\n{snippet}"
            }],
            temperature=0,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback: use first 150 chars of document
        return text[:150].strip()


def sac_chunk(
    parsed_contract: dict,
    chunk_size: int = 512,
    use_llm: bool = True,
) -> list[Chunk]:
    """Chunk + prepend document summary to each chunk."""
    chunks = chunk_contract(parsed_contract, chunk_size=chunk_size)

    summary = _generate_summary(parsed_contract["raw_text"]) if use_llm else \
              parsed_contract["raw_text"][:150]

    summary_prefix = f"[DOCUMENT SUMMARY: {summary}]\n"
    for chunk in chunks:
        chunk.enriched_text = summary_prefix + chunk.text

    return chunks
