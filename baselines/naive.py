"""Baseline 1: Naive recursive character splitting — no enrichment."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.pipeline.chunker import Chunk


def naive_chunk(parsed_contract: dict, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 5,   # approximate chars (~5 chars/word)
        chunk_overlap=overlap * 5,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(parsed_contract["raw_text"])
    chunks = []
    for i, text in enumerate(texts):
        c = Chunk(
            id=f"{parsed_contract['id']}_naive_{i}",
            contract_id=parsed_contract["id"],
            text=text,
        )
        c.enriched_text = text  # no enrichment
        chunks.append(c)
    return chunks
