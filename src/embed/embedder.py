"""
Embedder — computes embeddings and stores them in ChromaDB.

Two embedding options:
  1. OpenAI text-embedding-3-small  (best quality, costs ~$0.02/1M tokens)
  2. Local sentence-transformers    (free, slightly lower quality, no API key needed)
     Model: "BAAI/bge-small-en-v1.5" — best free model for retrieval tasks

ChromaDB runs locally (no server needed) — data stored in ./data/chroma/
"""

import os
from pathlib import Path
from typing import Literal
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

from src.pipeline.chunker import Chunk


EmbeddingProvider = Literal["openai", "local"]


def get_embedding_function(provider: EmbeddingProvider = "local", model: str = ""):
    """Return a ChromaDB-compatible embedding function."""
    if provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        from config import OPENAI_API_KEY
        model = model or "text-embedding-3-small"
        return OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=model)

    # Local: sentence-transformers (no cost, no key)
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    model = model or "BAAI/bge-small-en-v1.5"
    return SentenceTransformerEmbeddingFunction(model_name=model)


class LegalVectorStore:
    """
    Wraps ChromaDB. Manages two collections per contract set:
      - enriched:  chunks with injected definitions (our method)
      - baseline:  chunks without enrichment (for comparison)
    """

    def __init__(
        self,
        persist_dir: Path,
        provider: EmbeddingProvider = "local",
        embedding_model: str = "",
    ):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_fn = get_embedding_function(provider, embedding_model)

        self.enriched_col  = self._get_or_create("enriched_chunks")
        self.baseline_col  = self._get_or_create("baseline_chunks")

    def _get_or_create(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        collection: Literal["enriched", "baseline"] = "enriched",
        batch_size: int = 64,
    ) -> None:
        """
        Add chunks to the vector store.
        For 'enriched': embeds chunk.enriched_text (with injected definitions)
        For 'baseline': embeds chunk.text (plain text, no enrichment)
        """
        col = self.enriched_col if collection == "enriched" else self.baseline_col

        for i in tqdm(range(0, len(chunks), batch_size),
                      desc=f"Embedding ({collection})", unit="batch"):
            batch = chunks[i:i + batch_size]

            ids       = [c.id for c in batch]
            documents = [
                c.enriched_text if collection == "enriched" and c.enriched_text
                else c.text
                for c in batch
            ]
            metadatas = [
                {
                    "contract_id":      c.contract_id,
                    "section_heading":  c.section_heading,
                    "original_text":    c.text[:1000],  # store original for retrieval
                    "enriched":         str(c.metadata.get("enriched", False)),
                    "injected_terms":   ",".join(c.metadata.get("injected_terms", [])),
                    "char_start":       c.char_start,
                    "char_end":         c.char_end,
                }
                for c in batch
            ]

            # ChromaDB deduplicates by id — skip already-indexed chunks
            existing = set(col.get(ids=ids)["ids"])
            new_items = [(id_, doc, meta) for id_, doc, meta in
                         zip(ids, documents, metadatas) if id_ not in existing]
            if not new_items:
                continue

            n_ids, n_docs, n_metas = zip(*new_items)
            col.add(ids=list(n_ids), documents=list(n_docs), metadatas=list(n_metas))

    def query(
        self,
        query_text: str,
        collection: Literal["enriched", "baseline"] = "enriched",
        top_k: int = 10,
        contract_id: str | None = None,
    ) -> list[dict]:
        """
        Retrieve top_k most relevant chunks.
        Returns list of {id, text, metadata, distance} dicts.
        """
        col = self.enriched_col if collection == "enriched" else self.baseline_col

        where = {"contract_id": contract_id} if contract_id else None

        results = col.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            hits.append({
                "id":       doc_id,
                "text":     meta.get("original_text", results["documents"][0][i]),
                "metadata": meta,
                "distance": results["distances"][0][i],
                "score":    1 - results["distances"][0][i],  # cosine similarity
            })

        return hits

    def count(self, collection: Literal["enriched", "baseline"] = "enriched") -> int:
        col = self.enriched_col if collection == "enriched" else self.baseline_col
        return col.count()
