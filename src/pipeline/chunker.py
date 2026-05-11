"""
Structure-aware chunker for legal contracts.

Unlike RecursiveCharacterTextSplitter which blindly splits at token limits,
this chunker respects contract structure:
  - Never splits mid-sentence
  - Respects section boundaries (each section starts a new chunk)
  - Subsections stay together when small enough
  - Falls back to sentence splitting when a section exceeds chunk_size

Every chunk carries metadata:
  - section_heading, section_path (e.g. "Article II > Section 2.3")
  - char_start, char_end
  - contract_id
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str                          # {contract_id}_chunk_{n}
    contract_id: str
    text: str                        # original text (before enrichment)
    enriched_text: str = ""          # text + injected definitions (set by enricher)
    section_heading: str = ""
    section_path: str = ""
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping legal abbreviations intact."""
    # Protect common legal abbreviations from sentence splitting
    abbrevs = ["Inc", "Ltd", "Corp", "Co", "LLP", "LLC", "No", "Art", "Sec",
               "para", "et al", "e.g", "i.e", "vs", "U.S", "Mr", "Ms", "Dr"]
    protected = text
    for abbrev in abbrevs:
        protected = protected.replace(f"{abbrev}.", f"{abbrev}__DOT__")

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    sentences = [s.replace("__DOT__", ".") for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def chunk_contract(
    parsed_contract: dict,
    chunk_size: int = 512,      # approximate word count per chunk
    overlap_sentences: int = 1,  # sentences to repeat at chunk boundaries
) -> list[Chunk]:
    """
    Split a parsed contract into structured chunks.
    Respects section boundaries — this is critical for legal documents.
    """
    sections = parsed_contract.get("sections", [])
    contract_id = parsed_contract["id"]
    chunks: list[Chunk] = []
    chunk_counter = 0

    if not sections:
        # Fallback: treat whole doc as one section
        sections = [{"heading": "DOCUMENT", "text": parsed_contract["raw_text"], "char_start": 0}]

    for section in sections:
        heading = section.get("heading", "")
        text    = section.get("text", "").strip()
        c_start = section.get("char_start", 0)

        if not text:
            continue

        words = text.split()
        if len(words) <= chunk_size:
            # Section fits in one chunk
            chunks.append(Chunk(
                id=f"{contract_id}_chunk_{chunk_counter}",
                contract_id=contract_id,
                text=text,
                section_heading=heading,
                char_start=c_start,
                char_end=c_start + len(text),
            ))
            chunk_counter += 1
        else:
            # Section too large — split by sentences, respecting chunk_size
            sentences = _split_into_sentences(text)
            current_sentences = []
            current_word_count = 0

            for i, sentence in enumerate(sentences):
                sentence_words = len(sentence.split())

                if current_word_count + sentence_words > chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(Chunk(
                        id=f"{contract_id}_chunk_{chunk_counter}",
                        contract_id=contract_id,
                        text=chunk_text,
                        section_heading=heading,
                        char_start=c_start,
                        char_end=c_start + len(chunk_text),
                    ))
                    chunk_counter += 1

                    # Overlap: carry last N sentences into next chunk
                    current_sentences = current_sentences[-overlap_sentences:]
                    current_word_count = sum(len(s.split()) for s in current_sentences)

                current_sentences.append(sentence)
                current_word_count += sentence_words

            # Flush remaining sentences
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    id=f"{contract_id}_chunk_{chunk_counter}",
                    contract_id=contract_id,
                    text=chunk_text,
                    section_heading=heading,
                    char_start=c_start,
                    char_end=c_start + len(chunk_text),
                ))
                chunk_counter += 1

    return chunks
