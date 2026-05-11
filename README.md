# dapei-legal-rag
# DAPEI: Definition-Aware Pre-Embedding Injection for Legal Contract Retrieval

**Paper:** Closing the Definition Dependency Gap: Pre-Embedding Definition Injection for Legal Contract Retrieval  
**Author:** Kamal Goyal — Kaytech Software Canada Inc.  
**Preprint:** [Zenodo — add your DOI link here]  ] (https://doi.org/10.5281/zenodo.20126274)
**arxiv:** [add link after submission]

---

## The Problem

Standard RAG pipelines fail on legal contracts in a specific, repeatable way.

A clause on page 47 reading *"the Indemnified Party shall give written notice within 30 days of any Triggering Event"* gets embedded without knowing what **Indemnified Party** or **Triggering Event** mean. Those definitions sit on pages 4 and 6. The embedding model treats them as ordinary noun phrases. The resulting vector matches queries about notification procedures — not indemnification. Retrieval fails silently.

I call this the **Definition Dependency Gap (DDG)**: clauses embedded without their governing definitions produce semantically weak vectors that mismatch user intent.

## The Solution

**DAPEI** (Definition-Aware Pre-Embedding Injection) resolves this in three steps:

1. **Extract** — 3-layer cascade (regex → structural → LLM fallback) pulls every defined term from the contract
2. **Resolve** — a dependency graph topologically sorts nested terms so that a term whose definition references another defined term gets the full resolved meaning (bounded at depth 3)
3. **Inject** — resolved definitions are prepended to each chunk *before* embedding within a 256-token budget, so the vector represents what the clause *means* — not just what it *says*

The original chunk text is preserved separately and returned to the LLM at generation time. Only the embedding sees the enriched text.

---

## Quick Start

No API key needed — uses a free local embedding model.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python quickstart.py
```

This runs the full pipeline on 3 CUAD contracts end-to-end in under 2 minutes.

---

## Full Pipeline

```bash
# 1. Download 511 CUAD contracts + 200 SEC EDGAR contracts
python scripts/01_download_data.py

# 2. Extract definitions, resolve dependency graph, enrich chunks, embed
python scripts/02_build_pipeline.py

# 3. Build DefinitionBench (Type A / Type B query split)
python scripts/03_create_benchmark.py

# 4. Run evaluation — produces the paper's main results table
python scripts/04_evaluate.py
```

---

## Results

Evaluation on DefinitionBench. **Type B** = queries that require definition resolution to answer correctly.

| Method                        | Type A P@5 | Type A MRR | Type B P@5 | Type B MRR | Type B R@10 |
|-------------------------------|-----------|-----------|-----------|-----------|------------|
| Naive chunking (baseline)     | —         | —         | —         | —         | —          |
| Pre-embed injection (DAPEI)   | —         | —         | —         | —         | —          |

*Results will be updated once the full pipeline run completes.*

DAPEI is designed to match naive on Type A (queries that do not involve defined terms) while substantially outperforming all baselines on Type B.

---

## Architecture

```
Contract PDF
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  3-Layer Definition Extractor                           │
│  Layer 1: Regex (6 patterns — quoted-term-means, etc.)  │
│  Layer 2: Structural (Definitions section parsing)      │
│  Layer 3: GPT-4o-mini fallback (when < 3 found)         │
└────────────────────┬────────────────────────────────────┘
                     │  dict[term → definition]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Dependency Graph Resolver                              │
│  Builds G=(V,E): A→B means A's definition uses B        │
│  Topological sort → resolved glossary (max depth 3)     │
└────────────────────┬────────────────────────────────────┘
                     │  dict[term → ResolvedDefinition]
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Structure-Aware Chunker + Enricher                     │
│  Respects section boundaries (never splits mid-clause)  │
│  Injects definitions into enriched_text before embed    │
│  256-token budget — ranked by term frequency in chunk   │
└────────────────────┬────────────────────────────────────┘
                     │  Chunk(text, enriched_text, metadata)
                     ▼
┌─────────────────────────────────────────────────────────┐
│  ChromaDB Vector Store                                  │
│  enriched collection  ← embeds enriched_text (DAPEI)   │
│  baseline collection  ← embeds plain text (naive)      │
│  original text stored in metadata → returned to LLM    │
└─────────────────────────────────────────────────────────┘
```

---

## DefinitionBench

A benchmark built on CUAD contracts that splits queries into two types:

- **Type A** — answerable without definition resolution (control group)
- **Type B** — requires correct definition resolution to retrieve the right clause (treatment group)

This split is what makes it possible to measure the DDG's cost in retrieval quality precisely. Without it, definition-dependent failures average out and are invisible in aggregate metrics.

---

## Dataset

- **CUAD** — 511 annotated commercial contracts ([HuggingFace](https://huggingface.co/datasets/theatticusproject/cuad))
- **SEC EDGAR** — 200 material contracts (EX-10.x exhibits, public domain)

---

## Requirements

- Python 3.10+
- See `requirements.txt`
- Optional: OpenAI API key for LLM-based definition extraction fallback and faithfulness scoring (set `OPENAI_API_KEY` in `.env`)

---

## Project Structure

```
legal-rag-research/
├── src/
│   ├── data/          # CUAD loader, EDGAR downloader, contract parser
│   ├── extract/       # 3-layer definition extractor
│   ├── graph/         # dependency resolver
│   ├── pipeline/      # structure-aware chunker + enricher
│   ├── embed/         # ChromaDB vector store wrapper
│   └── eval/          # DefinitionBench metrics + benchmark runner
├── baselines/         # naive, SAC, post-retrieval injection
├── scripts/           # 01–04 pipeline steps
├── quickstart.py      # end-to-end smoke test (3 contracts)
├── config.py          # paths, model settings, API keys
└── generate_preprint.py  # regenerates DAPEI_preprint.pdf
```

---

## Citation

```bibtex
@misc{goyal2025dapei,
  title   = {Closing the Definition Dependency Gap: Pre-Embedding Definition
             Injection for Legal Contract Retrieval},
  author  = {Goyal, Kamal},
  year    = {2025},
  publisher = {Zenodo},
  doi     = {add your Zenodo DOI here},
  url     = {https://doi.org/add-your-doi-here}
}
```

---

## License

Code: MIT  
Data (CUAD): [CUAD License](https://huggingface.co/datasets/theatticusproject/cuad)
