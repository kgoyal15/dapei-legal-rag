"""
Definition Extractor — the core innovation.

Extracts (term → definition) pairs from legal contracts using a
3-layer approach:

  Layer 1 — Regex:   Fast, handles the 80% common patterns.
  Layer 2 — NLP:     spaCy NER + heuristics for trickier patterns.
  Layer 3 — LLM:     GPT-4o-mini fallback for complex / ambiguous cases.

Extracted definitions are used by the enricher to inject context into
chunks BEFORE embedding, so the vector representation carries semantic
signal about what defined terms actually mean.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Definition:
    term: str
    definition: str
    source: str            # "regex" | "nlp" | "llm"
    char_start: int = 0
    confidence: float = 1.0

    def __hash__(self):
        return hash(self.term.lower())

    def __eq__(self, other):
        return self.term.lower() == other.term.lower()


# ── Layer 1: Regex patterns ──────────────────────────────────────────────────
#
# Legal contracts use several consistent syntactic patterns for definitions.
# Order matters: more specific patterns first.

_REGEX_PATTERNS = [
    # "Material Adverse Effect" means any change...
    re.compile(
        r'"(?P<term>[A-Z][^"]{1,80})"\s+(?:means|shall mean|has the meaning|refers to|is defined as|shall have the meaning)\s+(?P<def>[^"]{10,600}?)(?=\n|\"|;|\.\s+[A-Z])',
        re.DOTALL,
    ),
    # 'Material Adverse Effect' means any change...  (single quotes)
    re.compile(
        r"'(?P<term>[A-Z][^']{1,80})'\s+(?:means|shall mean|has the meaning)\s+(?P<def>[^']{10,600}?)(?=\n|'|;|\.\s+[A-Z])",
        re.DOTALL,
    ),
    # Material Adverse Effect means any change... (no quotes, at start of line or after number)
    re.compile(
        r"(?:^|\n)\s*(?:\d+\.\s+)?(?P<term>[A-Z][A-Za-z\s]{2,60})\s+means\s+(?P<def>.{10,600}?)(?=\n\n|\n[A-Z\d]|;)",
        re.DOTALL,
    ),
    # Inline: (the "Term") or (as defined herein, the "Term")
    # These define terms parenthetically in the body. Capture what precedes the parens.
    re.compile(
        r'(?P<def>[^.]{10,300})\s+\((?:the\s+|hereinafter\s+)?"(?P<term>[A-Z][^"]{1,60})"\)',
        re.DOTALL,
    ),
    # "Term" (as used herein) means ...
    re.compile(
        r'"(?P<term>[A-Z][^"]{1,80})"\s+\([^)]+\)\s+means\s+(?P<def>[^"]{10,400}?)(?=\n|;|\.\s+[A-Z])',
        re.DOTALL,
    ),
    # As used in this Agreement, "Term" means ...
    re.compile(
        r'[Aa]s used (?:in this [A-Za-z]+|herein),\s+"(?P<term>[A-Z][^"]{1,80})"\s+means\s+(?P<def>[^"]{10,400}?)(?=\n|;|\.\s+[A-Z])',
        re.DOTALL,
    ),
]


def extract_via_regex(text: str) -> list[Definition]:
    defs = []
    seen_terms = set()

    for pattern in _REGEX_PATTERNS:
        for m in pattern.finditer(text):
            term = m.group("term").strip()
            defn = m.group("def").strip()

            # Basic quality filters
            if len(term) < 2 or len(term) > 100:
                continue
            if len(defn) < 5:
                continue
            if term.lower() in seen_terms:
                continue

            # Clean up definition text
            defn = re.sub(r"\s+", " ", defn).strip(" .;,")

            seen_terms.add(term.lower())
            defs.append(Definition(
                term=term,
                definition=defn,
                source="regex",
                char_start=m.start(),
                confidence=0.95,
            ))

    return defs


# ── Layer 2: Definitions-section focused extraction ──────────────────────────

def extract_from_definitions_section(section_text: str) -> list[Definition]:
    """
    When we know we're inside a definitions section, we can be more aggressive.
    Look for any paragraph that starts with a capitalized term.
    """
    defs = []
    seen_terms = set()

    # Split into paragraphs
    paragraphs = re.split(r"\n{2,}", section_text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Try regex first
        regex_hits = extract_via_regex(para)
        if regex_hits:
            for d in regex_hits:
                if d.term.lower() not in seen_terms:
                    seen_terms.add(d.term.lower())
                    defs.append(d)
            continue

        # Fallback: paragraph starts with quoted term
        m = re.match(r'^["\']?([A-Z][A-Za-z\s\-]{1,60})["\']?\s+(?:means|shall mean|refers)', para)
        if m:
            term = m.group(1).strip().strip('"\'')
            rest = para[m.end():].strip()
            if len(rest) > 10 and term.lower() not in seen_terms:
                seen_terms.add(term.lower())
                defs.append(Definition(
                    term=term,
                    definition=rest[:500],
                    source="nlp",
                    confidence=0.80,
                ))

    return defs


# ── Layer 3: LLM fallback ────────────────────────────────────────────────────

_LLM_PROMPT = """You are extracting defined terms from a legal contract section.

Return a JSON array of objects with keys "term" and "definition".
Only include terms that are explicitly defined in the text (not terms that are merely used).
If no definitions are found, return an empty array [].

Contract section:
---
{text}
---

JSON output:"""


def extract_via_llm(section_text: str, llm_model: str = "gpt-4o-mini") -> list[Definition]:
    """
    Use LLM to extract definitions that regex missed.
    Called only when regex finds < 2 definitions in a section we know contains them.
    """
    try:
        from openai import OpenAI
        from config import OPENAI_API_KEY
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"LLM extraction unavailable: {e}")
        return []

    # Truncate to avoid token limits
    text_snippet = section_text[:3000]

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(text=text_snippet)}],
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "[]"
        parsed = json.loads(raw)

        # Handle both {"definitions": [...]} and [...]
        if isinstance(parsed, dict):
            items = parsed.get("definitions") or parsed.get("terms") or list(parsed.values())[0]
        else:
            items = parsed

        defs = []
        for item in items:
            term = str(item.get("term", "")).strip()
            defn = str(item.get("definition", "")).strip()
            if term and defn and len(term) < 100:
                defs.append(Definition(
                    term=term,
                    definition=defn[:600],
                    source="llm",
                    confidence=0.85,
                ))
        return defs

    except Exception as e:
        print(f"LLM extraction error: {e}")
        return []


# ── Main extractor ───────────────────────────────────────────────────────────

def extract_definitions(
    parsed_contract: dict,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> dict[str, Definition]:
    """
    Full 3-layer extraction pipeline.

    Args:
        parsed_contract: output of contract_parser.py (has 'sections' + 'raw_text')
        use_llm: whether to call LLM when regex misses definitions
        llm_model: which OpenAI model to use for LLM fallback

    Returns:
        dict mapping lowercase term → Definition object
    """
    from src.data.contract_parser import is_definitions_section

    all_defs: dict[str, Definition] = {}

    # Pass 1: focused extraction from definitions sections
    definitions_sections = [
        s for s in parsed_contract.get("sections", [])
        if is_definitions_section(s["heading"])
    ]

    for section in definitions_sections:
        # Regex pass
        hits = extract_via_regex(section["text"])
        hits += extract_from_definitions_section(section["text"])
        for d in hits:
            key = d.term.lower()
            if key not in all_defs or d.confidence > all_defs[key].confidence:
                all_defs[key] = d

        # LLM fallback if we found very few definitions in a known definitions section
        if use_llm and len(hits) < 3:
            llm_hits = extract_via_llm(section["text"], llm_model)
            for d in llm_hits:
                key = d.term.lower()
                if key not in all_defs:
                    all_defs[key] = d

    # Pass 2: regex sweep over full document (catches inline definitions)
    full_hits = extract_via_regex(parsed_contract.get("raw_text", ""))
    for d in full_hits:
        key = d.term.lower()
        if key not in all_defs:
            all_defs[key] = d

    return all_defs


def definitions_to_dict(defs: dict[str, Definition]) -> dict[str, str]:
    """Convenience: return {term: definition_text} for simple lookups."""
    return {k: v.definition for k, v in defs.items()}
