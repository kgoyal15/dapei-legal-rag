"""
Evaluation metrics for legal RAG retrieval and QA quality.

Retrieval metrics:
  - Precision@k   : fraction of top-k results that are relevant
  - Recall@k      : fraction of all relevant results in top-k
  - MRR           : Mean Reciprocal Rank
  - NDCG@k        : Normalized Discounted Cumulative Gain

QA metrics:
  - Exact Match   : answer == ground truth (normalized)
  - F1            : token overlap between answer and ground truth
  - Faithfulness  : is the answer grounded in retrieved chunks? (LLM judge)
"""

import re
import string
from collections import Counter


# ── Retrieval metrics ────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not retrieved_ids or k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for id_ in top_k if id_ in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for id_ in top_k if id_ in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, id_ in enumerate(retrieved_ids, 1):
        if id_ in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    import math
    def dcg(ids, rel_set, k):
        score = 0.0
        for i, id_ in enumerate(ids[:k], 1):
            if id_ in rel_set:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids, relevant_ids, k)
    # Ideal DCG: all relevant docs at the top
    ideal_retrieved = list(relevant_ids)[:k]
    ideal_dcg = dcg(ideal_retrieved, relevant_ids, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── QA metrics ───────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens   = _normalize(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter   = Counter(gt_tokens)
    common       = pred_counter & gt_counter
    num_common   = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def faithfulness_score(
    answer: str,
    retrieved_chunks: list[str],
    llm_model: str = "gpt-4o-mini",
) -> float:
    """
    LLM judge: is every claim in the answer supported by the retrieved chunks?
    Returns score 0.0–1.0.
    """
    try:
        from openai import OpenAI
        from config import OPENAI_API_KEY
        client = OpenAI(api_key=OPENAI_API_KEY)

        context = "\n---\n".join(retrieved_chunks[:5])
        prompt = f"""You are evaluating whether an answer is grounded in provided context.

Context:
{context}

Answer: {answer}

Is every factual claim in the answer directly supported by the context?
Reply with ONLY a number between 0.0 and 1.0 where:
  1.0 = fully grounded, every claim is in the context
  0.5 = partially grounded
  0.0 = hallucinated, claims not in context

Score:"""

        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        score = float(resp.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


# ── Aggregate ────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    results: list[dict],   # [{retrieved_ids: [...], relevant_ids: set, ...}]
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """Compute mean retrieval metrics across a list of query results."""
    aggregated = {f"P@{k}": [] for k in k_values}
    aggregated.update({f"R@{k}": [] for k in k_values})
    aggregated["MRR"] = []
    aggregated[f"NDCG@{max(k_values)}"] = []

    for r in results:
        ret_ids = r["retrieved_ids"]
        rel_ids = r["relevant_ids"]
        for k in k_values:
            aggregated[f"P@{k}"].append(precision_at_k(ret_ids, rel_ids, k))
            aggregated[f"R@{k}"].append(recall_at_k(ret_ids, rel_ids, k))
        aggregated["MRR"].append(reciprocal_rank(ret_ids, rel_ids))
        aggregated[f"NDCG@{max(k_values)}"].append(ndcg_at_k(ret_ids, rel_ids, max(k_values)))

    return {metric: sum(vals) / len(vals) for metric, vals in aggregated.items() if vals}
