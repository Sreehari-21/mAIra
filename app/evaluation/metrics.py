from __future__ import annotations

import re
from typing import List, Set


class RetrievalMetrics:
    """Computes retrieval quality metrics given retrieved document IDs and ground-truth relevant IDs."""

    @staticmethod
    def hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Computes Hit@K: 1.0 if any ground truth document is in retrieved list, else 0.0."""
        if not ground_truth_ids:
            return 1.0 if not retrieved_ids else 0.0
        gt_set = set(ground_truth_ids)
        return 1.0 if any(r_id in gt_set for r_id in retrieved_ids) else 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Computes MRR: 1 / rank of first relevant retrieved document."""
        if not ground_truth_ids:
            return 1.0 if not retrieved_ids else 0.0
        gt_set = set(ground_truth_ids)
        for rank, r_id in enumerate(retrieved_ids, start=1):
            if r_id in gt_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        """Computes Precision@K: relevant retrieved documents / k."""
        if k <= 0 or not retrieved_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        gt_set = set(ground_truth_ids)
        hits = sum(1 for item in top_k if item in gt_set)
        return hits / len(top_k)


class GenerationMetrics:
    """Computes generation quality metrics: Groundedness and lexical overlap."""

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        # Filter basic stop words for cleaner overlap calculation
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with"
        }
        return {w for w in words if w not in stopwords and len(w) > 1}

    @classmethod
    def groundedness_score(cls, answer: str, context: str) -> float:
        """Calculates token-level grounding ratio: fraction of meaningful answer tokens present in context."""
        ans_tokens = cls._tokenize(answer)
        if not ans_tokens:
            return 1.0  # Empty answer has no ungrounded claims

        ctx_tokens = cls._tokenize(context)
        overlap = ans_tokens.intersection(ctx_tokens)
        return round(len(overlap) / len(ans_tokens), 3)

    @classmethod
    def token_overlap_f1(cls, prediction: str, ground_truth: str) -> float:
        """Calculates word-level F1 overlap score between predicted answer and reference ground truth."""
        pred_tokens = cls._tokenize(prediction)
        gt_tokens = cls._tokenize(ground_truth)

        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0

        common = pred_tokens.intersection(gt_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return round((2 * precision * recall) / (precision + recall), 3)

