from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.evaluation.metrics import GenerationMetrics, RetrievalMetrics
from app.services.retrieval_service import RetrievalService


class EvaluationService:
    """Evaluates RAG retrieval and generation performance using structured benchmarks."""

    def __init__(self, retrieval_service: Optional[RetrievalService] = None) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()

    def evaluate_retrieval(
        self,
        test_queries: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> Dict[str, float]:
        """Evaluates retrieval hit rate and MRR against reference ground truth sources."""
        hit_rates: List[float] = []
        mrrs: List[float] = []

        for item in test_queries:
            query = item["query"]
            expected_sources = item.get("expected_sources", [])
            if not expected_sources:
                continue

            result = self.retrieval_service.retrieve(query=query, top_k=top_k)
            retrieved_files = [c.filename for c in result.chunks]

            hr = RetrievalMetrics.hit_rate(retrieved_files, expected_sources)
            mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_files, expected_sources)

            hit_rates.append(hr)
            mrrs.append(mrr)

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

        return {
            "mean_hit_rate": round(avg_hit_rate, 3),
            "mrr": round(avg_mrr, 3),
            "queries_evaluated": len(hit_rates),
        }

    def evaluate_generation(
        self,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluates single response grounding and reference F1 score."""
        grounding = GenerationMetrics.groundedness_score(answer=answer, context=context)
        metrics = {"groundedness_score": grounding}

        if ground_truth:
            metrics["f1_overlap"] = GenerationMetrics.token_overlap_f1(
                prediction=answer,
                ground_truth=ground_truth,
            )

        return metrics

    def load_benchmark_cases(self, file_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Loads reference benchmark cases from JSON test suite."""
        path = (
            file_path
            or Path(__file__).resolve().parent.parent / "evaluation" / "test_cases" / "benchmark_cases.json"
        )
        if not path.exists():
            logger.warning(f"Benchmark file not found: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

