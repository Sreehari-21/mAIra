from __future__ import annotations

from app.evaluation.metrics import GenerationMetrics, RetrievalMetrics


def test_retrieval_hit_rate():
    retrieved = ["doc_a", "doc_b", "doc_c"]
    ground_truth = ["doc_b"]
    assert RetrievalMetrics.hit_rate(retrieved, ground_truth) == 1.0

    not_found_truth = ["doc_z"]
    assert RetrievalMetrics.hit_rate(retrieved, not_found_truth) == 0.0


def test_retrieval_mrr():
    retrieved = ["doc_a", "doc_b", "doc_c"]
    # doc_b is at rank 2 -> MRR = 0.5
    assert RetrievalMetrics.mean_reciprocal_rank(retrieved, ["doc_b"]) == 0.5
    # doc_a is at rank 1 -> MRR = 1.0
    assert RetrievalMetrics.mean_reciprocal_rank(retrieved, ["doc_a"]) == 1.0


def test_generation_groundedness():
    context = "The quick brown fox jumps over the lazy dog."
    good_answer = "A brown fox jumped over the dog."
    score = GenerationMetrics.groundedness_score(good_answer, context)
    assert score > 0.6

    hallucinated_answer = "Quantum entanglement produces relativistic gravity."
    score_bad = GenerationMetrics.groundedness_score(hallucinated_answer, context)
    assert score_bad == 0.0

