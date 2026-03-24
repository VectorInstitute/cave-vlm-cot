"""
evaluations.py
--------------
Central evaluation module for the CaVe-VLM-CoT pipeline.
Aggregates all per-component eval functions so every module can do:
    from evaluations import recall_at_k, compute_cave_score, ...
"""

# Re-export everything from each component's eval module so existing
# `from evaluations import X` calls across the codebase continue to work.
from extractor.planner_evals import (
    planner_coverage_score,
    planner_hit_rate,
    planner_specificity_score,
)
from retriever.retriever_evals import (
    answer_support_recall,
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from solver.solver_evals import (
    get_nli_model,
    extract_citation_claim_pairs,
    final_answer_accuracy,
    get_text_evidence_by_id,
    text_citation_precision,
    question_image_citation_precision,
    question_image_citation_coverage,
    attribution_score,
    evidence_grounding_check,
    compute_cave_score,
    compute_combined_metrics,
    citation_summary,
)
from verifier.verifier_evals import (
    evaluate_verifier_quality,
    evaluate_feedback_quality,
)

__all__ = [
    # Planner
    "planner_coverage_score",
    "planner_hit_rate",
    "planner_specificity_score",
    # Retriever
    "answer_support_recall",
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    # Solver
    "get_nli_model",
    "extract_citation_claim_pairs",
    "final_answer_accuracy",
    "get_text_evidence_by_id",
    "text_citation_precision",
    "question_image_citation_precision",
    "question_image_citation_coverage",
    "attribution_score",
    "evidence_grounding_check",
    "compute_cave_score",
    "compute_combined_metrics",
    "citation_summary",
    # Verifier
    "evaluate_verifier_quality",
    "evaluate_feedback_quality",
]