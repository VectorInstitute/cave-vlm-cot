from typing import Dict


###### VERIFIER EVALUATION
"""
Citation Verification: Does it correctly identify when citations don't support claims?
Hallucination Detection: Does it catch claims not grounded in evidence?
Reasoning Validity: Does it detect logical errors in reasoning steps?
Answer Consistency: Does the final answer follow from the reasoning?
"""


def evaluate_verifier_quality(state, cave_result: dict = None) -> dict:
    """
    Evaluate whether the verifier correctly analyzes the solver output.

    Compares verifier's assessment against our computed ground truth:
    - Citation precision (NLI-based)
    - AIS (hallucination detection)
    - Answer correctness

    Parameters
    ----------
    state : State
    cave_result : dict, optional
        Pre-computed result from compute_cave_score().  If supplied the
        expensive NLI / AIS inference is skipped; otherwise it is run here.
        Pass this in when the caller already has the cave_result (e.g. from
        experiments.py) to avoid computing it twice.

    Returns
    -------
        Dict with verifier quality metrics
    """
    # 1. Get verifier's claims
    verifier_verdict = state.verdict
    verifier_confidence = state.confidence
    verifier_hallucination = state.hallucination
    verifier_feedback = state.verifier_feedback

    # 2. Compute ground truth — reuse pre-computed result if available
    if cave_result is None:
        from solver.solver_evals import compute_cave_score  # lazy: avoids circular import

        cave_result = compute_cave_score(state, skip_nli=False, skip_clip=True, skip_ais=False)

    gt_answer_correct = cave_result["accuracy"] == 1.0
    gt_citations_valid = cave_result["citation_precision"] >= 0.7  # Threshold
    gt_reasoning_grounded = cave_result["ais"] >= 0.5  # Threshold
    gt_has_hallucinations = cave_result["hallucination_rate"] > 0.3  # >30% hallucinated

    # 3. What the verifier SHOULD have concluded
    gt_should_accept = gt_answer_correct and gt_citations_valid and gt_reasoning_grounded

    # 4. Compare verifier's decision to ground truth
    verifier_accepted = verifier_verdict == "VERIFIED"

    # Decision correctness
    decision_correct = verifier_accepted == gt_should_accept

    # 5. Evaluate specific verifier capabilities

    # Did verifier correctly identify hallucinations?
    verifier_found_hallucination = verifier_hallucination not in ["NONE DETECTED", "UNKNOWN", ""]
    hallucination_detection_correct = verifier_found_hallucination == gt_has_hallucinations

    # Did verifier's confidence match the evidence quality?
    if verifier_confidence == "HIGH":
        confidence_appropriate = gt_citations_valid and gt_reasoning_grounded
    elif verifier_confidence == "LOW":
        confidence_appropriate = not gt_citations_valid or not gt_reasoning_grounded
    else:
        confidence_appropriate = True  # MEDIUM is always acceptable

    # 6. Analyze verifier feedback quality (if rejected)
    feedback_quality = evaluate_feedback_quality(
        verifier_feedback, gt_citations_valid, gt_reasoning_grounded, gt_answer_correct
    )

    return {
        # Verifier outputs
        "verdict": verifier_verdict,
        "confidence": verifier_confidence,
        "hallucination_type": verifier_hallucination,
        # Ground truth
        "gt_answer_correct": gt_answer_correct,
        "gt_citations_valid": gt_citations_valid,
        "gt_reasoning_grounded": gt_reasoning_grounded,
        "gt_has_hallucinations": gt_has_hallucinations,
        "gt_should_accept": gt_should_accept,
        # Verifier quality metrics
        "decision_correct": decision_correct,
        "hallucination_detection_correct": hallucination_detection_correct,
        "confidence_appropriate": confidence_appropriate,
        "feedback_quality": feedback_quality["score"],
        # Detailed scores
        "citation_precision": cave_result["citation_precision"],
        "ais": cave_result["ais"],
        "hallucination_rate": cave_result["hallucination_rate"],
    }


# One hypothesis per feedback quality dimension.
# Each is written as "This feedback ..." so the NLI model receives a clear,
# consistent subject — the feedback text is the premise, the hypothesis states
# what we want to know about it.
_FEEDBACK_HYPOTHESES = {
    "identifies_citation_issues": ("This feedback identifies problems with citations, evidence, or source support."),
    "identifies_hallucination": (
        "This feedback identifies hallucinated, fabricated, or ungrounded claims in the reasoning."
    ),
    "identifies_answer_issues": ("This feedback identifies a problem with the final answer or conclusion."),
    "is_actionable": ("This feedback gives specific actions or queries the planner should take to improve."),
}

# Entailment probability threshold: treat a dimension as "detected" when the
# NLI model assigns at least this probability to entailment.  0.4 is
# intentionally a little below 0.5 to stay on the recall-favoring side —
# over-penalising valid feedback is worse than under-penalising weak feedback.
_FEEDBACK_NLI_THRESHOLD = 0.4


def _classify_feedback_dimensions(feedback: str) -> Dict[str, bool]:
    """
    Use the NLI model to decide which quality dimensions the feedback covers.
    One batched forward pass handles all four hypotheses at once.
    Returns a dict mapping each dimension name to a bool.
    """
    from solver.solver_evals import get_nli_model  # lazy: avoids circular import

    nli_model = get_nli_model()
    # Build (premise=feedback, hypothesis=<dimension hypothesis>) pairs in a
    # fixed order so we can zip results back to dimension names easily.
    dimension_names = list(_FEEDBACK_HYPOTHESES.keys())
    pairs = [(feedback, _FEEDBACK_HYPOTHESES[name]) for name in dimension_names]
    # apply_softmax=True converts logits to probabilities so the threshold
    # comparison is meaningful.
    scores = nli_model.predict(pairs, apply_softmax=True)  # shape: (4, num_labels)
    label_map = nli_model.config.id2label
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    entail_idx = label_to_idx.get("entailment", 1)
    return {name: float(row[entail_idx]) >= _FEEDBACK_NLI_THRESHOLD for name, row in zip(dimension_names, scores)}


def evaluate_feedback_quality(
    feedback: str, gt_citations_valid: bool, gt_reasoning_grounded: bool, gt_answer_correct: bool
) -> dict:
    if not feedback:
        return {
            "has_feedback": False,
            "identifies_citation_issues": False,
            "identifies_hallucination": False,
            "identifies_answer_issues": False,
            "is_actionable": False,
            "score": 0.0,
        }

    dims = _classify_feedback_dimensions(feedback)
    identifies_citations = dims["identifies_citation_issues"]
    identifies_hallucination = dims["identifies_hallucination"]
    identifies_answer = dims["identifies_answer_issues"]
    is_actionable = dims["is_actionable"]

    # Ground truth: which issues actually exist?
    issue_exists = [not gt_citations_valid, not gt_reasoning_grounded, not gt_answer_correct]
    issue_detected = [identifies_citations, identifies_hallucination, identifies_answer]

    # Score each issue dimension: +1 for correct detection/non-detection,
    # -1 for false alarm (flagging an issue that doesn't exist),
    # 0 for a missed real issue.
    dimension_scores = []
    for exists, detected in zip(issue_exists, issue_detected):
        if exists and detected:
            dimension_scores.append(1.0)  # true positive
        elif exists and not detected:
            dimension_scores.append(0.0)  # missed issue
        elif not exists and detected:
            dimension_scores.append(-1.0)  # false alarm — penalized
        else:
            dimension_scores.append(1.0)  # true negative (correctly quiet)

    # Normalize issue scores to [0, 1] (raw range is [-1, 1])
    issue_score = (sum(dimension_scores) / len(dimension_scores) + 1.0) / 2.0

    # Actionability is a separate quality gate — only meaningful when there
    # are real issues. If no issues exist, actionability is irrelevant.
    has_real_issues = any(issue_exists)
    if has_real_issues:
        # Weight: 70% issue accuracy, 30% actionability
        final_score = 0.7 * issue_score + 0.3 * (1.0 if is_actionable else 0.0)
    else:
        # Spurious feedback — cap at 0.5 regardless of content accuracy,
        # since the verifier should not have generated feedback at all.
        final_score = min(0.5, issue_score)

    return {
        "has_feedback": True,
        "identifies_citation_issues": identifies_citations,
        "identifies_hallucination": identifies_hallucination,
        "identifies_answer_issues": identifies_answer,
        "is_actionable": is_actionable,
        "score": round(final_score, 4),
    }
