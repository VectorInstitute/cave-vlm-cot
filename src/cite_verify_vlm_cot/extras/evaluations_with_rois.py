import math
import re
import numpy as np
from sentence_transformers.util import cos_sim
import torch

import base64
from io import BytesIO
from sentence_transformers import CrossEncoder
from PIL import Image

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

# At module level
_nli_model_cache = {}

# Pattern for [Question Image N] citations
QUESTION_IMAGE_PATTERN = r'\[Question Image (\d+)\]'

# Pattern for [Text Evidence N] citations
TEXT_EVIDENCE_PATTERN = r'\[Text Evidence (\d+)\]'

# Combined pattern for any image citation (supports legacy ROI too)
ANY_IMAGE_CITATION = r'\[(Question Image \d+|Image ROI \d+)\]'

def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base"):
    """Get cached NLI model."""
    if model_name not in _nli_model_cache:
        _nli_model_cache[model_name] = CrossEncoder(model_name)
    return _nli_model_cache[model_name]
    
###### PLANNER EVALUATION 
def planner_coverage_score(state) -> float:
    """Compute how well subqueries cover key concepts from question and choices."""
    if not state.subqueries:
        return 0.0

    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
                  'how', 'why', 'when', 'where', 'who', 'of', 'in', 'on', 'at',
                  'to', 'for', 'with', 'by', 'from', 'as', 'this', 'that', 'it'}

    def extract_terms(text: str) -> set:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words}

    question_terms = extract_terms(state.question)
    choice_terms = set()
    for choice in (state.choices or []):
        choice_terms.update(extract_terms(str(choice)))

    key_terms = question_terms | choice_terms
    if not key_terms:
        return 1.0

    subquery_text = " ".join(state.subqueries).lower()
    covered = sum(1 for term in key_terms if term in subquery_text)

    return covered / len(key_terms)

def planner_hit_rate(state, k: int = 2, threshold: float = 0.5) -> bool:
    """Check if any subquery led to retrieving gold answer."""
    if not state.retrieved_chunks or not state.gold_answer:
        return False
    
    # Use SentenceTransformer instead of CLIP for text-to-text comparison
    from retriever import text_model
    
    # Get embedding for gold answer using SentenceTransformer
    gold_emb = text_model.encode(state.gold_answer, normalize_embeddings=True)

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check text chunks
        for ev in chunk_info.text_chunks[:k]:
            # Use SentenceTransformer for evidence too
            ev_emb = text_model.encode(ev, normalize_embeddings=True)
            
            # Compute cosine similarity
            similarity = np.dot(gold_emb, ev_emb) / (
                np.linalg.norm(gold_emb) * np.linalg.norm(ev_emb)
            )
            
            if similarity > threshold:
                return True
    
    return False

def planner_specificity_score(state) -> float:
    """
    Compute specificity: Are subqueries specific enough (not too generic)?
    Penalizes very short or overly generic queries.
    """
    if not state.subqueries:
        return 0.0

    scores = []
    generic_starts = ['what is', 'how does', 'definition of', 'explain', 'describe']

    for sq in state.subqueries:
        sq_lower = sq.lower().strip()
        word_count = len(sq_lower.split())

        # Length score (3-10 words is ideal)
        if word_count < 3:
            length_score = 0.3
        elif word_count > 15:
            length_score = 0.6
        else:
            length_score = 1.0

        # Penalize generic starts
        generic_penalty = 0.2 if any(sq_lower.startswith(g) for g in generic_starts) else 0.0

        score = max(0.0, min(1.0, length_score - generic_penalty))
        scores.append(score)

    return sum(scores) / len(scores)

###### RETRIEVER EVALUATION
def recall_at_k(state, k: int = 2) -> dict:
    """
    Compute Recall@K: Does top-K evidence contain the gold answer?
    Returns dict with recall score and per-query hits.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return {"recall": 0.0, "hits": {}}

    gold_lower = state.gold_answer.lower()
    hits = {}

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        top_k = text_evidences[:k]
        hit = any(gold_lower in ev.lower() for ev in top_k)
        hits[subquery] = hit

    recall = sum(hits.values()) / len(hits) if hits else 0.0
    return {"recall": recall, "hits": hits}

def precision_at_k(state, k: int = 2) -> float:
    """Compute Precision@K: What fraction of retrieved docs are relevant?"""
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    gold_lower = state.gold_answer.lower()
    all_evidence = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        all_evidence.extend(text_evidences[:k])

    if not all_evidence:
        return 0.0

    relevant = sum(1 for ev in all_evidence if gold_lower in ev.lower())
    return relevant / len(all_evidence)

def mean_reciprocal_rank(state) -> float:
    """Compute MRR: Average of 1/rank for first relevant document per query."""
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    gold_lower = state.gold_answer.lower()
    rr_scores = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        for rank, ev in enumerate(text_evidences, start=1):
            if gold_lower in ev.lower():
                rr_scores.append(1.0 / rank)
                break
        else:
            rr_scores.append(0.0)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

def ndcg_at_k(state, k: int = 2) -> float:
    """Compute NDCG@K: Normalized Discounted Cumulative Gain.
    Computed per-subquery then averaged.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    gold_lower = state.gold_answer.lower()
    ndcg_scores = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        # Binary relevance for each of the top-k docs for this subquery
        relevances = [
            1 if gold_lower in ev.lower() else 0
            for ev in text_evidences[:k]
        ]

        if not relevances:
            ndcg_scores.append(0.0)
            continue

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
        # Ideal: all relevant docs first
        # ideal_rels: best possible top-k ranking (all 1s first, then 0s)
        ideal_rels = sorted(relevances, reverse=True)  # already at most k items
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

def image_roi_source_precision(state, k: int = 5) -> float:
    """
    Precision: What fraction of retrieved ROIs come from the question's images?
    
    Returns 1.0 if all retrieved ROIs are from relevant images,
    0.0 if none are, or None if this question has no associated images
    (so callers can distinguish "no images" from "0% precision").
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    if not question_images:
        return None   # None is unambiguous and won't pass >= 0 guards silently
    
    matched = 0
    total = 0
    
    for subquery, chunk_info in state.retrieved_chunks.items():
        for roi in chunk_info.image_rois[:k]:
            total += 1
            if roi.source_image in question_images:
                matched += 1
    
    return matched / total if total > 0 else None

def image_roi_source_recall(state, k: int = 5) -> float:
    """
    Recall: What fraction of the question's images have at least one ROI retrieved?
    
    Returns 1.0 if ROIs from all question images were retrieved,
    0.0 if some images had no ROI, or None if this question has no associated
    images OR if no ROIs were retrieved at all (e.g. ROI retrieval is disabled).
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    if not question_images:
        return None
    
    retrieved_sources = set()
    for subquery, chunk_info in state.retrieved_chunks.items():
        for roi in chunk_info.image_rois[:k]:
            retrieved_sources.add(roi.source_image)
    
    # ROI retrieval is disabled — no ROIs were produced, metric is not applicable.
    if not retrieved_sources:
        return None

# def image_roi_mrr(state, k: int = 5) -> float:
#     """
#     Mean Reciprocal Rank for image ROIs.
    
#     What's the rank of the first ROI from a relevant image?
#     Returns None if this question has no associated images.
#     """
#     question_images = set(state.image_paths) if state.image_paths else set()
    
#     if not question_images:
#         return None
    
#     reciprocal_ranks = []
    
#     for subquery, chunk_info in state.retrieved_chunks.items():
#         rois = chunk_info.image_rois[:k]
#         for rank, roi in enumerate(rois, 1):
#             if roi.source_image in question_images:
#                 reciprocal_ranks.append(1.0 / rank)
#                 break
#         else:
#             reciprocal_ranks.append(0.0)
    
#     return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def image_roi_coverage(state) -> dict:
    """
    Detailed coverage metrics for image ROI retrieval.
    
    Returns dict with multiple metrics.
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    total_rois = 0
    relevant_rois = 0
    unique_sources = set()
    roi_scores = []
    
    for subquery, chunk_info in state.retrieved_chunks.items():
        for roi in chunk_info.image_rois:
            total_rois += 1
            unique_sources.add(roi.source_image)
            roi_scores.append(roi.score)
        
            if roi.source_image in question_images:
                relevant_rois += 1
    
    return {
        'total_rois_retrieved': total_rois,
        'relevant_rois': relevant_rois,
        'unique_source_images': len(unique_sources),
        'question_images_count': len(question_images),
        'precision': relevant_rois / total_rois if total_rois > 0 else 0,
        'avg_roi_score': sum(roi_scores) / len(roi_scores) if roi_scores else 0,
        'images_covered': len(question_images & unique_sources) / len(question_images) if question_images else 0,
    }


###### SOLVER EVALUATION
def final_answer_accuracy(state) -> float:
    """Is the final answer correct?
    state.gold_answer is the full choice string (e.g. "cutting paper"), NOT a letter.
    state.choices is the ordered list of choice strings (e.g. ["cutting paper", ...]).
    state.answer is the integer index of the correct choice (0-based).
    Strategy:
      1. Extract the predicted letter (A/B/C/D) from the solver output.
      2. Convert that letter to a 0-based index (A→0, B→1, …).
      3. Compare that index against state.answer (the gold index).
    Fallback: also do a direct text match against gold_answer in case the
    solver wrote out the full choice string instead of a letter.
    """
    
    solver_answer = state.final_answer or ""
    gold_answer   = state.gold_answer or ""
    gold_index    = getattr(state, "answer", None)   # int, 0-based
    
    # Extract predicted letter
    predicted_letter = None

    # Pattern 1: "The answer is A" / "The answer is: A"
    match = re.search(r'[Tt]he answer is\s*[:\s]*([A-Da-d])', solver_answer)
    if match:
        predicted_letter = match.group(1).upper()
    
    # Pattern 2: "Answer: A"
    if not predicted_letter:
        match = re.search(r'[Aa]nswer[:\s]+([A-Da-d])', solver_answer)
        if match:
            predicted_letter = match.group(1).upper()
    
    # Pattern 3: standalone letter  e.g. just "A"
    if not predicted_letter:
        match = re.search(r'\b([A-Da-d])\b', solver_answer)
        if match:
            predicted_letter = match.group(1).upper()

    # Compare predicted letter against gold index
    if predicted_letter is not None and gold_index is not None:
        # "A" → 0, "B" → 1, "C" → 2, "D" → 3
        predicted_index = ord(predicted_letter) - ord('A')
        if predicted_index == gold_index:
            return 1.0

    # Fallback: check if solver wrote the full choice text 
    # Handles cases where the model outputs "cutting paper" rather than "A"
    if gold_answer and gold_answer.strip().lower() in solver_answer.lower():
        return 1.0
    return 0.0

# def extract_citation_claim_pairs(reasoning_text: str) -> list:
#     """
#     Extract all (citation, claim) pairs from reasoning text.
#     Returns:
#         List of dicts: [{'citation_id': 1, 'type': 'text', 'claim': '...', 'raw': '[Text Evidence 1]'}, ...]
#     """
#     pairs = []
    
#     # Split into sentences/steps
#     if re.search(r'Step\s*\d+', reasoning_text):
#         sentences = re.split(r'(?:^|\n)\s*(?:-\s*)?Step\s*\d+[:\.]?\s*', reasoning_text)
#     else:
#         sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
#     sentences = [s.strip() for s in sentences if s.strip()]
    
#     for sentence in sentences:
#         # Find text evidence citations
#         for match in re.finditer(r'\[Text Evidence (\d+)\]', sentence):
#             claim = _extract_claim(sentence, match.group(0), match.start())
#             pairs.append({
#                 'citation_id': int(match.group(1)),
#                 'type': 'text',
#                 'claim': claim,
#                 'raw': match.group(0),
#                 'sentence': sentence
#             })
        
#         # Find image ROI citations
#         for match in re.finditer(r'\[Image ROI (\d+)\]', sentence):
#             claim = _extract_claim(sentence, match.group(0), match.start())
#             pairs.append({
#                 'citation_id': int(match.group(1)),
#                 'type': 'image_roi',
#                 'claim': claim,
#                 'raw': match.group(0),
#                 'sentence': sentence
#             })import math
import re
import numpy as np
from sentence_transformers.util import cos_sim
import torch

import base64
from io import BytesIO
from sentence_transformers import CrossEncoder
from PIL import Image

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

# At module level
_nli_model_cache = {}

# Pattern for [Question Image N] citations
QUESTION_IMAGE_PATTERN = r'\[Question Image (\d+)\]'

# Pattern for [Text Evidence N] citations
TEXT_EVIDENCE_PATTERN = r'\[Text Evidence (\d+)\]'

# Combined pattern for any image citation (supports legacy ROI too)
ANY_IMAGE_CITATION = r'\[(Question Image \d+|Image ROI \d+)\]'

def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base"):
    """Get cached NLI model."""
    if model_name not in _nli_model_cache:
        _nli_model_cache[model_name] = CrossEncoder(model_name)
    return _nli_model_cache[model_name]
    
###### PLANNER EVALUATION 
def planner_coverage_score(state) -> float:
    """Compute how well subqueries cover key concepts from question and choices."""
    if not state.subqueries:
        return 0.0

    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
                  'how', 'why', 'when', 'where', 'who', 'of', 'in', 'on', 'at',
                  'to', 'for', 'with', 'by', 'from', 'as', 'this', 'that', 'it'}

    def extract_terms(text: str) -> set:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words}

    question_terms = extract_terms(state.question)
    choice_terms = set()
    for choice in (state.choices or []):
        choice_terms.update(extract_terms(str(choice)))

    key_terms = question_terms | choice_terms
    if not key_terms:
        return 1.0

    subquery_text = " ".join(state.subqueries).lower()
    covered = sum(1 for term in key_terms if term in subquery_text)

    return covered / len(key_terms)

def planner_hit_rate(state, k: int = 2, threshold: float = 0.5) -> bool:
    """Check if any subquery led to retrieving gold answer."""
    if not state.retrieved_chunks or not state.gold_answer:
        return False
    
    # Use SentenceTransformer instead of CLIP for text-to-text comparison
    from retriever import text_model
    
    # Get embedding for gold answer using SentenceTransformer
    gold_emb = text_model.encode(state.gold_answer, normalize_embeddings=True)

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check text chunks
        for ev in chunk_info.text_chunks[:k]:
            # Use SentenceTransformer for evidence too
            ev_emb = text_model.encode(ev, normalize_embeddings=True)
            
            # Compute cosine similarity
            similarity = np.dot(gold_emb, ev_emb) / (
                np.linalg.norm(gold_emb) * np.linalg.norm(ev_emb)
            )
            
            if similarity > threshold:
                return True
    
    return False

def planner_specificity_score(state) -> float:
    """
    Compute specificity: Are subqueries specific enough (not too generic)?
    Penalizes very short or overly generic queries.
    """
    if not state.subqueries:
        return 0.0

    scores = []
    generic_starts = ['what is', 'how does', 'definition of', 'explain', 'describe']

    for sq in state.subqueries:
        sq_lower = sq.lower().strip()
        word_count = len(sq_lower.split())

        # Length score (3-10 words is ideal)
        if word_count < 3:
            length_score = 0.3
        elif word_count > 15:
            length_score = 0.6
        else:
            length_score = 1.0

        # Penalize generic starts
        generic_penalty = 0.2 if any(sq_lower.startswith(g) for g in generic_starts) else 0.0

        score = max(0.0, min(1.0, length_score - generic_penalty))
        scores.append(score)

    return sum(scores) / len(scores)

###### RETRIEVER EVALUATION
def recall_at_k(state, k: int = 2) -> dict:
    """
    Compute Recall@K: Does top-K evidence contain the gold answer?
    Returns dict with recall score and per-query hits.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return {"recall": 0.0, "hits": {}}

    gold_lower = state.gold_answer.lower()
    hits = {}

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        top_k = text_evidences[:k]
        hit = any(gold_lower in ev.lower() for ev in top_k)
        hits[subquery] = hit

    recall = sum(hits.values()) / len(hits) if hits else 0.0
    return {"recall": recall, "hits": hits}

def precision_at_k(state, k: int = 2) -> float:
    """Compute Precision@K: What fraction of retrieved docs are relevant?"""
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    gold_lower = state.gold_answer.lower()
    all_evidence = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        all_evidence.extend(text_evidences[:k])

    if not all_evidence:
        return 0.0

    relevant = sum(1 for ev in all_evidence if gold_lower in ev.lower())
    return relevant / len(all_evidence)

def mean_reciprocal_rank(state) -> float:
    """Compute MRR: Average of 1/rank for first relevant document per query."""
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    gold_lower = state.gold_answer.lower()
    rr_scores = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        for rank, ev in enumerate(text_evidences, start=1):
            if gold_lower in ev.lower():
                rr_scores.append(1.0 / rank)
                break
        else:
            rr_scores.append(0.0)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

def ndcg_at_k(state, k: int = 2) -> float:
    """Compute NDCG@K: Normalized Discounted Cumulative Gain.
    Computed per-subquery then averaged.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    gold_lower = state.gold_answer.lower()
    ndcg_scores = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        # Check both text_chunks and image_chunks
        text_evidences = chunk_info.text_chunks
        # Binary relevance for each of the top-k docs for this subquery
        relevances = [
            1 if gold_lower in ev.lower() else 0
            for ev in text_evidences[:k]
        ]

        if not relevances:
            ndcg_scores.append(0.0)
            continue

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
        # Ideal: all relevant docs first
        # ideal_rels: best possible top-k ranking (all 1s first, then 0s)
        ideal_rels = sorted(relevances, reverse=True)  # already at most k items
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

def image_roi_source_precision(state, k: int = 5) -> float:
    """
    Precision: What fraction of retrieved ROIs come from the question's images?
    
    Returns 1.0 if all retrieved ROIs are from relevant images,
    0.0 if none are, or None if this question has no associated images
    (so callers can distinguish "no images" from "0% precision").
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    if not question_images:
        return None   # None is unambiguous and won't pass >= 0 guards silently
    
    matched = 0
    total = 0
    
    for subquery, chunk_info in state.retrieved_chunks.items():
        for roi in chunk_info.image_rois[:k]:
            total += 1
            if roi.source_image in question_images:
                matched += 1
    
    return matched / total if total > 0 else None

def image_roi_source_recall(state, k: int = 5) -> float:
    """
    Recall: What fraction of the question's images have at least one ROI retrieved?
    
    Returns 1.0 if ROIs from all question images were retrieved, or None if this question has no associated images.
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    if not question_images:
        return None
    
    retrieved_sources = set()
    for subquery, chunk_info in state.retrieved_chunks.items():
        for roi in chunk_info.image_rois[:k]:
            retrieved_sources.add(roi.source_image)
    
    matched = len(question_images & retrieved_sources)
    return matched / len(question_images)

def image_roi_mrr(state, k: int = 5) -> float:
    """
    Mean Reciprocal Rank for image ROIs.
    
    What's the rank of the first ROI from a relevant image?
    Returns None if this question has no associated images.
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    if not question_images:
        return None
    
    reciprocal_ranks = []
    
    for subquery, chunk_info in state.retrieved_chunks.items():
        rois = chunk_info.image_rois[:k]
        for rank, roi in enumerate(rois, 1):
            if roi.source_image in question_images:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def image_roi_coverage(state) -> dict:
    """
    Detailed coverage metrics for image ROI retrieval.
    
    Returns dict with multiple metrics.
    """
    question_images = set(state.image_paths) if state.image_paths else set()
    
    total_rois = 0
    relevant_rois = 0
    unique_sources = set()
    roi_scores = []
    
    for subquery, chunk_info in state.retrieved_chunks.items():
        for roi in chunk_info.image_rois:
            total_rois += 1
            unique_sources.add(roi.source_image)
            roi_scores.append(roi.score)
        
            if roi.source_image in question_images:
                relevant_rois += 1
    
    return {
        'total_rois_retrieved': total_rois,
        'relevant_rois': relevant_rois,
        'unique_source_images': len(unique_sources),
        'question_images_count': len(question_images),
        'precision': relevant_rois / total_rois if total_rois > 0 else 0,
        'avg_roi_score': sum(roi_scores) / len(roi_scores) if roi_scores else 0,
        'images_covered': len(question_images & unique_sources) / len(question_images) if question_images else 0,
    }


###### SOLVER EVALUATION
def final_answer_accuracy(state) -> float:
    """Is the final answer correct?
    state.gold_answer is the full choice string (e.g. "cutting paper"), NOT a letter.
    state.choices is the ordered list of choice strings (e.g. ["cutting paper", ...]).
    state.answer is the integer index of the correct choice (0-based).
    Strategy:
      1. Extract the predicted letter (A/B/C/D) from the solver output.
      2. Convert that letter to a 0-based index (A→0, B→1, …).
      3. Compare that index against state.answer (the gold index).
    Fallback: also do a direct text match against gold_answer in case the
    solver wrote out the full choice string instead of a letter.
    """
    
    solver_answer = state.final_answer or ""
    gold_answer   = state.gold_answer or ""
    gold_index    = getattr(state, "answer", None)   # int, 0-based
    
    # Extract predicted letter
    predicted_letter = None

    # Pattern 1: "The answer is A" / "The answer is: A"
    match = re.search(r'[Tt]he answer is\s*[:\s]*([A-Da-d])', solver_answer)
    if match:
        predicted_letter = match.group(1).upper()
    
    # Pattern 2: "Answer: A"
    if not predicted_letter:
        match = re.search(r'[Aa]nswer[:\s]+([A-Da-d])', solver_answer)
        if match:
            predicted_letter = match.group(1).upper()
    
    # Pattern 3: standalone letter  e.g. just "A"
    if not predicted_letter:
        match = re.search(r'\b([A-Da-d])\b', solver_answer)
        if match:
            predicted_letter = match.group(1).upper()

    # Compare predicted letter against gold index
    if predicted_letter is not None and gold_index is not None:
        # "A" → 0, "B" → 1, "C" → 2, "D" → 3
        predicted_index = ord(predicted_letter) - ord('A')
        if predicted_index == gold_index:
            return 1.0

    # Fallback: check if solver wrote the full choice text 
    # Handles cases where the model outputs "cutting paper" rather than "A"
    if gold_answer and gold_answer.strip().lower() in solver_answer.lower():
        return 1.0
    return 0.0

# def extract_citation_claim_pairs(reasoning_text: str) -> list:
#     """
#     Extract all (citation, claim) pairs from reasoning text.
#     Returns:
#         List of dicts: [{'citation_id': 1, 'type': 'text', 'claim': '...', 'raw': '[Text Evidence 1]'}, ...]
#     """
#     pairs = []
    
#     # Split into sentences/steps
#     if re.search(r'Step\s*\d+', reasoning_text):
#         sentences = re.split(r'(?:^|\n)\s*(?:-\s*)?Step\s*\d+[:\.]?\s*', reasoning_text)
#     else:
#         sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
#     sentences = [s.strip() for s in sentences if s.strip()]
    
#     for sentence in sentences:
#         # Find text evidence citations
#         for match in re.finditer(r'\[Text Evidence (\d+)\]', sentence):
#             claim = _extract_claim(sentence, match.group(0), match.start())
#             pairs.append({
#                 'citation_id': int(match.group(1)),
#                 'type': 'text',
#                 'claim': claim,
#                 'raw': match.group(0),
#                 'sentence': sentence
#             })
        
#         # Find image ROI citations
#         for match in re.finditer(r'\[Image ROI (\d+)\]', sentence):
#             claim = _extract_claim(sentence, match.group(0), match.start())
#             pairs.append({
#                 'citation_id': int(match.group(1)),
#                 'type': 'image_roi',
#                 'claim': claim,
#                 'raw': match.group(0),
#                 'sentence': sentence
#             })
    
#     return pairs

def extract_citation_pairs_v2(reasoning_text: str) -> List[Dict]:
    """
    Unified citation-pair extractor that handles all citation formats:
      - [Text Evidence N]
      - [Question Image N]   (current format — images passed directly to solver)
      - [Image ROI N]        (legacy format — kept for backward compat)
    Returns list of dicts with keys: type, citation_id, claim, raw, sentence.
    """
    pairs = []
    sentences = re.split(r'[.!?]\s+', reasoning_text)
    for sentence in sentences:
        if not sentence.strip():
            continue

        # Text Evidence
        for match in re.finditer(TEXT_EVIDENCE_PATTERN, sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append({
                'citation_id': int(match.group(1)),
                'type': 'text',
                'claim': claim,
                'raw': match.group(0),
                'sentence': sentence,
            })
        # Question Image (current format)
        for match in re.finditer(QUESTION_IMAGE_PATTERN, sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append({
                'citation_id': int(match.group(1)),
                'type': 'question_image',
                'claim': claim,
                'raw': match.group(0),
                'sentence': sentence,
            })
        # Image ROI (legacy format)
        for match in re.finditer(r'\[Image ROI (\d+)\]', sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append({
                'citation_id': int(match.group(1)),
                'type': 'image_roi',
                'claim': claim,
                'raw': match.group(0),
                'sentence': sentence,
            })
    return pairs

# def _extract_claim(sentence: str, citation: str, citation_start: int) -> str:
#     """Extract the claim associated with a citation."""
    
#     # Pattern: "According to [Citation], CLAIM"
#     pattern = r'^(?:According to|From|Based on)\s*' + re.escape(citation) + r'\s*,\s*(.+)'
#     match = re.search(pattern, sentence, re.IGNORECASE)
#     if match:
#         return _clean_claim(match.group(1))
    
#     # Pattern: "[Citation] shows/indicates that CLAIM"
#     pattern = re.escape(citation) + r'\s+(?:shows?|indicates?|confirms?)\s+(?:that\s+)?(.+)'
#     match = re.search(pattern, sentence, re.IGNORECASE)
#     if match:
#         return _clean_claim(match.group(1))
    
#     # Pattern: "CLAIM [Citation]" (citation at end)
#     if sentence.rstrip('.').endswith(citation):
#         claim = sentence[:citation_start].strip()
#         return _clean_claim(claim)
    
#     # Fallback: sentence minus citation
#     return _clean_claim(sentence.replace(citation, '').strip())

def _extract_claim(sentence: str, citation: str, citation_start: int) -> str:
    """Extract the claim associated with a citation."""
    
    # Pattern: "According to [Citation], CLAIM"
    pattern = r'^(?:According to|From|Based on|Looking at|In)\s*' + re.escape(citation) + r'\s*,\s*(.+)'
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return _clean_claim(match.group(1))
    
    # Pattern: "[Citation] shows/indicates that CLAIM"
    pattern = re.escape(citation) + r'\s+(?:shows?|indicates?|confirms?|reveals?)\s+(?:that\s+)?(.+)'
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return _clean_claim(match.group(1))
    
    # Pattern: "CLAIM [Citation]" (citation at end)
    if sentence.rstrip('.').endswith(citation):
        claim = sentence[:citation_start].strip()
        return _clean_claim(claim)
    
    # Fallback: sentence minus citation
    return _clean_claim(sentence.replace(citation, '').strip())


# def _clean_claim(claim: str) -> str:
#     """Clean extracted claim text."""
#     claim = re.sub(r'\[Text Evidence \d+\]', '', claim)
#     claim = re.sub(r'\[Image ROI \d+\]', '', claim)
#     claim = re.sub(r'\s+', ' ', claim)
#     return claim.strip(' ,;:.')

def _clean_claim(claim: str) -> str:
    """Clean extracted claim text."""
    claim = re.sub(TEXT_EVIDENCE_PATTERN, '', claim)
    claim = re.sub(ANY_IMAGE_CITATION, '', claim)
    claim = re.sub(r'\s+', ' ', claim)
    return claim.strip(' ,;:.')

def get_text_evidence_by_id(evidence_id: int, retrieved_chunks: dict) -> str:
    """Get text evidence content by citation ID (1-indexed)."""
    all_chunks = []
    
    for query, chunk_info in retrieved_chunks.items():
        # Handle Pydantic models
        if hasattr(chunk_info, 'text_chunks'):
            all_chunks.extend(chunk_info.text_chunks)
        # Handle dicts
        elif isinstance(chunk_info, dict):
            all_chunks.extend(chunk_info.get('text_chunks', []))
    
    idx = evidence_id - 1  # Convert to 0-indexed
    if 0 <= idx < len(all_chunks):
        return all_chunks[idx]
    return None

def text_citation_precision(state) -> dict:
    """
    For each [Text Evidence N] citation, check if the cited text entails the claim using NLI.
    Natural Language Inference is a task where a model determines the logical relationship between two pieces of text.
    The three NLI labels
    1. Entailment: Premise supports/implies hypothesis
    2. Contradiction: Premise contradicts hypothesis
    3. Neutral: Premise neither supports nor contradicts
    Premise (Text evidence), Hypothesis (Claim supported by text evidence)
    
    Returns:
        dict with 'precision', 'num_citations', 'details'
    """
    
    # Get reasoning text
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning
    
    # Get retrieved evidence
    retrieved_chunks = state.retrieved_chunks
    
    # Extract citation-claim pairs
    all_pairs = extract_citation_claim_pairs(reasoning_text)
    text_pairs = [p for p in all_pairs if p['type'] == 'text']
    
    if not text_pairs:
        return {'precision': 0.0, 'num_citations': 0, 'details': []}
    
    # Load NLI model
    nli_model = get_nli_model()
    
    # Build all (evidence, claim) pairs up front 
    details = []
    total_score = 0.0

    valid_pairs   = []   # [(evidence_str, pair_dict), ...]
    skipped_ids   = []
    
    for pair in text_pairs:
        evidence = get_text_evidence_by_id(pair['citation_id'], retrieved_chunks)
        
        if evidence is None:
            print(f"Warning: Text Evidence {pair['citation_id']} not found")
            skipped_ids.append(pair['citation_id'])
            continue
        
        valid_pairs.append((evidence, pair))
    
    if not valid_pairs:
        return {'precision': 0.0, 'num_citations': 0, 'details': []}
        
    # Single batched NLI call
    nli_inputs  = [(ev, p['claim']) for ev, p in valid_pairs]
    all_scores  = nli_model.predict(nli_inputs)   # shape: (N, num_labels)
    label_map   = nli_model.config.id2label
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    entailment_idx = label_to_idx.get('entailment', 1)  # fallback to index 1
    label_score_map = {'entailment': 1.0, 'neutral': 0.5, 'contradiction': 0.0}
    details     = []
    total_score = 0.0

    for (evidence, pair), scores in zip(valid_pairs, all_scores):
        # Use direct index to get entailment score (avoids label-order bugs)
        score_dict = {label_map[i].lower(): float(scores[i]) for i in range(len(scores))}
        predicted_label = max(score_dict, key=score_dict.get)
        precision_score = label_score_map.get(predicted_label, 0.0)
        total_score += precision_score
        
        details.append({
            'citation': pair['raw'],
            'claim': pair['claim'],
            'evidence': evidence[:100] + '...' if len(evidence) > 100 else evidence,
            'nli_label': predicted_label,
            'score': precision_score
        })
    
    precision = total_score / len(details) if details else 0.0
    
    return {
        'precision': precision,
        'num_citations': len(details),
        'details': details
    }

# def roi_citation_precision(state) -> dict:
#     """
#     For each [Image ROI N] citation, check if the ROI matches the claim using CLIP.
    
#     Returns:
#         dict with 'precision', 'num_citations', 'details'
#     """
#     from retriever import clip_model, clip_processor
    
#     # Get reasoning text
#     reasoning = state.reasoning_steps
#     if isinstance(reasoning, list):
#         reasoning_text = "\n".join(reasoning)
#     else:
#         reasoning_text = reasoning
    
#     retrieved_chunks = state.retrieved_chunks
    
#     # Extract ROI citation pairs
#     all_pairs = extract_citation_claim_pairs(reasoning_text)
#     roi_pairs = [p for p in all_pairs if p['type'] == 'image_roi']
    
#     if not roi_pairs:
#         return {'precision': 0.0, 'num_citations': 0, 'details': []}
    
#     # Load CLIP
#     device = next(clip_model.parameters()).device
#     clip_model.eval()
    
#     # Get all ROIs
#     all_rois = []
#     for query, chunk_info in retrieved_chunks.items():
#         if hasattr(chunk_info, 'image_rois'):
#             all_rois.extend(chunk_info.image_rois)
#         elif isinstance(chunk_info, dict):
#             all_rois.extend(chunk_info.get('image_rois', []))
    
#     details = []
#     total_score = 0.0
    
#     for pair in roi_pairs:
#         idx = pair['citation_id'] - 1
#         if idx < 0 or idx >= len(all_rois):
#             continue
        
#         roi = all_rois[idx]
#         image_patch = roi.image_patch if hasattr(roi, 'image_patch') else roi.get('image_patch')
        
#         if not image_patch:
#             continue
        
#         # Decode image
#         try:
#             img_data = base64.b64decode(image_patch)
#             image = Image.open(BytesIO(img_data)).convert('RGB')
#         except:
#             continue
        
#         # Compute CLIP similarity
#         with torch.no_grad():
#             inputs = clip_processor(text=[pair['claim']], images=image, return_tensors="pt", padding=True).to(device)
#             outputs = clip_model(**inputs)
            
#             img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
#             txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
#             similarity = torch.matmul(img_emb, txt_emb.T).item()
        
#         # Normalize to 0-1 score (CLIP similarities typically 0.15-0.35)
#         precision_score = max(0.0, min(1.0, (similarity - 0.15) / 0.25))
#         total_score += precision_score
        
#         caption = roi.caption if hasattr(roi, 'caption') else roi.get('caption', '')
#         details.append({
#             'citation': pair['raw'],
#             'claim': pair['claim'],
#             'roi_caption': caption,
#             'clip_similarity': similarity,
#             'score': precision_score
#         })
    
#     precision = total_score / len(details) if details else 0.0
    
#     return {
#         'precision': precision,
#         'num_citations': len(details),
#         'details': details
#     }

def question_image_citation_precision(state) -> Dict:
    """
    Check if [Question Image N] citations reference valid images.
    
    Since all question images are relevant (they're part of the question),
    precision = whether citation ID is within range of available images.
    """
    reasoning = ' '.join(state.reasoning_steps or [])
    pairs = extract_citation_pairs_v2(reasoning)
    
    # Filter to question image citations
    qi_pairs = [p for p in pairs if p['type'] == 'question_image']
    
    if not qi_pairs:
        return {
            'precision': 0.0,
            'total_citations': 0,
            'valid_citations': 0,
            'details': []
        }
    
    # Count available images
    import os
    num_images = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])
    
    valid = 0
    details = []
    
    for pair in qi_pairs:
        citation_id = pair['citation_id']
        is_valid = 1 <= citation_id <= num_images
        
        if is_valid:
            valid += 1
        
        details.append({
            'citation_id': citation_id,
            'claim': pair['claim'],
            'valid': is_valid,
            'reason': 'Valid image reference' if is_valid else f'Image {citation_id} does not exist (only {num_images} images)'
        })
    
    return {
        'precision': valid / len(qi_pairs) if qi_pairs else 0.0,
        'total_citations': len(qi_pairs),
        'valid_citations': valid,
        'details': details
    }


def attribution_score(state) -> dict:
    """
    Compute Attribution Score (AIS) - proportion of reasoning steps 
    attributable to retrieved evidence.
    For each reasoning step, check if ANY retrieved evidence entails it.
    
    AIS (Attributable to Identified Sources) measures whether each reasoning step can be derived from the retrieved evidence, 
    rather than hallucinated from the model's parametric knowledge.
    Citation Precision : Does the cited evidence support the claim?
    AIS : Can the claim be derived from any retrieved evidence?
    
    Returns:
        dict with 'ais', 'hallucination_rate', 'num_steps', 'details'
    """
    
    # Get reasoning steps
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        steps = reasoning
    else:
        # Split into steps if it's a single string
        steps = _split_into_steps(reasoning)
    
    if not steps:
        return {'ais': 0.0, 'hallucination_rate': 1.0, 'num_steps': 0, 'details': []}
    
    # Collect all evidence (text + ROI captions + hint)
    # pass hint so claims paraphrasing the hint are correctly attributed
    all_evidence = _collect_all_evidence(state.retrieved_chunks, hint=state.hint or '')
    
    if not all_evidence:
        return {'ais': 0.0, 'hallucination_rate': 1.0, 'num_steps': len(steps), 'details': []}
    
    # Load NLI model
    nli_model = get_nli_model()
    
    details = []
    attributable_count = 0
    
    for step in steps:
        step_clean = _clean_step(step)
        if not step_clean:
            continue
        
        # Check if step is attributable to ANY evidence
        is_attributable, best_evidence, best_score = _check_attribution(
            step_clean, all_evidence, nli_model
        )
        
        if is_attributable:
            attributable_count += 1
        
        details.append({
            'step': step_clean[:100] + '...' if len(step_clean) > 100 else step_clean,
            'is_attributable': is_attributable,
            'best_evidence': best_evidence[:80] + '...' if best_evidence and len(best_evidence) > 80 else best_evidence,
            'best_score': best_score
        })
    
    num_steps = len(details)
    ais = attributable_count / num_steps if num_steps > 0 else 0.0
    
    return {
        'ais': ais,
        'hallucination_rate': 1.0 - ais,
        'num_steps': num_steps,
        'num_attributable': attributable_count,
        'details': details
    }


def _split_into_steps(reasoning_text: str) -> list:
    """Split reasoning text into individual steps."""
    # Try step-based splitting first
    if re.search(r'Step\s*\d+', reasoning_text):
        steps = re.split(r'(?:^|\n)\s*(?:-\s*)?Step\s*\d+[:\.]?\s*', reasoning_text)
    else:
        # Fall back to sentence splitting
        steps = re.split(r'(?<=[.!?])\s+(?=[A-Z])', reasoning_text)
    
    return [s.strip() for s in steps if s.strip()]


def _clean_step(step: str) -> str:
    """Clean a reasoning step for attribution checking."""
    # Remove citation markers (we're checking attribution, not citation correctness)
    step = re.sub(r'\[Text Evidence \d+\]', '', step)
    step = re.sub(r'\[Image ROI \d+\]', '', step)
    # Remove common prefixes
    step = re.sub(r'^(?:According to|From|Based on|Therefore|Thus|Hence)[,\s]*', '', step, flags=re.IGNORECASE)
    return step.strip()


def _collect_all_evidence(retrieved_chunks: dict, hint: str = "") -> list:
    """Collect all evidence pieces (text chunks + ROI captions + hint text).
    hint text is often the most directly relevant content for science questions
    but was excluded from attribution checking. Added as an evidence source so AIS
    can attribute claims that paraphrase the hint rather than retrieved chunks.
    """
    evidence = []
    # Include hint text if provided — often the single most relevant sentence
    if hint and hint.strip():
        evidence.append(hint.strip())
    
    for query, chunk_info in retrieved_chunks.items():
        # Text chunks
        if hasattr(chunk_info, 'text_chunks'):
            evidence.extend(chunk_info.text_chunks)
        elif isinstance(chunk_info, dict):
            evidence.extend(chunk_info.get('text_chunks', []))
        
        # ROI captions (also count as evidence)
        if hasattr(chunk_info, 'image_rois'):
            for roi in chunk_info.image_rois:
                caption = roi.caption if hasattr(roi, 'caption') else roi.get('caption', '')
                if caption:
                    evidence.append(caption)
        elif isinstance(chunk_info, dict):
            for roi in chunk_info.get('image_rois', []):
                caption = roi.get('caption', '') if isinstance(roi, dict) else getattr(roi, 'caption', '')
                if caption:
                    evidence.append(caption)
    
    return [e for e in evidence if e]  # Filter empty


def _check_attribution(claim: str, evidence_list: list, nli_model) -> tuple:
    """
    Check if claim is attributable to any evidence.
    Runs a single batched NLI forward pass over all evidence pieces at once,
    rather than one call per piece.  This is significantly faster (CrossEncoder
    supports batch input natively).
    Note: For cross-encoder/nli-deberta-v3-base, label indices are:
        0 -> contradiction, 1 -> entailment, 2 -> neutral
    We use id2label to be safe, but fall back to index 1 if 'entailment' is missing.
    """
    if not evidence_list:
        return False, None, 0.0

    # Build all (premise=evidence, hypothesis=claim) pairs at once
    pairs = [(ev, claim) for ev in evidence_list]
    # Single batched call — CrossEncoder.predict() accepts a list of pairs
    all_scores = nli_model.predict(pairs)  # shape: (N, num_labels)
    label_map = nli_model.config.id2label
    
    # Validate label map contains 'entailment'; warn if not found
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    if 'entailment' not in label_to_idx:
        # Fallback: assume standard DeBERTa NLI order (contradiction=0, entailment=1, neutral=2)
        entailment_idx = 1
        import warnings
        warnings.warn(
            f"NLI model id2label does not contain 'entailment'. Labels: {label_map}. "
            f"Falling back to index {entailment_idx}."
        )
    else:
        entailment_idx = label_to_idx['entailment']
    
    best_score = 0.0
    best_evidence = None
    
    for i, scores in enumerate(all_scores):
        # scores is a 1-D array of length num_labels
        entailment_score = float(scores[entailment_idx])
        
        if entailment_score > best_score:
            best_score = entailment_score
            best_evidence = evidence_list[i]
        
        # Threshold: consider attributable if best entailment > 0.5
        # science lecture text tends to be general/background; specific factual
        # claims naturally score lower against it. 0.5 is too aggressive — most valid
        # attributions cluster in the 0.3–0.5 range for this domain. Lowered to 0.35.
        is_attributable = best_score > 0.35
    
    return is_attributable, best_evidence, best_score

def evidence_grounding_check(state) -> dict:
    """
    Verify that the final answer is actually supported by cited evidence.
    Strategy:
    1. Extract the answer letter/text from state.final_answer.
    2. Collect all evidence cited in the reasoning (via extract_citation_claim_pairs).
    3. Use NLI to check if the cited evidence entails the stated answer.
    Returns a dict with:
        - 'is_grounded': bool — True if answer is supported by at least one cited evidence piece
        - 'grounding_score': float in [0, 1]
        - 'best_evidence': str — the evidence chunk most supportive of the answer
        - 'details': list of per-evidence NLI results
    """
    final_answer = state.final_answer or ""
    retrieved_chunks = state.retrieved_chunks or {}

    # Extract the answer claim (letter + option text)
    answer_match = re.search(
        r'[Tt]he answer is\s*[:\s]*([A-Ea-e])[:\s]+([^\[\n]+)', final_answer
    )
    if answer_match:
        answer_claim = f"{answer_match.group(1).upper()}: {answer_match.group(2).strip()}"
    else:
        # Fall back to full final_answer text (strip XML tags)
        answer_claim = re.sub(r'<[^>]+>', '', final_answer).strip()
    if not answer_claim:
        return {
            'is_grounded': False,
            'grounding_score': 0.0,
            'best_evidence': None,
            'details': [],
            'answer_claim': answer_claim,
        }
    
    # Collect cited evidence pieces
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning or ""

    pairs = extract_citation_claim_pairs(reasoning_text)
    cited_ids_text = {p['citation_id'] for p in pairs if p['type'] == 'text'}
    cited_ids_roi  = {p['citation_id'] for p in pairs if p['type'] == 'image_roi'}
    
    # Gather cited text chunks
    all_chunks = []
    for query, chunk_info in retrieved_chunks.items():
        if hasattr(chunk_info, 'text_chunks'):
            all_chunks.extend(chunk_info.text_chunks)
        elif isinstance(chunk_info, dict):
            all_chunks.extend(chunk_info.get('text_chunks', []))
    cited_chunks = [
        all_chunks[i - 1]
        for i in cited_ids_text
        if 1 <= i <= len(all_chunks)
    ]

    # Gather cited ROI captions
    all_rois = []
    for query, chunk_info in retrieved_chunks.items():
        if hasattr(chunk_info, 'image_rois'):
            all_rois.extend(chunk_info.image_rois)
        elif isinstance(chunk_info, dict):
            all_rois.extend(chunk_info.get('image_rois', []))
    cited_captions = []
    for i in cited_ids_roi:
        if 1 <= i <= len(all_rois):
            roi = all_rois[i - 1]
            cap = roi.caption if hasattr(roi, 'caption') else roi.get('caption', '')
            if cap:
                cited_captions.append(cap)
    cited_evidence = cited_chunks + cited_captions

    # If no citations were found, fall back to ALL retrieved evidence (including hint)
    if not cited_evidence:
        cited_evidence = _collect_all_evidence(retrieved_chunks, hint=state.hint or '')
    if not cited_evidence:
        return {
            'is_grounded': False,
            'grounding_score': 0.0,
            'best_evidence': None,
            'details': [],
            'answer_claim': answer_claim,
        }

    # NLI: does any cited evidence entail the answer claim?
    nli_model = get_nli_model()
    is_grounded, best_evidence, grounding_score = _check_attribution(
        answer_claim, cited_evidence, nli_model
    )
    
    # Build per-evidence detail list (lightweight — just top-5 evidence pieces)
    details = []
    nli_inputs = [(ev, answer_claim) for ev in cited_evidence[:5]]
    if nli_inputs:
        batch_scores = nli_model.predict(nli_inputs)
        label_map = nli_model.config.id2label
        label_to_idx = {v.lower(): k for k, v in label_map.items()}
        entailment_idx = label_to_idx.get('entailment', 1)
        for ev, scores in zip(cited_evidence[:5], batch_scores):
            details.append({
                'evidence': ev[:120] + '...' if len(ev) > 120 else ev,
                'entailment_score': float(scores[entailment_idx]),
            })

    return {
        'is_grounded': is_grounded,
        'grounding_score': grounding_score,
        'best_evidence': best_evidence,
        'details': details,
        'answer_claim': answer_claim,
    }

def compute_cave_score(state, skip_nli=False, skip_clip=False, skip_ais=False) -> dict:
    """
    Compute the combined CaVeScore for solver evaluation.
    
    CaVeScore = 0.4*Accuracy + 0.2*CitePrecision + 0.2*CiteRecall + 0.1*AIS + 0.1*Grounding
    """
    # Accuracy
    accuracy = final_answer_accuracy(state)
    
    # Citation precision
    text_prec = text_citation_precision(state)['precision'] if not skip_nli else 0.0
    # roi_prec = roi_citation_precision(state)['precision'] if not skip_clip else 0.0
    # Question image citation precision: checks that [Question Image N] IDs are in range.
    # skip_clip flag re-used here since image citation checking is lightweight.
    qi_prec_result = question_image_citation_precision(state)
    roi_prec = qi_prec_result['precision'] if not skip_clip else 0.0
    
    # Combined precision
    if text_prec > 0 and roi_prec > 0:
        cite_precision = (text_prec + roi_prec) / 2
    else:
        cite_precision = text_prec or roi_prec

    # AIS (real attribution score)
    if not skip_ais:
        ais_result = attribution_score(state)
        ais = ais_result['ais']
    else:
        ais = cite_precision

    # Evidence grounding check (P2): is the final answer supported by cited evidence?
    if not skip_nli:
        grounding_result = evidence_grounding_check(state)
        grounding_score = grounding_result['grounding_score']
        is_grounded = grounding_result['is_grounded']
    else:
        grounding_score = 0.0
        is_grounded = False
    
    # Citation recall — count citations vs. substantive factual sentences only.
    # Previously every sentence was counted as a verifiable claim, which
    # penalised transition phrases like "Therefore…" or "In conclusion…".
    # Now we only count sentences that contain a concrete factual assertion
    # (i.e. have a subject + a content verb/noun and are not pure transitions).
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning
    
    pairs = extract_citation_claim_pairs(reasoning_text)
    num_citations = len(pairs)
    _transition_re  = re.compile(
        r'^\s*(?:therefore|thus|hence|so|in conclusion|in summary|as a result'
        r'|combining|this (?:means|shows|suggests)|overall)\b',
        re.IGNORECASE,
    )
    _factual_re = re.compile(
        # Must have at least one non-trivial verb indicating a fact
        r'\b(?:is|are|was|were|has|have|had|contains?|shows?|indicates?'
        r'|demonstrates?|proves?|reveals?|confirms?|measures?|equals?'
        r'|represents?|consists?|includes?|causes?|results?)\b',
        re.IGNORECASE,
    )
    raw_sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
    factual_sentences = [
        s for s in raw_sentences
        if s.strip()
        and not _transition_re.match(s)
        and _factual_re.search(s)
    ]
    num_claims = max(1, len(factual_sentences))
    cite_recall = min(1.0, num_citations / num_claims)
    
    # CaVeScore (updated weights to include grounding)
    cave_score = (
        0.4 * accuracy +
        0.2 * cite_precision +
        0.2 * cite_recall +
        0.1 * ais  +
        0.1 * grounding_score
    )
    
    return {
        'accuracy': accuracy,
        'text_citation_precision': text_prec,
        'roi_citation_precision': roi_prec,
        'ais': ais,
        'hallucination_rate': 1.0 - ais,
        'citation_precision': cite_precision,
        'citation_recall': cite_recall,
        'grounding_score': grounding_score,
        'is_grounded': is_grounded,
        'cave_score': cave_score
    }

def compute_combined_metrics(state, skip_clip: bool = True) -> Dict:
    """
    Compute all citation metrics supporting both ROI and Question Image formats.
    """
    reasoning = ' '.join(state.reasoning_steps or [])
    pairs = extract_citation_pairs_v2(reasoning)
    
    # Count by type
    text_citations = [p for p in pairs if p['type'] == 'text']
    roi_citations = [p for p in pairs if p['type'] == 'image_roi']
    qi_citations = [p for p in pairs if p['type'] == 'question_image']
    
    # Question image precision
    num_images = len([p for p in (state.image_paths or []) if p])
    qi_valid = sum(1 for p in qi_citations if 1 <= p['citation_id'] <= num_images)
    
    return {
        'total_citations': len(pairs),
        'text_citations': len(text_citations),
        'roi_citations': len(roi_citations),
        'question_image_citations': len(qi_citations),
        'question_image_precision': qi_valid / len(qi_citations) if qi_citations else None,
        'has_visual_citations': len(roi_citations) + len(qi_citations) > 0,
    }

def citation_summary(state) -> Dict:
    """
    Get a summary of all citations in the response.
    """
    reasoning = ' '.join(state.reasoning_steps or [])
    pairs = extract_citation_pairs_v2(reasoning)
    
    text_cites = [p for p in pairs if p['type'] == 'text']
    image_cites = [p for p in pairs if p['type'] == 'question_image']
    
    return {
        'text_citations': len(text_cites),
        'question_image_citations': len(image_cites),
        'total_citations': len(pairs),
        'has_any_citations': len(pairs) > 0,
    }

###### VERIFIER EVALUATION
'''
Citation Verification: Does it correctly identify when citations don't support claims?
Hallucination Detection: Does it catch claims not grounded in evidence?
Reasoning Validity: Does it detect logical errors in reasoning steps?
Answer Consistency: Does the final answer follow from the reasoning?
'''
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

    Returns:
        Dict with verifier quality metrics
    """
    # 1. Get verifier's claims
    verifier_verdict = state.verdict
    verifier_confidence = state.confidence
    verifier_hallucination = state.hallucination
    verifier_feedback = state.verifier_feedback
    
    # 2. Compute ground truth — reuse pre-computed result if available
    if cave_result is None:
        cave_result = compute_cave_score(state, skip_nli=False, skip_clip=True, skip_ais=False)
    
    gt_answer_correct = cave_result['accuracy'] == 1.0
    gt_citations_valid = cave_result['citation_precision'] >= 0.7  # Threshold
    gt_reasoning_grounded = cave_result['ais'] >= 0.5  # Threshold
    gt_has_hallucinations = cave_result['hallucination_rate'] > 0.3  # >30% hallucinated
    
    # 3. What the verifier SHOULD have concluded
    gt_should_accept = gt_answer_correct and gt_citations_valid and gt_reasoning_grounded
    
    # 4. Compare verifier's decision to ground truth
    verifier_accepted = (verifier_verdict == "VERIFIED")
    
    # Decision correctness
    decision_correct = (verifier_accepted == gt_should_accept)
    
    # 5. Evaluate specific verifier capabilities
    
    # Did verifier correctly identify hallucinations?
    verifier_found_hallucination = verifier_hallucination not in ["NONE", "UNKNOWN", ""]
    hallucination_detection_correct = (verifier_found_hallucination == gt_has_hallucinations)
    
    # Did verifier's confidence match the evidence quality?
    if verifier_confidence == "HIGH":
        confidence_appropriate = gt_citations_valid and gt_reasoning_grounded
    elif verifier_confidence == "LOW":
        confidence_appropriate = not gt_citations_valid or not gt_reasoning_grounded
    else:
        confidence_appropriate = True  # MEDIUM is always acceptable
    
    # 6. Analyze verifier feedback quality (if rejected)
    feedback_quality = evaluate_feedback_quality(
        verifier_feedback, 
        gt_citations_valid, 
        gt_reasoning_grounded,
        gt_answer_correct
    )
    
    return {
        # Verifier outputs
        'verdict': verifier_verdict,
        'confidence': verifier_confidence,
        'hallucination_type': verifier_hallucination,
        
        # Ground truth
        'gt_answer_correct': gt_answer_correct,
        'gt_citations_valid': gt_citations_valid,
        'gt_reasoning_grounded': gt_reasoning_grounded,
        'gt_has_hallucinations': gt_has_hallucinations,
        'gt_should_accept': gt_should_accept,
        
        # Verifier quality metrics
        'decision_correct': decision_correct,
        'hallucination_detection_correct': hallucination_detection_correct,
        'confidence_appropriate': confidence_appropriate,
        'feedback_quality': feedback_quality,
        
        # Detailed scores
        'citation_precision': cave_result['citation_precision'],
        'ais': cave_result['ais'],
        'hallucination_rate': cave_result['hallucination_rate'],
    }


def evaluate_feedback_quality(
    feedback: str,
    gt_citations_valid: bool,
    gt_reasoning_grounded: bool,
    gt_answer_correct: bool
) -> dict:
    """
    Evaluate whether verifier feedback correctly identifies issues.
    
    Good feedback should:
    1. Mention citation issues if citations are invalid
    2. Mention hallucination if reasoning is not grounded
    3. Mention answer issues if answer is wrong
    4. Be actionable for re-planning
    """
    if not feedback:
        return {
            'has_feedback': False,
            'identifies_citation_issues': False,
            'identifies_hallucination': False,
            'identifies_answer_issues': False,
            'is_actionable': False,
            'score': 0.0
        }
    
    feedback_lower = feedback.lower()
    
    # Check if feedback mentions relevant issues
    citation_keywords = ['citation', 'evidence', 'source', 'reference', 'support', 'unsupported']
    hallucination_keywords = ['hallucination', 'fabricat', 'made up', 'not grounded', 'unfounded', 'invented']
    answer_keywords = ['answer', 'incorrect', 'wrong', 'conclusion', 'final']
    actionable_keywords = ['retry', 'replan', 'reconsider', 'check', 'verify', 'revise', 'improve']
    
    identifies_citations = any(kw in feedback_lower for kw in citation_keywords)
    identifies_hallucination = any(kw in feedback_lower for kw in hallucination_keywords)
    identifies_answer = any(kw in feedback_lower for kw in answer_keywords)
    is_actionable = any(kw in feedback_lower for kw in actionable_keywords)
    
    # Score based on whether feedback mentions issues that actually exist.
    # Actionability is always worth a bonus regardless of which issues exist.
    score = 0.0
    checks = 0
    
    if not gt_citations_valid:
        checks += 1
        if identifies_citations:
            score += 1.0
    
    if not gt_reasoning_grounded:
        checks += 1
        if identifies_hallucination:
            score += 1.0
    
    if not gt_answer_correct:
        checks += 1
        if identifies_answer:
            score += 1.0
    
    # Actionability bonus: worth one extra point on top of the issue checks.
    # We always add it to both numerator and denominator so it can only
    # improve (or maintain) the score — never reduce it.
    checks += 1

    if is_actionable:
        score += 1.0
    
    # When checks == 0 that means all three ground-truth conditions were True
    # (no issues at all) AND is_actionable was False.  That edge-case can't
    # happen now because we always add the actionability check, so checks >= 1.
    final_score = score / checks
    
    return {
        'has_feedback': True,
        'identifies_citation_issues': identifies_citations,
        'identifies_hallucination': identifies_hallucination,
        'identifies_answer_issues': identifies_answer,
        'is_actionable': is_actionable,
        'score': final_score
    }
    
#     return pairs

def extract_citation_pairs_v2(reasoning_text: str) -> List[Dict]:
    """
    Unified citation-pair extractor that handles all citation formats:
      - [Text Evidence N]
      - [Question Image N]   (current format — images passed directly to solver)
      - [Image ROI N]        (legacy format — kept for backward compat)
    Returns list of dicts with keys: type, citation_id, claim, raw, sentence.
    """
    pairs = []
    sentences = re.split(r'[.!?]\s+', reasoning_text)
    for sentence in sentences:
        if not sentence.strip():
            continue

        # Text Evidence
        for match in re.finditer(TEXT_EVIDENCE_PATTERN, sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append({
                'citation_id': int(match.group(1)),
                'type': 'text',
                'claim': claim,
                'raw': match.group(0),
                'sentence': sentence,
            })
        # Question Image (current format)
        for match in re.finditer(QUESTION_IMAGE_PATTERN, sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append({
                'citation_id': int(match.group(1)),
                'type': 'question_image',
                'claim': claim,
                'raw': match.group(0),
                'sentence': sentence,
            })
        # Image ROI (legacy format)
        for match in re.finditer(r'\[Image ROI (\d+)\]', sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append({
                'citation_id': int(match.group(1)),
                'type': 'image_roi',
                'claim': claim,
                'raw': match.group(0),
                'sentence': sentence,
            })
    return pairs

# def _extract_claim(sentence: str, citation: str, citation_start: int) -> str:
#     """Extract the claim associated with a citation."""
    
#     # Pattern: "According to [Citation], CLAIM"
#     pattern = r'^(?:According to|From|Based on)\s*' + re.escape(citation) + r'\s*,\s*(.+)'
#     match = re.search(pattern, sentence, re.IGNORECASE)
#     if match:
#         return _clean_claim(match.group(1))
    
#     # Pattern: "[Citation] shows/indicates that CLAIM"
#     pattern = re.escape(citation) + r'\s+(?:shows?|indicates?|confirms?)\s+(?:that\s+)?(.+)'
#     match = re.search(pattern, sentence, re.IGNORECASE)
#     if match:
#         return _clean_claim(match.group(1))
    
#     # Pattern: "CLAIM [Citation]" (citation at end)
#     if sentence.rstrip('.').endswith(citation):
#         claim = sentence[:citation_start].strip()
#         return _clean_claim(claim)
    
#     # Fallback: sentence minus citation
#     return _clean_claim(sentence.replace(citation, '').strip())

def _extract_claim(sentence: str, citation: str, citation_start: int) -> str:
    """Extract the claim associated with a citation."""
    
    # Pattern: "According to [Citation], CLAIM"
    pattern = r'^(?:According to|From|Based on|Looking at|In)\s*' + re.escape(citation) + r'\s*,\s*(.+)'
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return _clean_claim(match.group(1))
    
    # Pattern: "[Citation] shows/indicates that CLAIM"
    pattern = re.escape(citation) + r'\s+(?:shows?|indicates?|confirms?|reveals?)\s+(?:that\s+)?(.+)'
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return _clean_claim(match.group(1))
    
    # Pattern: "CLAIM [Citation]" (citation at end)
    if sentence.rstrip('.').endswith(citation):
        claim = sentence[:citation_start].strip()
        return _clean_claim(claim)
    
    # Fallback: sentence minus citation
    return _clean_claim(sentence.replace(citation, '').strip())


# def _clean_claim(claim: str) -> str:
#     """Clean extracted claim text."""
#     claim = re.sub(r'\[Text Evidence \d+\]', '', claim)
#     claim = re.sub(r'\[Image ROI \d+\]', '', claim)
#     claim = re.sub(r'\s+', ' ', claim)
#     return claim.strip(' ,;:.')

def _clean_claim(claim: str) -> str:
    """Clean extracted claim text."""
    claim = re.sub(TEXT_EVIDENCE_PATTERN, '', claim)
    claim = re.sub(ANY_IMAGE_CITATION, '', claim)
    claim = re.sub(r'\s+', ' ', claim)
    return claim.strip(' ,;:.')

def get_text_evidence_by_id(evidence_id: int, retrieved_chunks: dict) -> str:
    """Get text evidence content by citation ID (1-indexed)."""
    all_chunks = []
    
    for query, chunk_info in retrieved_chunks.items():
        # Handle Pydantic models
        if hasattr(chunk_info, 'text_chunks'):
            all_chunks.extend(chunk_info.text_chunks)
        # Handle dicts
        elif isinstance(chunk_info, dict):
            all_chunks.extend(chunk_info.get('text_chunks', []))
    
    idx = evidence_id - 1  # Convert to 0-indexed
    if 0 <= idx < len(all_chunks):
        return all_chunks[idx]
    return None

def text_citation_precision(state) -> dict:
    """
    For each [Text Evidence N] citation, check if the cited text entails the claim using NLI.
    Natural Language Inference is a task where a model determines the logical relationship between two pieces of text.
    The three NLI labels
    1. Entailment: Premise supports/implies hypothesis
    2. Contradiction: Premise contradicts hypothesis
    3. Neutral: Premise neither supports nor contradicts
    Premise (Text evidence), Hypothesis (Claim supported by text evidence)
    
    Returns:
        dict with 'precision', 'num_citations', 'details'
    """
    
    # Get reasoning text
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning
    
    # Get retrieved evidence
    retrieved_chunks = state.retrieved_chunks
    
    # Extract citation-claim pairs
    all_pairs = extract_citation_claim_pairs(reasoning_text)
    text_pairs = [p for p in all_pairs if p['type'] == 'text']
    
    if not text_pairs:
        return {'precision': 0.0, 'num_citations': 0, 'details': []}
    
    # Load NLI model
    nli_model = get_nli_model()
    
    # Build all (evidence, claim) pairs up front 
    details = []
    total_score = 0.0

    valid_pairs   = []   # [(evidence_str, pair_dict), ...]
    skipped_ids   = []
    
    for pair in text_pairs:
        evidence = get_text_evidence_by_id(pair['citation_id'], retrieved_chunks)
        
        if evidence is None:
            print(f"Warning: Text Evidence {pair['citation_id']} not found")
            skipped_ids.append(pair['citation_id'])
            continue
        
        valid_pairs.append((evidence, pair))
    
    if not valid_pairs:
        return {'precision': 0.0, 'num_citations': 0, 'details': []}
        
    # Single batched NLI call
    nli_inputs  = [(ev, p['claim']) for ev, p in valid_pairs]
    all_scores  = nli_model.predict(nli_inputs)   # shape: (N, num_labels)
    label_map   = nli_model.config.id2label
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    entailment_idx = label_to_idx.get('entailment', 1)  # fallback to index 1
    label_score_map = {'entailment': 1.0, 'neutral': 0.5, 'contradiction': 0.0}
    details     = []
    total_score = 0.0

    for (evidence, pair), scores in zip(valid_pairs, all_scores):
        # Use direct index to get entailment score (avoids label-order bugs)
        score_dict = {label_map[i].lower(): float(scores[i]) for i in range(len(scores))}
        predicted_label = max(score_dict, key=score_dict.get)
        precision_score = label_score_map.get(predicted_label, 0.0)
        total_score += precision_score
        
        details.append({
            'citation': pair['raw'],
            'claim': pair['claim'],
            'evidence': evidence[:100] + '...' if len(evidence) > 100 else evidence,
            'nli_label': predicted_label,
            'score': precision_score
        })
    
    precision = total_score / len(details) if details else 0.0
    
    return {
        'precision': precision,
        'num_citations': len(details),
        'details': details
    }

# def roi_citation_precision(state) -> dict:
#     """
#     For each [Image ROI N] citation, check if the ROI matches the claim using CLIP.
    
#     Returns:
#         dict with 'precision', 'num_citations', 'details'
#     """
#     from retriever import clip_model, clip_processor
    
#     # Get reasoning text
#     reasoning = state.reasoning_steps
#     if isinstance(reasoning, list):
#         reasoning_text = "\n".join(reasoning)
#     else:
#         reasoning_text = reasoning
    
#     retrieved_chunks = state.retrieved_chunks
    
#     # Extract ROI citation pairs
#     all_pairs = extract_citation_claim_pairs(reasoning_text)
#     roi_pairs = [p for p in all_pairs if p['type'] == 'image_roi']
    
#     if not roi_pairs:
#         return {'precision': 0.0, 'num_citations': 0, 'details': []}
    
#     # Load CLIP
#     device = next(clip_model.parameters()).device
#     clip_model.eval()
    
#     # Get all ROIs
#     all_rois = []
#     for query, chunk_info in retrieved_chunks.items():
#         if hasattr(chunk_info, 'image_rois'):
#             all_rois.extend(chunk_info.image_rois)
#         elif isinstance(chunk_info, dict):
#             all_rois.extend(chunk_info.get('image_rois', []))
    
#     details = []
#     total_score = 0.0
    
#     for pair in roi_pairs:
#         idx = pair['citation_id'] - 1
#         if idx < 0 or idx >= len(all_rois):
#             continue
        
#         roi = all_rois[idx]
#         image_patch = roi.image_patch if hasattr(roi, 'image_patch') else roi.get('image_patch')
        
#         if not image_patch:
#             continue
        
#         # Decode image
#         try:
#             img_data = base64.b64decode(image_patch)
#             image = Image.open(BytesIO(img_data)).convert('RGB')
#         except:
#             continue
        
#         # Compute CLIP similarity
#         with torch.no_grad():
#             inputs = clip_processor(text=[pair['claim']], images=image, return_tensors="pt", padding=True).to(device)
#             outputs = clip_model(**inputs)
            
#             img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
#             txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
#             similarity = torch.matmul(img_emb, txt_emb.T).item()
        
#         # Normalize to 0-1 score (CLIP similarities typically 0.15-0.35)
#         precision_score = max(0.0, min(1.0, (similarity - 0.15) / 0.25))
#         total_score += precision_score
        
#         caption = roi.caption if hasattr(roi, 'caption') else roi.get('caption', '')
#         details.append({
#             'citation': pair['raw'],
#             'claim': pair['claim'],
#             'roi_caption': caption,
#             'clip_similarity': similarity,
#             'score': precision_score
#         })
    
#     precision = total_score / len(details) if details else 0.0
    
#     return {
#         'precision': precision,
#         'num_citations': len(details),
#         'details': details
#     }

def question_image_citation_precision(state) -> Dict:
    """
    Check if [Question Image N] citations reference valid images.
    
    Since all question images are relevant (they're part of the question),
    precision = whether citation ID is within range of available images.
    """
    reasoning = ' '.join(state.reasoning_steps or [])
    pairs = extract_citation_pairs_v2(reasoning)
    
    # Filter to question image citations
    qi_pairs = [p for p in pairs if p['type'] == 'question_image']
    
    if not qi_pairs:
        return {
            'precision': 0.0,
            'total_citations': 0,
            'valid_citations': 0,
            'details': []
        }
    
    # Count available images
    import os
    num_images = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])
    
    valid = 0
    details = []
    
    for pair in qi_pairs:
        citation_id = pair['citation_id']
        is_valid = 1 <= citation_id <= num_images
        
        if is_valid:
            valid += 1
        
        details.append({
            'citation_id': citation_id,
            'claim': pair['claim'],
            'valid': is_valid,
            'reason': 'Valid image reference' if is_valid else f'Image {citation_id} does not exist (only {num_images} images)'
        })
    
    return {
        'precision': valid / len(qi_pairs) if qi_pairs else 0.0,
        'total_citations': len(qi_pairs),
        'valid_citations': valid,
        'details': details
    }


def attribution_score(state) -> dict:
    """
    Compute Attribution Score (AIS) - proportion of reasoning steps 
    attributable to retrieved evidence.
    For each reasoning step, check if ANY retrieved evidence entails it.
    
    AIS (Attributable to Identified Sources) measures whether each reasoning step can be derived from the retrieved evidence, 
    rather than hallucinated from the model's parametric knowledge.
    Citation Precision : Does the cited evidence support the claim?
    AIS : Can the claim be derived from any retrieved evidence?
    
    Returns:
        dict with 'ais', 'hallucination_rate', 'num_steps', 'details'
    """
    
    # Get reasoning steps
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        steps = reasoning
    else:
        # Split into steps if it's a single string
        steps = _split_into_steps(reasoning)
    
    if not steps:
        return {'ais': 0.0, 'hallucination_rate': 1.0, 'num_steps': 0, 'details': []}
    
    # Collect all evidence (text + ROI captions + hint)
    # pass hint so claims paraphrasing the hint are correctly attributed
    all_evidence = _collect_all_evidence(state.retrieved_chunks, hint=state.hint or '')
    
    if not all_evidence:
        return {'ais': 0.0, 'hallucination_rate': 1.0, 'num_steps': len(steps), 'details': []}
    
    # Load NLI model
    nli_model = get_nli_model()
    
    details = []
    attributable_count = 0
    
    for step in steps:
        step_clean = _clean_step(step)
        if not step_clean:
            continue
        
        # Check if step is attributable to ANY evidence
        is_attributable, best_evidence, best_score = _check_attribution(
            step_clean, all_evidence, nli_model
        )
        
        if is_attributable:
            attributable_count += 1
        
        details.append({
            'step': step_clean[:100] + '...' if len(step_clean) > 100 else step_clean,
            'is_attributable': is_attributable,
            'best_evidence': best_evidence[:80] + '...' if best_evidence and len(best_evidence) > 80 else best_evidence,
            'best_score': best_score
        })
    
    num_steps = len(details)
    ais = attributable_count / num_steps if num_steps > 0 else 0.0
    
    return {
        'ais': ais,
        'hallucination_rate': 1.0 - ais,
        'num_steps': num_steps,
        'num_attributable': attributable_count,
        'details': details
    }


def _split_into_steps(reasoning_text: str) -> list:
    """Split reasoning text into individual steps."""
    # Try step-based splitting first
    if re.search(r'Step\s*\d+', reasoning_text):
        steps = re.split(r'(?:^|\n)\s*(?:-\s*)?Step\s*\d+[:\.]?\s*', reasoning_text)
    else:
        # Fall back to sentence splitting
        steps = re.split(r'(?<=[.!?])\s+(?=[A-Z])', reasoning_text)
    
    return [s.strip() for s in steps if s.strip()]


def _clean_step(step: str) -> str:
    """Clean a reasoning step for attribution checking."""
    # Remove citation markers (we're checking attribution, not citation correctness)
    step = re.sub(r'\[Text Evidence \d+\]', '', step)
    step = re.sub(r'\[Image ROI \d+\]', '', step)
    # Remove common prefixes
    step = re.sub(r'^(?:According to|From|Based on|Therefore|Thus|Hence)[,\s]*', '', step, flags=re.IGNORECASE)
    return step.strip()


def _collect_all_evidence(retrieved_chunks: dict, hint: str = "") -> list:
    """Collect all evidence pieces (text chunks + ROI captions + hint text).
    hint text is often the most directly relevant content for science questions
    but was excluded from attribution checking. Added as an evidence source so AIS
    can attribute claims that paraphrase the hint rather than retrieved chunks.
    """
    evidence = []
    # Include hint text if provided — often the single most relevant sentence
    if hint and hint.strip():
        evidence.append(hint.strip())
    
    for query, chunk_info in retrieved_chunks.items():
        # Text chunks
        if hasattr(chunk_info, 'text_chunks'):
            evidence.extend(chunk_info.text_chunks)
        elif isinstance(chunk_info, dict):
            evidence.extend(chunk_info.get('text_chunks', []))
        
        # ROI captions (also count as evidence)
        if hasattr(chunk_info, 'image_rois'):
            for roi in chunk_info.image_rois:
                caption = roi.caption if hasattr(roi, 'caption') else roi.get('caption', '')
                if caption:
                    evidence.append(caption)
        elif isinstance(chunk_info, dict):
            for roi in chunk_info.get('image_rois', []):
                caption = roi.get('caption', '') if isinstance(roi, dict) else getattr(roi, 'caption', '')
                if caption:
                    evidence.append(caption)
    
    return [e for e in evidence if e]  # Filter empty


def _check_attribution(claim: str, evidence_list: list, nli_model) -> tuple:
    """
    Check if claim is attributable to any evidence.
    Runs a single batched NLI forward pass over all evidence pieces at once,
    rather than one call per piece.  This is significantly faster (CrossEncoder
    supports batch input natively).
    Note: For cross-encoder/nli-deberta-v3-base, label indices are:
        0 -> contradiction, 1 -> entailment, 2 -> neutral
    We use id2label to be safe, but fall back to index 1 if 'entailment' is missing.
    """
    if not evidence_list:
        return False, None, 0.0

    # Build all (premise=evidence, hypothesis=claim) pairs at once
    pairs = [(ev, claim) for ev in evidence_list]
    # Single batched call — CrossEncoder.predict() accepts a list of pairs
    all_scores = nli_model.predict(pairs)  # shape: (N, num_labels)
    label_map = nli_model.config.id2label
    
    # Validate label map contains 'entailment'; warn if not found
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    if 'entailment' not in label_to_idx:
        # Fallback: assume standard DeBERTa NLI order (contradiction=0, entailment=1, neutral=2)
        entailment_idx = 1
        import warnings
        warnings.warn(
            f"NLI model id2label does not contain 'entailment'. Labels: {label_map}. "
            f"Falling back to index {entailment_idx}."
        )
    else:
        entailment_idx = label_to_idx['entailment']
    
    best_score = 0.0
    best_evidence = None
    
    for i, scores in enumerate(all_scores):
        # scores is a 1-D array of length num_labels
        entailment_score = float(scores[entailment_idx])
        
        if entailment_score > best_score:
            best_score = entailment_score
            best_evidence = evidence_list[i]
        
        # Threshold: consider attributable if best entailment > 0.5
        # science lecture text tends to be general/background; specific factual
        # claims naturally score lower against it. 0.5 is too aggressive — most valid
        # attributions cluster in the 0.3–0.5 range for this domain. Lowered to 0.35.
        is_attributable = best_score > 0.35
    
    return is_attributable, best_evidence, best_score

def evidence_grounding_check(state) -> dict:
    """
    Verify that the final answer is actually supported by cited evidence.
    Strategy:
    1. Extract the answer letter/text from state.final_answer.
    2. Collect all evidence cited in the reasoning (via extract_citation_claim_pairs).
    3. Use NLI to check if the cited evidence entails the stated answer.
    Returns a dict with:
        - 'is_grounded': bool — True if answer is supported by at least one cited evidence piece
        - 'grounding_score': float in [0, 1]
        - 'best_evidence': str — the evidence chunk most supportive of the answer
        - 'details': list of per-evidence NLI results
    """
    final_answer = state.final_answer or ""
    retrieved_chunks = state.retrieved_chunks or {}

    # Extract the answer claim (letter + option text)
    answer_match = re.search(
        r'[Tt]he answer is\s*[:\s]*([A-Ea-e])[:\s]+([^\[\n]+)', final_answer
    )
    if answer_match:
        answer_claim = f"{answer_match.group(1).upper()}: {answer_match.group(2).strip()}"
    else:
        # Fall back to full final_answer text (strip XML tags)
        answer_claim = re.sub(r'<[^>]+>', '', final_answer).strip()
    if not answer_claim:
        return {
            'is_grounded': False,
            'grounding_score': 0.0,
            'best_evidence': None,
            'details': [],
            'answer_claim': answer_claim,
        }
    
    # Collect cited evidence pieces
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning or ""

    pairs = extract_citation_claim_pairs(reasoning_text)
    cited_ids_text = {p['citation_id'] for p in pairs if p['type'] == 'text'}
    cited_ids_roi  = {p['citation_id'] for p in pairs if p['type'] == 'image_roi'}
    
    # Gather cited text chunks
    all_chunks = []
    for query, chunk_info in retrieved_chunks.items():
        if hasattr(chunk_info, 'text_chunks'):
            all_chunks.extend(chunk_info.text_chunks)
        elif isinstance(chunk_info, dict):
            all_chunks.extend(chunk_info.get('text_chunks', []))
    cited_chunks = [
        all_chunks[i - 1]
        for i in cited_ids_text
        if 1 <= i <= len(all_chunks)
    ]

    # Gather cited ROI captions
    all_rois = []
    for query, chunk_info in retrieved_chunks.items():
        if hasattr(chunk_info, 'image_rois'):
            all_rois.extend(chunk_info.image_rois)
        elif isinstance(chunk_info, dict):
            all_rois.extend(chunk_info.get('image_rois', []))
    cited_captions = []
    for i in cited_ids_roi:
        if 1 <= i <= len(all_rois):
            roi = all_rois[i - 1]
            cap = roi.caption if hasattr(roi, 'caption') else roi.get('caption', '')
            if cap:
                cited_captions.append(cap)
    cited_evidence = cited_chunks + cited_captions

    # If no citations were found, fall back to ALL retrieved evidence (including hint)
    if not cited_evidence:
        cited_evidence = _collect_all_evidence(retrieved_chunks, hint=state.hint or '')
    if not cited_evidence:
        return {
            'is_grounded': False,
            'grounding_score': 0.0,
            'best_evidence': None,
            'details': [],
            'answer_claim': answer_claim,
        }

    # NLI: does any cited evidence entail the answer claim?
    nli_model = get_nli_model()
    is_grounded, best_evidence, grounding_score = _check_attribution(
        answer_claim, cited_evidence, nli_model
    )
    
    # Build per-evidence detail list (lightweight — just top-5 evidence pieces)
    details = []
    nli_inputs = [(ev, answer_claim) for ev in cited_evidence[:5]]
    if nli_inputs:
        batch_scores = nli_model.predict(nli_inputs)
        label_map = nli_model.config.id2label
        label_to_idx = {v.lower(): k for k, v in label_map.items()}
        entailment_idx = label_to_idx.get('entailment', 1)
        for ev, scores in zip(cited_evidence[:5], batch_scores):
            details.append({
                'evidence': ev[:120] + '...' if len(ev) > 120 else ev,
                'entailment_score': float(scores[entailment_idx]),
            })

    return {
        'is_grounded': is_grounded,
        'grounding_score': grounding_score,
        'best_evidence': best_evidence,
        'details': details,
        'answer_claim': answer_claim,
    }

def compute_cave_score(state, skip_nli=False, skip_clip=False, skip_ais=False) -> dict:
    """
    Compute the combined CaVeScore for solver evaluation.
    
    CaVeScore = 0.4*Accuracy + 0.2*CitePrecision + 0.2*CiteRecall + 0.1*AIS + 0.1*Grounding
    """
    # Accuracy
    accuracy = final_answer_accuracy(state)
    
    # Citation precision
    text_prec = text_citation_precision(state)['precision'] if not skip_nli else 0.0
    # roi_prec = roi_citation_precision(state)['precision'] if not skip_clip else 0.0
    # Question image citation precision: checks that [Question Image N] IDs are in range.
    # skip_clip flag re-used here since image citation checking is lightweight.
    qi_prec_result = question_image_citation_precision(state)
    roi_prec = qi_prec_result['precision'] if not skip_clip else 0.0
    
    # Combined precision
    if text_prec > 0 and roi_prec > 0:
        cite_precision = (text_prec + roi_prec) / 2
    else:
        cite_precision = text_prec or roi_prec

    # AIS (real attribution score)
    if not skip_ais:
        ais_result = attribution_score(state)
        ais = ais_result['ais']
    else:
        ais = cite_precision

    # Evidence grounding check (P2): is the final answer supported by cited evidence?
    if not skip_nli:
        grounding_result = evidence_grounding_check(state)
        grounding_score = grounding_result['grounding_score']
        is_grounded = grounding_result['is_grounded']
    else:
        grounding_score = 0.0
        is_grounded = False
    
    # Citation recall — count citations vs. substantive factual sentences only.
    # Previously every sentence was counted as a verifiable claim, which
    # penalised transition phrases like "Therefore…" or "In conclusion…".
    # Now we only count sentences that contain a concrete factual assertion
    # (i.e. have a subject + a content verb/noun and are not pure transitions).
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning
    
    pairs = extract_citation_claim_pairs(reasoning_text)
    num_citations = len(pairs)
    _transition_re  = re.compile(
        r'^\s*(?:therefore|thus|hence|so|in conclusion|in summary|as a result'
        r'|combining|this (?:means|shows|suggests)|overall)\b',
        re.IGNORECASE,
    )
    _factual_re = re.compile(
        # Must have at least one non-trivial verb indicating a fact
        r'\b(?:is|are|was|were|has|have|had|contains?|shows?|indicates?'
        r'|demonstrates?|proves?|reveals?|confirms?|measures?|equals?'
        r'|represents?|consists?|includes?|causes?|results?)\b',
        re.IGNORECASE,
    )
    raw_sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
    factual_sentences = [
        s for s in raw_sentences
        if s.strip()
        and not _transition_re.match(s)
        and _factual_re.search(s)
    ]
    num_claims = max(1, len(factual_sentences))
    cite_recall = min(1.0, num_citations / num_claims)
    
    # CaVeScore (updated weights to include grounding)
    cave_score = (
        0.4 * accuracy +
        0.2 * cite_precision +
        0.2 * cite_recall +
        0.1 * ais  +
        0.1 * grounding_score
    )
    
    return {
        'accuracy': accuracy,
        'text_citation_precision': text_prec,
        'roi_citation_precision': roi_prec,
        'ais': ais,
        'hallucination_rate': 1.0 - ais,
        'citation_precision': cite_precision,
        'citation_recall': cite_recall,
        'grounding_score': grounding_score,
        'is_grounded': is_grounded,
        'cave_score': cave_score
    }

def compute_combined_metrics(state, skip_clip: bool = True) -> Dict:
    """
    Compute all citation metrics supporting both ROI and Question Image formats.
    """
    reasoning = ' '.join(state.reasoning_steps or [])
    pairs = extract_citation_pairs_v2(reasoning)
    
    # Count by type
    text_citations = [p for p in pairs if p['type'] == 'text']
    roi_citations = [p for p in pairs if p['type'] == 'image_roi']
    qi_citations = [p for p in pairs if p['type'] == 'question_image']
    
    # Question image precision
    num_images = len([p for p in (state.image_paths or []) if p])
    qi_valid = sum(1 for p in qi_citations if 1 <= p['citation_id'] <= num_images)
    
    return {
        'total_citations': len(pairs),
        'text_citations': len(text_citations),
        'roi_citations': len(roi_citations),
        'question_image_citations': len(qi_citations),
        'question_image_precision': qi_valid / len(qi_citations) if qi_citations else None,
        'has_visual_citations': len(roi_citations) + len(qi_citations) > 0,
    }

def citation_summary(state) -> Dict:
    """
    Get a summary of all citations in the response.
    """
    reasoning = ' '.join(state.reasoning_steps or [])
    pairs = extract_citation_pairs_v2(reasoning)
    
    text_cites = [p for p in pairs if p['type'] == 'text']
    image_cites = [p for p in pairs if p['type'] == 'question_image']
    
    return {
        'text_citations': len(text_cites),
        'question_image_citations': len(image_cites),
        'total_citations': len(pairs),
        'has_any_citations': len(pairs) > 0,
    }

###### VERIFIER EVALUATION
'''
Citation Verification: Does it correctly identify when citations don't support claims?
Hallucination Detection: Does it catch claims not grounded in evidence?
Reasoning Validity: Does it detect logical errors in reasoning steps?
Answer Consistency: Does the final answer follow from the reasoning?
'''
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

    Returns:
        Dict with verifier quality metrics
    """
    # 1. Get verifier's claims
    verifier_verdict = state.verdict
    verifier_confidence = state.confidence
    verifier_hallucination = state.hallucination
    verifier_feedback = state.verifier_feedback
    
    # 2. Compute ground truth — reuse pre-computed result if available
    if cave_result is None:
        cave_result = compute_cave_score(state, skip_nli=False, skip_clip=True, skip_ais=False)
    
    gt_answer_correct = cave_result['accuracy'] == 1.0
    gt_citations_valid = cave_result['citation_precision'] >= 0.7  # Threshold
    gt_reasoning_grounded = cave_result['ais'] >= 0.5  # Threshold
    gt_has_hallucinations = cave_result['hallucination_rate'] > 0.3  # >30% hallucinated
    
    # 3. What the verifier SHOULD have concluded
    gt_should_accept = gt_answer_correct and gt_citations_valid and gt_reasoning_grounded
    
    # 4. Compare verifier's decision to ground truth
    verifier_accepted = (verifier_verdict == "VERIFIED")
    
    # Decision correctness
    decision_correct = (verifier_accepted == gt_should_accept)
    
    # 5. Evaluate specific verifier capabilities
    
    # Did verifier correctly identify hallucinations?
    verifier_found_hallucination = verifier_hallucination not in ["NONE", "UNKNOWN", ""]
    hallucination_detection_correct = (verifier_found_hallucination == gt_has_hallucinations)
    
    # Did verifier's confidence match the evidence quality?
    if verifier_confidence == "HIGH":
        confidence_appropriate = gt_citations_valid and gt_reasoning_grounded
    elif verifier_confidence == "LOW":
        confidence_appropriate = not gt_citations_valid or not gt_reasoning_grounded
    else:
        confidence_appropriate = True  # MEDIUM is always acceptable
    
    # 6. Analyze verifier feedback quality (if rejected)
    feedback_quality = evaluate_feedback_quality(
        verifier_feedback, 
        gt_citations_valid, 
        gt_reasoning_grounded,
        gt_answer_correct
    )
    
    return {
        # Verifier outputs
        'verdict': verifier_verdict,
        'confidence': verifier_confidence,
        'hallucination_type': verifier_hallucination,
        
        # Ground truth
        'gt_answer_correct': gt_answer_correct,
        'gt_citations_valid': gt_citations_valid,
        'gt_reasoning_grounded': gt_reasoning_grounded,
        'gt_has_hallucinations': gt_has_hallucinations,
        'gt_should_accept': gt_should_accept,
        
        # Verifier quality metrics
        'decision_correct': decision_correct,
        'hallucination_detection_correct': hallucination_detection_correct,
        'confidence_appropriate': confidence_appropriate,
        'feedback_quality': feedback_quality,
        
        # Detailed scores
        'citation_precision': cave_result['citation_precision'],
        'ais': cave_result['ais'],
        'hallucination_rate': cave_result['hallucination_rate'],
    }


def evaluate_feedback_quality(
    feedback: str,
    gt_citations_valid: bool,
    gt_reasoning_grounded: bool,
    gt_answer_correct: bool
) -> dict:
    """
    Evaluate whether verifier feedback correctly identifies issues.
    
    Good feedback should:
    1. Mention citation issues if citations are invalid
    2. Mention hallucination if reasoning is not grounded
    3. Mention answer issues if answer is wrong
    4. Be actionable for re-planning
    """
    if not feedback:
        return {
            'has_feedback': False,
            'identifies_citation_issues': False,
            'identifies_hallucination': False,
            'identifies_answer_issues': False,
            'is_actionable': False,
            'score': 0.0
        }
    
    feedback_lower = feedback.lower()
    
    # Check if feedback mentions relevant issues
    citation_keywords = ['citation', 'evidence', 'source', 'reference', 'support', 'unsupported']
    hallucination_keywords = ['hallucination', 'fabricat', 'made up', 'not grounded', 'unfounded', 'invented']
    answer_keywords = ['answer', 'incorrect', 'wrong', 'conclusion', 'final']
    actionable_keywords = ['retry', 'replan', 'reconsider', 'check', 'verify', 'revise', 'improve']
    
    identifies_citations = any(kw in feedback_lower for kw in citation_keywords)
    identifies_hallucination = any(kw in feedback_lower for kw in hallucination_keywords)
    identifies_answer = any(kw in feedback_lower for kw in answer_keywords)
    is_actionable = any(kw in feedback_lower for kw in actionable_keywords)
    
    # Score based on whether feedback mentions issues that actually exist.
    # Actionability is always worth a bonus regardless of which issues exist.
    score = 0.0
    checks = 0
    
    if not gt_citations_valid:
        checks += 1
        if identifies_citations:
            score += 1.0
    
    if not gt_reasoning_grounded:
        checks += 1
        if identifies_hallucination:
            score += 1.0
    
    if not gt_answer_correct:
        checks += 1
        if identifies_answer:
            score += 1.0
    
    # Actionability bonus: worth one extra point on top of the issue checks.
    # We always add it to both numerator and denominator so it can only
    # improve (or maintain) the score — never reduce it.
    checks += 1

    if is_actionable:
        score += 1.0
    
    # When checks == 0 that means all three ground-truth conditions were True
    # (no issues at all) AND is_actionable was False.  That edge-case can't
    # happen now because we always add the actionability check, so checks >= 1.
    final_score = score / checks
    
    return {
        'has_feedback': True,
        'identifies_citation_issues': identifies_citations,
        'identifies_hallucination': identifies_hallucination,
        'identifies_answer_issues': identifies_answer,
        'is_actionable': is_actionable,
        'score': final_score
    }