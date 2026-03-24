import math
import re
import numpy as np
from typing import Dict, List

###### RETRIEVER EVALUATION
def answer_support_recall(state) -> float:
    """
    NLI-based: does any retrieved chunk ENTAIL that the gold choice is correct?
    More meaningful than string-match recall for choice-based QA.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0
    
    from solver.solver_evals import get_nli_model  # lazy: avoids circular import

    nli_model = get_nli_model()
    all_chunks = [
        chunk for ci in state.retrieved_chunks.values()
        for chunk in ci.text_chunks[:3] if chunk and len(chunk.strip()) > 20
    ][:10]
    
    if not all_chunks:
        return 0.0
    
    # Hypothesis: "The answer is [gold_answer]" or just the gold answer string
    hypothesis = f"The correct answer is: {state.gold_answer}"
    pairs = [(chunk, hypothesis) for chunk in all_chunks]
    # CrossEncoder.predict() returns raw logits, not softmax probabilities.
    # Applying a probability threshold (0.5) directly to logits is unreliable —
    # a logit of 0.5 does not correspond to 50% confidence.
    # We use apply_softmax=True so scores are genuine probabilities in [0, 1].
    scores = nli_model.predict(pairs, apply_softmax=True)
    label_map = nli_model.config.id2label
    entail_idx = {v.lower(): k for k, v in label_map.items()}.get('entailment', 1)
    
    best_entailment = max(float(s[entail_idx]) for s in scores)
    return 1.0 if best_entailment > 0.5 else 0.0

def recall_at_k(state, k: int = 2) -> dict:
    """
    Compute Recall@K: Does top-K evidence support the gold answer?
    
    Embeds (question + gold_answer) together so short answer labels like
    "sturgeon" get enough context to match relevant evidence passages.
    Falls back to substring check as a secondary signal.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return {"recall": 0.0, "hits": {}}

    from retriever.retriever import text_model  # lazy: avoids circular import
    SIMILARITY_THRESHOLD = 0.35  # lowered from 0.5 — contextual query is denser

    # Combine question + answer for a richer query embedding
    question_text = getattr(state, 'question', '') or ''
    gold_query = f"{question_text} {state.gold_answer}".strip()
    gold_emb = text_model.encode(gold_query, normalize_embeddings=True)

    hits = {}
    for subquery, chunk_info in state.retrieved_chunks.items():
        top_k = chunk_info.text_chunks[:k]
        hit = False
        if top_k:
            # Semantic match
            ev_embs = text_model.encode(top_k, normalize_embeddings=True)
            sims = ev_embs @ gold_emb
            semantic_hit = bool(sims.max() > SIMILARITY_THRESHOLD)

            # Substring fallback for cases where answer is explicitly mentioned
            substring_hit = any(
                state.gold_answer.lower() in chunk.lower() for chunk in top_k
            )

            hit = semantic_hit or substring_hit
        hits[subquery] = hit

    recall = sum(hits.values()) / len(hits) if hits else 0.0
    return {"recall": recall, "hits": hits}

def precision_at_k(state, k: int = 2) -> float:
    """
    Compute Precision@K: What fraction of retrieved docs are relevant?
    Uses semantic similarity to avoid false positives/negatives from exact matching.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    from retriever.retriever import text_model  # lazy: avoids circular import
    SIMILARITY_THRESHOLD = 0.5

    gold_emb = text_model.encode(state.gold_answer, normalize_embeddings=True)
    all_evidence = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        all_evidence.extend(chunk_info.text_chunks[:k])

    if not all_evidence:
        return 0.0

    ev_embs = text_model.encode(all_evidence, normalize_embeddings=True)
    sims = ev_embs @ gold_emb
    relevant = int((sims > SIMILARITY_THRESHOLD).sum())
    return relevant / len(all_evidence)

def mean_reciprocal_rank(state) -> float:
    """
    Compute MRR: Average of 1/rank for first relevant document per query.
    Uses semantic similarity instead of exact substring match.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    from retriever.retriever import text_model  # lazy: avoids circular import
    SIMILARITY_THRESHOLD = 0.5

    gold_emb = text_model.encode(state.gold_answer, normalize_embeddings=True)
    rr_scores = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        text_evidences = chunk_info.text_chunks
        if not text_evidences:
            rr_scores.append(0.0)
            continue
        ev_embs = text_model.encode(text_evidences, normalize_embeddings=True)
        sims = ev_embs @ gold_emb
        hit_indices = [i for i, s in enumerate(sims) if s > SIMILARITY_THRESHOLD]
        if hit_indices:
            rr_scores.append(1.0 / (hit_indices[0] + 1))
        else:
            rr_scores.append(0.0)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

def ndcg_at_k(state, k: int = 2) -> float:
    """
    Compute NDCG@K: Normalized Discounted Cumulative Gain.
    Uses semantic similarity for relevance judgment.
    IDCG is computed assuming the best possible ranking of ALL corpus documents
    (i.e. k relevant docs at the top), not just the retrieved set.
    Computed per-subquery then averaged.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return 0.0

    from retriever.retriever import text_model  # lazy: avoids circular import
    SIMILARITY_THRESHOLD = 0.5

    gold_emb = text_model.encode(state.gold_answer, normalize_embeddings=True)
    ndcg_scores = []

    for subquery, chunk_info in state.retrieved_chunks.items():
        text_evidences = chunk_info.text_chunks
        if not text_evidences:
            ndcg_scores.append(0.0)
            continue

        ev_embs = text_model.encode(text_evidences[:k], normalize_embeddings=True)
        sims = ev_embs @ gold_emb
        relevances = [1 if s > SIMILARITY_THRESHOLD else 0 for s in sims]

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

        # IDCG: assume we could retrieve up to k relevant docs.
        # The number of actually relevant docs caps the ideal.
        num_relevant = sum(relevances)
        ideal_rels = [1] * num_relevant + [0] * (len(relevances) - num_relevant)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0