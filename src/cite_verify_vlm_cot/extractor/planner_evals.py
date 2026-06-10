import re


###### PLANNER EVALUATION
def planner_coverage_score(state) -> float:
    """Compute how well subqueries cover key concepts from question and choices."""
    if not state.subqueries:
        return 0.0

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "this",
        "that",
        "it",
    }

    def extract_terms(text: str) -> set:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {w for w in words if w not in stop_words}

    question_terms = extract_terms(state.question)
    choice_terms = set()
    for choice in state.choices or []:
        choice_terms.update(extract_terms(str(choice)))

    key_terms = question_terms | choice_terms
    if not key_terms:
        return 1.0

    subquery_text = " ".join(state.subqueries).lower()
    covered = sum(1 for term in key_terms if term in subquery_text)

    return covered / len(key_terms)


def planner_hit_rate(state, k: int = 2, threshold: float = 0.5) -> bool:
    """Check if any subquery led to retrieving the gold answer.
    Uses batch encoding for efficiency — calling text_model.encode() inside
    a loop creates one inference call per evidence chunk.
    """
    if not state.retrieved_chunks or not state.gold_answer:
        return False

    from retriever.retriever import text_model  # lazy: avoids circular import

    gold_emb = text_model.encode(state.gold_answer, normalize_embeddings=True)

    for subquery, chunk_info in state.retrieved_chunks.items():
        evidences = chunk_info.text_chunks[:k]
        if not evidences:
            continue
        # Encode all chunks for this subquery in a single call
        ev_embs = text_model.encode(evidences, normalize_embeddings=True)
        sims = ev_embs @ gold_emb  # cosine similarity (unit vectors)
        if bool(sims.max() > threshold):
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
    generic_starts = ["what is", "how does", "definition of", "explain", "describe"]

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
