import os
import re
from typing import Any, Dict, List, Optional

from citation_injector.citation_injector import _build_text_chunk_list
from sentence_transformers import CrossEncoder


# At module level
_nli_model_cache = {}

# Pattern for [Question Image N] citations
IMAGE_CITATION_PATTERN = r"\[Question Image (\d+)\]"

# Pattern for [Text Evidence N] citations
TEXT_CITATION_PATTERN = r"\[Text Evidence (\d+)\]"

# Combined pattern for any image citation (supports legacy ROI too)
IMAGE_CITATION_CLEAN_PATTERN = r"\[Question Image \d+\]"

TEXT_EVIDENCE_PATTERN = r"\[Text Evidence \d+\]"
ANY_IMAGE_CITATION = r"\[Question Image \d+\]|\[Image ROI \d+\]"


def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base"):
    """Get cached NLI model."""
    if model_name not in _nli_model_cache:
        _nli_model_cache[model_name] = CrossEncoder(model_name)
    return _nli_model_cache[model_name]


def extract_citation_claim_pairs(reasoning_text: str) -> List[Dict]:
    """
    Extract citation-claim pairs from reasoning text.
    Supports [Question Image N] format.
    """
    pairs = []

    sentences = re.split(r"[.!?]\s+", reasoning_text)

    for sentence in sentences:
        if not sentence.strip():
            continue

        # Find text citations
        for match in re.finditer(TEXT_CITATION_PATTERN, sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append(
                {
                    "citation_id": int(match.group(1)),
                    "type": "text",
                    "claim": claim,
                    "raw": match.group(0),
                    "sentence": sentence,
                }
            )

        # Find question image citations
        for match in re.finditer(IMAGE_CITATION_PATTERN, sentence):
            claim = _extract_claim(sentence, match.group(0), match.start())
            pairs.append(
                {
                    "citation_id": int(match.group(1)),
                    "type": "question_image",
                    "claim": claim,
                    "raw": match.group(0),
                    "sentence": sentence,
                }
            )

    return pairs


def _extract_claim(sentence: str, citation: str, citation_start: int) -> str:
    """Extract the claim associated with a citation."""
    # Pattern: "According to [Citation], CLAIM"
    pattern = r"^(?:According to|From|Based on|Looking at|In)\s*" + re.escape(citation) + r"\s*,\s*(.+)"
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return _clean_claim(match.group(1))

    # Pattern: "[Citation] shows/indicates that CLAIM"
    pattern = re.escape(citation) + r"\s+(?:shows?|indicates?|confirms?|reveals?)\s+(?:that\s+)?(.+)"
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return _clean_claim(match.group(1))

    # Pattern: "CLAIM [Citation]" (citation at end)
    if sentence.rstrip(".").endswith(citation):
        claim = sentence[:citation_start].strip()
        return _clean_claim(claim)

    # Fallback: sentence minus citation
    return _clean_claim(sentence.replace(citation, "").strip())


def _clean_claim(claim: str) -> str:
    """Clean extracted claim text."""
    claim = re.sub(TEXT_EVIDENCE_PATTERN, "", claim)
    claim = re.sub(ANY_IMAGE_CITATION, "", claim)
    claim = re.sub(r"\s+", " ", claim)
    return claim.strip(" ,;:.")


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
    If solver output extraction fails, also try verifier_answer
    (a single letter A-E) which the verifier reliably extracts from its own
    structured output even when the solver's CONCLUSION format is irregular.
    """
    solver_answer = state.final_answer or ""
    gold_answer = state.gold_answer or ""
    gold_index = getattr(state, "answer", None)  # int, 0-based

    # Extract predicted letter.
    # ScienceQA has up to 5 choices (A–E); the old A–D range silently failed
    # on any question where the correct answer was E.
    predicted_letter = None

    # Pattern 1: "The answer is A" / "The answer is: A"
    match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])", solver_answer)
    if match:
        predicted_letter = match.group(1).upper()

    # Pattern 2: "Answer: A"
    if not predicted_letter:
        match = re.search(r"[Aa]nswer[:\s]+([A-Ea-e])", solver_answer)
        if match:
            predicted_letter = match.group(1).upper()

    # Pattern 3: standalone letter e.g. just "A"
    # Only apply this on very short outputs (≤10 chars) to avoid extracting
    # letters that appear inside the reasoning body. For example, "Solution A
    # has five particles" contains "A" but the answer may actually be "B".
    # The CONCLUSION-block extraction above handles normal structured outputs;
    # this is only a last-resort fallback for degenerate single-character replies.
    if not predicted_letter and len(solver_answer.strip()) <= 10:
        match = re.search(r"\b([A-Ea-e])\b", solver_answer)
        if match:
            predicted_letter = match.group(1).upper()

    # Compare predicted letter against gold index
    if predicted_letter is not None and gold_index is not None:
        # "A" → 0, "B" → 1, "C" → 2, "D" → 3
        predicted_index = ord(predicted_letter) - ord("A")
        if predicted_index == gold_index:
            return 1.0

    # Fallback: check if solver wrote the full choice text
    # Handles cases where the model outputs "cutting paper" rather than "A"
    if gold_answer and gold_answer.strip().lower() in solver_answer.lower():
        return 1.0
    # If the solver output doesn't parse cleanly, use the verifier's
    # extracted letter. The verifier reliably outputs a single A-E letter in its
    # structured format
    verifier_answer = getattr(state, "verifier_answer", "") or ""
    if verifier_answer and len(verifier_answer) == 1 and verifier_answer in "ABCDE":
        verifier_index = ord(verifier_answer) - ord("A")
        if gold_index is not None and verifier_index == gold_index:
            return 1.0
    return 0.0


def get_text_evidence_by_id(evidence_id: int, retrieved_chunks: dict) -> str:
    """Get text evidence content by citation ID (1-indexed).

    Delegates to citation_injector._build_text_chunk_list — the single source
    of truth — so the chunk resolved here is always the same one that was
    labelled [Text Evidence N] in the solver prompt and verifier display.
    """
    all_chunks = _build_text_chunk_list(retrieved_chunks, total_cap=15, truncate=500)
    idx = evidence_id - 1  # convert to 0-indexed
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

    Returns
    -------
        dict with 'precision', 'num_citations', 'details'
    """
    # Get reasoning text
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning or ""

    # Get retrieved evidence
    retrieved_chunks = state.retrieved_chunks

    # Extract citation-claim pairs
    all_pairs = extract_citation_claim_pairs(reasoning_text)
    text_pairs = [p for p in all_pairs if p["type"] == "text"]

    if not text_pairs:
        return {"precision": 0.0, "num_citations": 0, "details": []}

    # Load NLI model
    nli_model = get_nli_model()

    # Build all (evidence, claim) pairs up front

    valid_pairs = []  # [(evidence_str, pair_dict), ...]
    skipped_ids = []

    for pair in text_pairs:
        evidence = get_text_evidence_by_id(pair["citation_id"], retrieved_chunks)

        if evidence is None:
            print(f"Warning: Text Evidence {pair['citation_id']} not found")
            skipped_ids.append(pair["citation_id"])
            continue

        valid_pairs.append((evidence, pair))

    if not valid_pairs:
        return {"precision": 0.0, "num_citations": 0, "details": []}

    # Single batched NLI call
    nli_inputs = [(ev, p["claim"]) for ev, p in valid_pairs]
    all_scores = nli_model.predict(nli_inputs)  # shape: (N, num_labels)
    label_map = nli_model.config.id2label
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    entailment_idx = label_to_idx.get("entailment", 1)  # fallback to index 1
    label_score_map = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}
    details = []
    total_score = 0.0

    for (evidence, pair), scores in zip(valid_pairs, all_scores):
        # Use direct index to get entailment score (avoids label-order bugs)
        score_dict = {label_map[i].lower(): float(scores[i]) for i in range(len(scores))}
        predicted_label = max(score_dict, key=score_dict.get)
        precision_score = label_score_map.get(predicted_label, 0.0)
        total_score += precision_score

        details.append(
            {
                "citation": pair["raw"],
                "claim": pair["claim"],
                "evidence": evidence[:100] + "..." if len(evidence) > 100 else evidence,
                "nli_label": predicted_label,
                "score": precision_score,
            }
        )

    precision = total_score / len(details) if details else 0.0

    return {"precision": precision, "num_citations": len(details), "details": details}


def question_image_citation_precision(state) -> Dict:
    """
    Check if [Question Image N] citations reference valid images.

    Since all question images are relevant (they're part of the question),
    precision = whether citation ID is within range of available images.
    """
    reasoning = " ".join(state.reasoning_steps or [])
    pairs = extract_citation_claim_pairs(reasoning)

    # Filter to question image citations
    qi_pairs = [p for p in pairs if p["type"] == "question_image"]

    if not qi_pairs:
        return {"precision": 0.0, "total_citations": 0, "valid_citations": 0, "details": []}

    # Count available images
    import os

    num_images = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])

    valid = 0
    details = []

    for pair in qi_pairs:
        citation_id = pair["citation_id"]
        is_valid = 1 <= citation_id <= num_images

        if is_valid:
            valid += 1

        details.append(
            {
                "citation_id": citation_id,
                "claim": pair["claim"],
                "valid": is_valid,
                "reason": "Valid image reference"
                if is_valid
                else f"Image {citation_id} does not exist (only {num_images} images)",
            }
        )

    return {
        "precision": valid / len(qi_pairs) if qi_pairs else 0.0,
        "total_citations": len(qi_pairs),
        "valid_citations": valid,
        "details": details,
    }


def question_image_citation_coverage(state) -> Dict:
    """
    Check if samples WITH images have Question Image citations.

    Returns
    -------
        - has_images: whether this sample has question images
        - has_qi_citations: whether any [Question Image N] citations exist
        - coverage: 1.0 if has_qi_citations OR no images, 0.0 otherwise
        - qi_citation_count: number of Question Image citations
    """
    # Count available images
    num_images = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])
    has_images = num_images > 0

    # Count Question Image citations
    reasoning = " ".join(state.reasoning_steps or [])
    pairs = extract_citation_claim_pairs(reasoning)
    qi_pairs = [p for p in pairs if p["type"] == "question_image"]
    qi_count = len(qi_pairs)
    has_qi_citations = qi_count > 0
    # Coverage: did we cite images when images were available?
    if not has_images:
        # No images to cite - not applicable (return 1.0 to not penalize)
        coverage = 1.0
    elif has_qi_citations:
        # Has images AND cited them
        coverage = 1.0
    else:
        # Has images but didn't cite any
        coverage = 0.0
    return {
        "has_images": has_images,
        "num_images": num_images,
        "has_qi_citations": has_qi_citations,
        "qi_citation_count": qi_count,
        "coverage": coverage,
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

    Returns
    -------
        dict with 'ais', 'hallucination_rate', 'num_steps', 'details'
    """
    # Get reasoning steps.
    # solver.py always sets state.reasoning_steps = [full_output] — a single-element
    # list containing the entire model output.  Taking the list directly as `steps`
    # means the whole output is evaluated as one monolithic "step", so NLI either
    # attributes everything (score=1.0) or nothing (score=0.0) depending on whether
    # any sentence in the long string gets an entailment hit.  We must always
    # flatten to a joined string and then split into individual sentences/steps.
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning or ""
    steps = _split_into_steps(reasoning_text)

    if not steps:
        return {"ais": 0.0, "hallucination_rate": 1.0, "num_steps": 0, "details": []}

    # Collect all evidence (text + ROI captions + hint)
    # pass hint so claims paraphrasing the hint are correctly attributed
    all_evidence = _collect_all_evidence(
        state.retrieved_chunks, hint=state.hint or "", question=state.question or "", lecture=state.lecture or ""
    )

    if not all_evidence:
        return {"ais": 0.0, "hallucination_rate": 1.0, "num_steps": len(steps), "details": []}

    # Load NLI model
    nli_model = get_nli_model()

    details = []
    attributable_count = 0

    # Patterns that indicate a sentence is merely restating the question/choices
    # rather than making a factual claim — skip these for AIS.
    _meta_re = re.compile(
        r"^\s*(?:the question (?:asks?|is|states?)|we need to (?:find|determine|identify)|"
        r"i (?:need to|must|should|will)|this (?:question|problem)|"
        r"the (?:task|goal|objective) is)",
        re.IGNORECASE,
    )

    for step in steps:
        step_clean = _clean_step(step)
        if not step_clean:
            continue

        # Skip very short fragments (< 15 chars) — likely artefacts of splitting
        if len(step_clean) < 15:
            continue

        # Skip sentences that only restate the question/task framing
        if _meta_re.match(step_clean):
            continue

        # Check if step is attributable to ANY evidence
        is_attributable, best_evidence, best_score = _check_attribution(step_clean, all_evidence, nli_model)

        if is_attributable:
            attributable_count += 1

        details.append(
            {
                "step": step_clean[:100] + "..." if len(step_clean) > 100 else step_clean,
                "is_attributable": is_attributable,
                "best_evidence": best_evidence[:80] + "..."
                if best_evidence and len(best_evidence) > 80
                else best_evidence,
                "best_score": best_score,
            }
        )

    num_steps = len(details)
    ais = attributable_count / num_steps if num_steps > 0 else 0.0

    return {
        "ais": ais,
        "hallucination_rate": 1.0 - ais,
        "num_steps": num_steps,
        "num_attributable": attributable_count,
        "details": details,
    }


def _split_into_steps(reasoning_text: str) -> list:
    """Split reasoning text into individual steps, excluding structural/non-factual sections."""
    # Strip SUMMARY and OBSERVATIONS/CAPTION blocks — these are either structural
    # framing or image-derived observations that cannot be attributed to text
    # evidence and would unfairly inflate the hallucination_rate.
    text = re.sub(
        r"<(?:SUMMARY|OBSERVATIONS|CAPTION)>.*?</(?:SUMMARY|OBSERVATIONS|CAPTION)>",
        "",
        reasoning_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Also strip XML-like structural tags themselves
    text = re.sub(r"</?(?:SUMMARY|OBSERVATIONS|CAPTION|CONCLUSION|REASONING)>", "", text, flags=re.IGNORECASE)

    # Transition-only sentences do not carry new factual claims and cannot
    # be attributed to evidence; skip them to avoid penalising valid reasoning.
    _transition_re = re.compile(
        r"^\s*(?:therefore|thus|hence|so\b|in conclusion|in summary|as a result"
        r"|combining|this (?:means|shows|suggests)|overall|given (?:this|that)|"
        r"based on (?:the (?:above|foregoing))|to summarize)",
        re.IGNORECASE,
    )
    # Try step-based splitting first
    if re.search(r"Step\s*\d+", text):
        steps = re.split(r"(?:^|\n)\s*(?:-\s*)?Step\s*\d+[:\.]?\s*", text)
    else:
        # Fall back to sentence splitting
        steps = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    return [s.strip() for s in steps if s.strip() and not _transition_re.match(s.strip())]


def _clean_step(step: str) -> str:
    """Clean a reasoning step for attribution checking."""
    # Remove citation markers (we're checking attribution, not citation correctness)
    step = re.sub(r"\[Text Evidence \d+\]", "", step)
    step = re.sub(r"\[Image ROI \d+\]", "", step)
    step = re.sub(r"\[Question Image \d+\]", "", step)
    # Remove common prefixes
    step = re.sub(r"^(?:According to|From|Based on|Therefore|Thus|Hence)[,\s]*", "", step, flags=re.IGNORECASE)
    return step.strip()


def _collect_all_evidence(retrieved_chunks: dict, hint: str = "", question: str = "", lecture: str = "") -> list:
    """Collect all evidence pieces (text chunks + ROI captions + hint text).
    hint text is often the most directly relevant content for science questions
    but was excluded from attribution checking. Added as an evidence source so AIS
    can attribute claims that paraphrase the hint rather than retrieved chunks.
    """
    evidence = []
    # Include hint text if provided — often the single most relevant sentence
    if hint and hint.strip():
        evidence.append(hint.strip())

    if question and question.strip():
        evidence.append(question.strip())

    if lecture and lecture.strip():
        evidence.append(lecture.strip())

    for query, chunk_info in retrieved_chunks.items():
        # Text chunks
        if hasattr(chunk_info, "text_chunks"):
            evidence.extend(chunk_info.text_chunks)
        elif isinstance(chunk_info, dict):
            evidence.extend(chunk_info.get("text_chunks", []))

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
    # Single batched call — CrossEncoder.predict() accepts a list of pairs.
    # apply_softmax=True converts logits to probabilities so the 0.35 threshold
    # is meaningful: a raw logit of 0.35 does not correspond to 35% confidence,
    # but a softmax probability of 0.35 does.
    all_scores = nli_model.predict(pairs, apply_softmax=True)  # shape: (N, num_labels)
    label_map = nli_model.config.id2label

    # Validate label map contains 'entailment'; warn if not found
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    if "entailment" not in label_to_idx:
        # Fallback: assume standard DeBERTa NLI order (contradiction=0, entailment=1, neutral=2)
        entailment_idx = 1
        import warnings

        warnings.warn(
            f"NLI model id2label does not contain 'entailment'. Labels: {label_map}. "
            f"Falling back to index {entailment_idx}."
        )
    else:
        entailment_idx = label_to_idx["entailment"]

    best_score = 0.0
    best_evidence = None

    for i, scores in enumerate(all_scores):
        # scores is a 1-D array of length num_labels
        entailment_score = float(scores[entailment_idx])

        if entailment_score > best_score:
            best_score = entailment_score
            best_evidence = evidence_list[i]

    # Evaluate attribution only after scanning all evidence.
    # science lecture text tends to be general/background; specific factual
    # claims naturally score lower against it. 0.5 is too aggressive — most valid
    # attributions cluster in the 0.3–0.5 range for this domain. Lowered to 0.35.
    is_attributable = best_score > 0.28

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
    answer_match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])[:\s]+([^\[\n]+)", final_answer)
    if answer_match:
        answer_claim = f"{answer_match.group(1).upper()}: {answer_match.group(2).strip()}"
    else:
        # Fall back to full final_answer text (strip XML tags)
        answer_claim = re.sub(r"<[^>]+>", "", final_answer).strip()
    if not answer_claim:
        return {
            "is_grounded": False,
            "grounding_score": 0.0,
            "best_evidence": None,
            "details": [],
            "answer_claim": answer_claim,
        }

    # Collect cited evidence pieces
    reasoning = state.reasoning_steps
    if isinstance(reasoning, list):
        reasoning_text = "\n".join(reasoning)
    else:
        reasoning_text = reasoning or ""

    pairs = extract_citation_claim_pairs(reasoning_text)
    cited_ids_text = {p["citation_id"] for p in pairs if p["type"] == "text"}
    # cited_ids_roi  = {p['citation_id'] for p in pairs if p['type'] == 'image_roi'}

    # Gather cited text chunks.
    # cited_ids_text holds citation IDs from the solver's [Text Evidence N] labels.
    # Those IDs were assigned by solver.format_evidence which: skips "_" keys,
    # filters len > 10, caps each query at 5, caps total at 10.
    # We must apply the same rules here so all_chunks[i-1] resolves to the same
    # chunk the solver labelled [Text Evidence i].
    # Build the canonical chunk list — same ordering and cap as solver/verifier/
    # citation_injector — so cited_ids_text resolves to the correct chunks.
    all_chunks = _build_text_chunk_list(retrieved_chunks, total_cap=15, truncate=500)
    cited_chunks = [all_chunks[i - 1] for i in cited_ids_text if 1 <= i <= len(all_chunks)]

    # Gather cited ROI captions
    cited_evidence = cited_chunks

    # If no citations were found, fall back to ALL retrieved evidence (including hint)
    if not cited_evidence:
        cited_evidence = _collect_all_evidence(
            retrieved_chunks, hint=state.hint or "", question=state.question or "", lecture=state.lecture or ""
        )
    if not cited_evidence:
        return {
            "is_grounded": False,
            "grounding_score": 0.0,
            "best_evidence": None,
            "details": [],
            "answer_claim": answer_claim,
        }

    # NLI: does any cited evidence entail the answer claim?
    nli_model = get_nli_model()
    is_grounded, best_evidence, grounding_score = _check_attribution(answer_claim, cited_evidence, nli_model)

    # Build per-evidence detail list (top-5) by reusing the scores already
    # computed inside _check_attribution via a single additional batched call
    # over the (small) top-5 slice — avoids running the full evidence list twice.
    details = []
    detail_evidence = cited_evidence[:5]
    if detail_evidence:
        label_map = nli_model.config.id2label
        label_to_idx = {v.lower(): k for k, v in label_map.items()}
        entailment_idx = label_to_idx.get("entailment", 1)
        nli_inputs = [(ev, answer_claim) for ev in detail_evidence]
        batch_scores = nli_model.predict(nli_inputs, apply_softmax=True)
        for ev, scores in zip(detail_evidence, batch_scores):
            details.append(
                {
                    "evidence": ev[:120] + "..." if len(ev) > 120 else ev,
                    "entailment_score": float(scores[entailment_idx]),
                }
            )

    return {
        "is_grounded": is_grounded,
        "grounding_score": grounding_score,
        "best_evidence": best_evidence,
        "details": details,
        "answer_claim": answer_claim,
    }


# Experiment 4: CaVeScore weight configurations for sensitivity analysis.
# Each entry is a dict with keys matching the five CaVeScore components.
# Weights within each config must sum to 1.0.
# WEIGHT_CONFIGS dict with 7 named configurations — default, accuracy-heavy, citation-heavy,
# uniform, AIS-heavy, grounding-heavy, recall-skewed, all summing to 1.0.

WEIGHT_CONFIGS: Dict[str, Dict[str, float]] = {
    # Default (published weights)
    "default": {
        "accuracy": 0.4,
        "citation_precision": 0.2,
        "citation_recall": 0.2,
        "ais": 0.1,
        "grounding": 0.1,
    },
    # Heavy emphasis on correctness — treats citation quality as secondary
    "accuracy_heavy": {
        "accuracy": 0.6,
        "citation_precision": 0.1,
        "citation_recall": 0.1,
        "ais": 0.1,
        "grounding": 0.1,
    },
    # Heavy emphasis on citation quality — penalises uncited claims strongly
    "citation_heavy": {
        "accuracy": 0.2,
        "citation_precision": 0.3,
        "citation_recall": 0.3,
        "ais": 0.1,
        "grounding": 0.1,
    },
    # Uniform — no component is privileged
    "uniform": {
        "accuracy": 0.2,
        "citation_precision": 0.2,
        "citation_recall": 0.2,
        "ais": 0.2,
        "grounding": 0.2,
    },
    # AIS-heavy — emphasises neural attribution quality
    "ais_heavy": {
        "accuracy": 0.3,
        "citation_precision": 0.15,
        "citation_recall": 0.15,
        "ais": 0.25,
        "grounding": 0.15,
    },
    # Grounding-heavy — penalises answers unsupported by cited passages
    "grounding_heavy": {
        "accuracy": 0.3,
        "citation_precision": 0.15,
        "citation_recall": 0.15,
        "ais": 0.15,
        "grounding": 0.25,
    },
    # Recall-skewed — prioritises coverage of factual claims with citations
    "recall_skewed": {
        "accuracy": 0.35,
        "citation_precision": 0.1,
        "citation_recall": 0.35,
        "ais": 0.1,
        "grounding": 0.1,
    },
}


def compute_cave_score(
    state,
    skip_nli: bool = False,
    skip_clip: bool = False,
    skip_ais: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Compute the combined CaVeScore for solver evaluation.

    Default formula:
        CaVeScore = 0.4*Accuracy + 0.2*CitePrecision + 0.2*CiteRecall
                    + 0.1*AIS + 0.1*Grounding

    Pass ``weights`` to override the default configuration, e.g.::

        from solver.solver_evals import WEIGHT_CONFIGS

        result = compute_cave_score(state, weights=WEIGHT_CONFIGS["uniform"])

    The ``weights`` dict must contain the five keys: accuracy,
    citation_precision, citation_recall, ais, grounding.  Values must sum
    to 1.0 (enforced via assertion).
    """
    # Accuracy
    accuracy = final_answer_accuracy(state)

    # Citation precision
    text_prec = text_citation_precision(state)["precision"] if not skip_nli else 0.0
    # roi_prec = roi_citation_precision(state)['precision'] if not skip_clip else 0.0
    # Question image citation precision: checks that [Question Image N] IDs are in range.
    # skip_clip flag re-used here since image citation checking is lightweight.
    qi_prec_result = question_image_citation_precision(state)
    qi_prec = qi_prec_result["precision"] if not skip_clip else 0.0

    # Question image citation coverage
    qi_coverage_result = question_image_citation_coverage(state)
    # Combined precision (text + question image).
    # Count how many citation types are actually present in this sample so the
    # average denominator reflects reality.
    # the old `text_prec or qi_prec` short-circuits to qi_prec whenever
    # text_prec == 0.0, silently hiding cases where all text citations are wrong.
    reasoning_text = " ".join(state.reasoning_steps or [])
    has_text_citations = bool(re.search(r"\[Text Evidence \d+\]", reasoning_text))
    has_qi_citations = bool(re.search(r"\[Question Image \d+\]", reasoning_text))

    if has_text_citations and has_qi_citations:
        cite_precision = (text_prec + qi_prec) / 2
    elif has_text_citations:
        cite_precision = text_prec
    elif has_qi_citations:
        cite_precision = qi_prec
    else:
        cite_precision = 0.0

    # AIS (real attribution score)
    if not skip_ais:
        ais_result = attribution_score(state)
        ais = ais_result["ais"]
    else:
        # AIS and citation precision are distinct metrics; using one as a proxy
        # for the other produces misleading composite scores.  Return 0.0 to
        # signal that AIS was not computed rather than silently substituting an
        # unrelated value.
        ais = 0.0

    # Evidence grounding check (P2): is the final answer supported by cited evidence?
    if not skip_nli:
        grounding_result = evidence_grounding_check(state)
        grounding_score = grounding_result["grounding_score"]
        is_grounded = grounding_result["is_grounded"]
    else:
        grounding_score = 0.0
        is_grounded = False

    # Citation recall — count citations vs. substantive factual sentences only.
    # Previously every sentence was counted as a verifiable claim, which
    # penalised transition phrases like "Therefore…" or "In conclusion…".
    # Now we only count sentences that contain a concrete factual assertion
    # (i.e. have a subject + a content verb/noun and are not pure transitions).

    pairs = extract_citation_claim_pairs(reasoning_text)
    num_citations = len(pairs)
    _transition_re = re.compile(
        r"^\s*(?:therefore|thus|hence|so|in conclusion|in summary|as a result"
        r"|combining|this (?:means|shows|suggests)|overall)\b",
        re.IGNORECASE,
    )
    _factual_re = re.compile(
        # Must have at least one non-trivial verb indicating a fact
        r"\b(?:is|are|was|were|has|have|had|contains?|shows?|indicates?"
        r"|demonstrates?|proves?|reveals?|confirms?|measures?|equals?"
        r"|represents?|consists?|includes?|causes?|results?)\b",
        re.IGNORECASE,
    )
    raw_sentences = re.split(r"(?<=[.!?])\s+", reasoning_text)
    factual_sentences = [
        s for s in raw_sentences if s.strip() and not _transition_re.match(s) and _factual_re.search(s)
    ]
    num_claims = max(1, len(factual_sentences))
    cite_recall = min(1.0, num_citations / num_claims)

    # Resolve weights — fall back to published defaults when not supplied.
    if weights is None:
        weights = WEIGHT_CONFIGS["default"]
    _required_keys = {"accuracy", "citation_precision", "citation_recall", "ais", "grounding"}
    assert _required_keys == set(weights.keys()), (
        f"weights dict must contain exactly {_required_keys}, got {set(weights.keys())}"
    )
    _weight_sum = sum(weights.values())
    assert abs(_weight_sum - 1.0) < 1e-6, f"weights must sum to 1.0, got {_weight_sum:.6f}"

    # CaVeScore
    cave_score = (
        weights["accuracy"] * accuracy
        + weights["citation_precision"] * cite_precision
        + weights["citation_recall"] * cite_recall
        + weights["ais"] * ais
        + weights["grounding"] * grounding_score
    )

    # Count question images
    num_question_images = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])

    return {
        "accuracy": accuracy,
        "text_citation_precision": text_prec,
        "qi_citation_precision": qi_prec,
        "ais": ais,
        "hallucination_rate": 1.0 - ais,
        "citation_precision": cite_precision,
        "citation_recall": cite_recall,
        "grounding_score": grounding_score,
        "is_grounded": is_grounded,
        "cave_score": cave_score,
        # Question Image citation metrics
        "qi_citation_coverage": qi_coverage_result["coverage"],
        "qi_citation_count": qi_coverage_result["qi_citation_count"],
        "num_question_images": num_question_images,
    }


# Experiment 4: Weight sensitivity analysis helpers
def _cave_score_from_components(
    components: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Compute a single CaVeScore from pre-extracted metric components and a
    weight configuration dict.

    ``components`` must have keys: accuracy, citation_precision,
    citation_recall, ais, grounding_score.
    ``weights`` must have keys: accuracy, citation_precision, citation_recall,
    ais, grounding  (note: grounding_score → grounding in the weight key).
    """
    return (
        weights["accuracy"] * components["accuracy"]
        + weights["citation_precision"] * components["citation_precision"]
        + weights["citation_recall"] * components["citation_recall"]
        + weights["ais"] * components["ais"]
        + weights["grounding"] * components["grounding_score"]
    )


def run_weight_sensitivity_analysis(
    cave_results: List[Dict[str, float]],
    configs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Re-score a list of pre-computed CaVeScore component dicts under every
    named weight configuration and return aggregate statistics.

    Parameters
    ----------
    cave_results : list of dict
        Each dict is the return value of ``compute_cave_score(state)`` for one
        sample — it must contain the keys accuracy, citation_precision,
        citation_recall, ais, and grounding_score.
    configs : dict, optional
        Named weight configurations to evaluate.  Defaults to
        ``WEIGHT_CONFIGS`` (all seven built-in configurations).

    Returns
    -------
    dict
        Keyed by config name.  Each value is a dict with:
            mean_cave_score   – mean CaVeScore across all samples
            std_cave_score    – standard deviation
            min_cave_score    – minimum per-sample score
            max_cave_score    – maximum per-sample score
            delta_vs_default  – difference in mean score vs the "default" config
            weights           – the weight dict used
            n_samples         – number of samples evaluated

    Example
    -------
    >>> results = [compute_cave_score(s) for s in state_list]
    >>> summary = run_weight_sensitivity_analysis(results)
    >>> for cfg_name, stats in summary.items():
    ...     print(f"{cfg_name:20s}  mean={stats['mean_cave_score']:.4f}  Δdefault={stats['delta_vs_default']:+.4f}")
    """
    import statistics

    if configs is None:
        configs = WEIGHT_CONFIGS

    if not cave_results:
        raise ValueError("cave_results is empty — pass at least one sample dict.")

    analysis: Dict[str, Dict[str, Any]] = {}

    for config_name, weights in configs.items():
        scores = [_cave_score_from_components(r, weights) for r in cave_results]
        analysis[config_name] = {
            "mean_cave_score": statistics.mean(scores),
            "std_cave_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_cave_score": min(scores),
            "max_cave_score": max(scores),
            "delta_vs_default": 0.0,  # filled in below
            "weights": weights,
            "n_samples": len(scores),
        }

    # Compute Δ relative to the default configuration.
    default_mean = analysis.get("default", {}).get("mean_cave_score", 0.0)
    for stats in analysis.values():
        stats["delta_vs_default"] = stats["mean_cave_score"] - default_mean

    return analysis


def compute_combined_metrics(state, skip_clip: bool = True) -> Dict:
    """
    Compute all citation metrics supporting Question Image format.
    """
    reasoning = " ".join(state.reasoning_steps or [])
    pairs = extract_citation_claim_pairs(reasoning)

    # Count by type
    text_citations = [p for p in pairs if p["type"] == "text"]
    roi_citations = [p for p in pairs if p["type"] == "image_roi"]
    qi_citations = [p for p in pairs if p["type"] == "question_image"]

    # Count available images the same way question_image_citation_precision does:
    # only files that actually exist on disk.
    import os

    num_images = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])
    qi_valid = sum(1 for p in qi_citations if 1 <= p["citation_id"] <= num_images)

    return {
        "total_citations": len(pairs),
        "text_citations": len(text_citations),
        "roi_citations": len(roi_citations),
        "question_image_citations": len(qi_citations),
        "question_image_precision": qi_valid / len(qi_citations) if qi_citations else None,
        "has_visual_citations": len(qi_citations) > 0,
    }


def citation_summary(state) -> Dict:
    """
    Get a summary of all citations in the response.
    """
    reasoning = " ".join(state.reasoning_steps or [])
    pairs = extract_citation_claim_pairs(reasoning)

    text_cites = [p for p in pairs if p["type"] == "text"]
    image_cites = [p for p in pairs if p["type"] == "question_image"]

    return {
        "text_citations": len(text_cites),
        "question_image_citations": len(image_cites),
        "total_citations": len(pairs),
        "has_any_citations": len(pairs) > 0,
    }
