import json
import os
import re
from typing import Dict, List

import torch
from langgraph.graph import END, StateGraph
from tracer import tracer


try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # pip install qwen-vl-utils

# Import State and step functions
from functools import partial

from citation_injector.citation_injector import _build_text_chunk_list, inject_citations_step
from extractor.planner import planner_step
from prompts import VERIFIER_PROMPT_TEMPLATE
from retriever.retriever import retriever_step
from solver.solver import solver_step_with_citation_retry
from utils import State


def _extract_choice_letter(text: str) -> str:
    """Return the first standalone multiple-choice letter from text, or INCONCLUSIVE."""
    match = re.search(r"\b([A-E])\b", text or "", re.IGNORECASE)
    return match.group(1).upper() if match else "INCONCLUSIVE"


def _extract_labeled_choice(output: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}:\s*\*{{0,2}}\[?([A-E]|INCONCLUSIVE)\]?\*{{0,2}}",
        output or "",
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else "INCONCLUSIVE"


def _first_parameter_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_topic_from_claim(claim: str) -> str:
    """Extract the topic being discussed from a claim"""
    # Remove citation markers
    clean = re.sub(r"\[.*?\]", "", claim)
    # Get main subject (simple heuristic)
    words = clean.split()
    return " ".join(words[:10]) + "..." if len(words) > 10 else clean


def extract_key_terms(question: str, choices: List[str]) -> List[str]:
    """Extract important terms from question and choices"""
    key_terms = set()
    # Extract capitalized terms (likely proper nouns or important concepts)
    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    key_terms.update(capitalized)
    # Extract numbers and units
    numbers_units = re.findall(r"\d+(?:\.\d+)?\s*(?:km|m|cm|kg|g|°C|°F|mph|%)?", question)
    key_terms.update(numbers_units)
    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', question)
    key_terms.update(quoted)
    # Extract important words from choices (nouns, likely)
    for choice in choices:
        words = [
            w
            for w in choice.split()
            if len(w) > 3 and w.lower() not in {"that", "this", "with", "from", "have", "been", "were", "what", "when"}
        ]
        key_terms.update(words[:2])
    return list(key_terms)


def generate_targeted_feedback(
    hallucination_type: str,
    hallucination_details: List[Dict[str, str]],
    question: str,
    previous_queries: List[str],
    choices: List[str],
) -> str:
    """
    Generate specific, actionable feedback based on hallucination analysis
    """
    if hallucination_type == "NONE DETECTED":
        return None

    feedback_parts = []

    # Analyze what went wrong
    fake_citations = [h for h in hallucination_details if "fake citation" in h.get("issue", "").lower()]
    misrepresented = [
        h
        for h in hallucination_details
        if "misrepresented" in h.get("issue", "").lower() or "not in evidence" in h.get("issue", "").lower()
    ]
    fabricated = [h for h in hallucination_details if "fabricated" in h.get("issue", "").lower()]

    # Header
    feedback_parts.append(" VERIFICATION FAILED\n")

    if fake_citations:
        feedback_parts.append("PROBLEM: Solver cited evidence that doesn't exist")
        feedback_parts.append("\nMissing evidence types:")
        for h in fake_citations[:3]:  # Show top 3
            if "Text Evidence" in h["claim"]:
                feedback_parts.append(f"  • Needed text about: {extract_topic_from_claim(h['claim'])}")
            elif "Question Image" in h["claim"]:
                feedback_parts.append(f"  • Needed image showing: {extract_topic_from_claim(h['claim'])}")

        feedback_parts.append("\n ACTION: Generate queries to retrieve this missing evidence")
        feedback_parts.append("STRATEGY:")
        feedback_parts.append("  1. Use EXACT terms from the question")
        feedback_parts.append("  2. Include choice keywords (e.g., from these options: " + ", ".join(choices[:2]) + ")")
        feedback_parts.append("  3. Add 'definition', 'properties', 'characteristics' for concepts")
        feedback_parts.append("  4. For images: use 'anatomy of', 'structure of', 'diagram of'")
    elif misrepresented:
        feedback_parts.append("PROBLEM: Retrieved evidence was insufficient or misinterpreted")
        feedback_parts.append("\n ACTION: Add complementary queries for context")
        feedback_parts.append("STRATEGY:")
        feedback_parts.append("  1. Add broader educational queries (e.g., 'what is X', 'types of Y')")
        feedback_parts.append("  2. Include comparison queries between choices")
        feedback_parts.append("  3. Query for specific attributes mentioned in question")
    elif fabricated:
        feedback_parts.append("PROBLEM: Solver made unsupported factual claims")
        feedback_parts.append("\nFabricated information:")
        for h in fabricated[:3]:
            feedback_parts.append(f'  • "{h["claim"][:80]}..."')

        feedback_parts.append("\n ACTION: Retrieve authoritative sources")
        feedback_parts.append("STRATEGY:")
        feedback_parts.append("  1. Query for specific facts/numbers/dates mentioned in question")
        feedback_parts.append("  2. Add 'scientific study', 'research', 'experiment' to queries")
        feedback_parts.append("  3. Query each choice individually with 'what is', 'define'")
    else:
        # Generic fallback
        feedback_parts.append("PROBLEM: Evidence quality insufficient")
        feedback_parts.append("\n ACTION: Generate more diverse, specific queries")
        feedback_parts.append("STRATEGY:")
        feedback_parts.append("  1. Break question into atomic concepts")
        feedback_parts.append("  2. Query each choice separately")
        feedback_parts.append("  3. Add domain-specific terms (biology, physics, etc.)")

    # Show what didn't work
    if previous_queries:
        feedback_parts.append("\n AVOID similar queries to these (didn't retrieve what we need):")
        for q in previous_queries[-3:]:  # Last 3 queries
            feedback_parts.append(f"  • {q}")

    # Add specific query suggestions based on question analysis
    key_terms = extract_key_terms(question, choices)
    if key_terms:
        feedback_parts.append(f"\n KEY TERMS to include: {', '.join(key_terms[:5])}")
    return "\n".join(feedback_parts)


def log_attempt(state: State):
    """
    Log current attempt for tracking progress across retries
    """
    attempt = {
        "retry_count": state.retry_count,
        "subqueries": state.subqueries.copy() if state.subqueries else [],
        "verdict": state.verdict,
        "confidence": state.confidence,
        "hallucination": state.hallucination,
        "num_hallucinations": len(state.hallucination_details),
        "final_answer": state.final_answer,
    }
    state.attempt_history.append(attempt)


def parse_hallucination_details(full_output: str) -> List[Dict[str, str]]:
    """
    Extract structured hallucination information from verifier output
    Returns list of dicts with keys: 'claim', 'issue', 'evidence'
    """
    hallucinations = []

    # Find all claim blocks
    claim_pattern = r'Claim:\s*"([^"]+)"\s*\nIssue:\s*([^\n]+)\s*\nEvidence:\s*([^\n]+)'
    matches = re.finditer(claim_pattern, full_output, re.MULTILINE)

    for match in matches:
        hallucinations.append(
            {
                "claim": match.group(1).strip(),
                "issue": match.group(2).strip(),
                "evidence": match.group(3).strip(),
            }
        )

    return hallucinations


def prepare_images_for_verifier(state):
    """
    Prepare question image sources and descriptions for Qwen2.5-VL.
    ROI retrieval has been removed; only question images (from state.image_paths)
    are passed to the verifier, matching the solver's [Question Image N] citation format.

    Returns (image_sources, descriptions) where image_sources are strings:
    file:// paths so process_vision_info can load them.
    """
    image_sources = []
    descriptions = []

    MAX_IMAGES = 4  # keep low for consistent processor token counts
    # Question images: use file:// path (Qwen2.5-VL supports local files)

    # Question images (from paths)
    for idx, img_path in enumerate(state.image_paths or []):
        if len(image_sources) >= MAX_IMAGES:
            remaining = len(state.image_paths) - idx
            print(
                f"Warning: Verifier reached MAX_IMAGES={MAX_IMAGES}; "
                f"{remaining} question image(s) not passed to verifier. "
                f"[Question Image N] citations for N>{MAX_IMAGES} will be "
                f"flagged as out-of-range — this is expected, not a hallucination."
            )
            break

        if img_path and os.path.exists(img_path):
            try:
                image_sources.append("file://" + os.path.abspath(img_path))
                descriptions.append(f"[Question Image {idx + 1}]")
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
    print(f"Verifier: Prepared {len(image_sources)} question images (max allowed: {MAX_IMAGES})")
    return image_sources, descriptions


def verifier_step(state: State, model, processor) -> State:
    """VLM verifier that examines actual images"""
    with tracer.start_as_current_span("Verifier", openinference_span_kind="chain") as verifier_span:
        # Prepare image sources and descriptions (file:// paths or data:image/png;base64,...)
        image_sources, image_descriptions = prepare_images_for_verifier(state)

        # Format text evidence — MUST mirror solver.format_evidence exactly so that
        # [Text Evidence N] labels in the solver's reasoning correspond to the same
        # chunk displayed here. solver.format_evidence:
        #   1. skips query keys starting with "_"
        #   2. filters chunks with len(stripped) <= 10
        #   3. caps each query at 5 chunks
        #   4. caps total at 15 entries
        # Applying different rules here causes the verifier to see a different chunk
        # under the same label → it flags correct citations as fake/misrepresented.
        # Build the evidence list via the canonical helper in citation injector so
        # [Text Evidence N] labels seen by the verifier are identical to what the solver
        # was shown and what citation_injector used for injection.
        all_text_chunks = _build_text_chunk_list(state.retrieved_chunks, total_cap=15, truncate=400)

        text_evidence = (
            "\n".join([f"[Text Evidence {i + 1}]: {chunk}" for i, chunk in enumerate(all_text_chunks)])
            if all_text_chunks
            else "No text evidence retrieved."
        )

        # Format visual evidence descriptions (not with image tokens)
        # Include the count of images actually shown so the verifier does not
        # flag [Question Image N] citations for images that were dropped due to
        # MAX_IMAGES capping as "fake citations".
        num_images_shown = len(image_sources)
        num_images_total = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])
        images_note = (
            f"NOTE: Only {num_images_shown} of {num_images_total} question images are shown above "
            f"due to display limits. [Question Image N] citations for N > {num_images_shown} "
            f"reference valid images that were not displayed — do NOT flag these as fake citations.\n"
            if num_images_total > num_images_shown
            else ""
        )
        visual_evidence = images_note + (
            "\n".join(image_descriptions) if image_descriptions else "No visual evidence retrieved."
        )

        # Create prompt (WITHOUT manual image tokens - let processor handle it)
        # solver_reasoning must be a plain string — state.reasoning_steps is a List[str].
        # Passing the list directly causes Python's str.format() to call str() on it,
        # producing "['<SUMMARY>...full output...']" (with brackets + quotes) in the prompt.
        # choices must also be a readable labeled string, not a raw Python list repr.
        solver_reasoning_str = "\n".join(state.reasoning_steps or [])
        solver_reasoning_str = solver_reasoning_str[:1500]  # cap at ~375 tokens

        choices_str = "\n".join(f"{chr(ord('A') + i)}: {c}" for i, c in enumerate(state.choices or []))
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            question=state.question,
            choices=choices_str,
            solver_reasoning=solver_reasoning_str,
            text_evidence=text_evidence,
            visual_evidence=visual_evidence,
            solver_answer=state.final_answer,
        )

        # Log input attributes
        verifier_span.set_attribute("verifier.question", state.question)
        verifier_span.set_attribute("verifier.choices", json.dumps(state.choices))
        verifier_span.set_attribute("verifier.retry_attempt", state.retry_count)
        verifier_span.set_attribute("verifier.num_images", len(image_sources))
        model_name = getattr(getattr(model, "config", None), "_name_or_path", model.__class__.__name__)

        with tracer.start_as_current_span("Qwen2.5-VL-Verifier", openinference_span_kind="llm") as vlm_span:
            vlm_span.set_attribute("vlm.model_name", model_name)

            # Official Qwen2.5-VL flow (https://huggingface.co/Qwen/Qwen2.5-VL-8B-Instruct):
            # 1) Messages with image sources in content; 2) apply_chat_template;
            # 3) process_vision_info(messages);
            # 4) processor(text=..., images=image_inputs, videos=video_inputs)
            if image_sources:
                content = [
                    *[{"type": "image", "image": src} for src in image_sources],
                    {"type": "text", "text": prompt},
                ]
                messages = [{"role": "user", "content": content}]
            else:
                # Text-only conversation
                messages = [{"role": "user", "content": prompt}]

            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if process_vision_info is None:
                raise ImportError(
                    "qwen_vl_utils is required for Qwen2.5-VL verifier. Install with: pip install qwen-vl-utils"
                )
            image_inputs, video_inputs = process_vision_info(messages)

            print(f"[Verifier] Prompt length (chars): {len(prompt)}")
            print(f"[Verifier] Solver reasoning length (chars): {len(solver_reasoning_str)}")
            print(f"[Verifier] Text evidence length (chars): {len(text_evidence)}")

            # Process inputs - Qwen2.5-VL specific format
            inputs = processor(
                text=[text],  # text as list for Qwen2.5-VL
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                # max_length=8192,
                return_tensors="pt",
            ).to(_first_parameter_device(model))

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,  # 5-step CoT + structured output
                    temperature=0.0,  # Deterministic for consistency
                    do_sample=False,
                )
                # outputs.shape = [batch_size, sequence_length]
                # e.g., tensor([[1, 2, 3, 4, 5, ...]])  # 2D tensor

            # Decode the output properly
            # outputs is a tensor of shape [batch_size, sequence_length]
            # Use batch_decode for proper decoding
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            full_output = processor.decode(new_tokens, skip_special_tokens=True).strip()
            decoded = full_output  # keep for vlm_span logging

            print(f"[Verifier] Generated output length: {len(full_output)} characters")

            vlm_span.set_attribute("llm.output", full_output)
            vlm_span.set_attribute("llm.full_output", decoded)

        # Initialize with defaults
        state.verdict = "UNKNOWN"
        state.confidence = "LOW"
        state.verifier_answer = "INCONCLUSIVE"
        state.hallucination = "UNKNOWN"
        solver_answer = _extract_choice_letter(state.final_answer or "")

        # Parse the response and update state.
        # Use IGNORECASE throughout and handle markdown bold (**VERIFIED**) that
        # Qwen2.5-VL sometimes emits.
        final_verdict = re.search(
            r"Final\s+Verdict:\s*\*{0,2}\[?(VERIFIED|REJECTED)\]?\*{0,2}", full_output, re.IGNORECASE
        )
        if final_verdict:
            state.verdict = final_verdict.group(1).upper()

        # The 5-step CoT model often outputs "Hallucination Check:" and
        # "Verified Answer:" but silently skips the "Final Verdict:" line.
        # Derive verdict from hallucination classification so downstream logic works.
        if state.verdict == "UNKNOWN":
            halluc_hint = re.search(
                r"Hallucination Check:\s*\[?(NONE DETECTED|MINOR HALLUCINATIONS|MAJOR HALLUCINATIONS)\]?",
                full_output,
                re.IGNORECASE,
            )
            if halluc_hint:
                hval = halluc_hint.group(1).upper()
                state.verdict = "VERIFIED" if hval == "NONE DETECTED" else "REJECTED"
                print(
                    f"  [VERDICT DERIVED] Final Verdict line missing — "
                    f"derived {state.verdict} from Hallucination Check: {hval}"
                )

        confidence_match = re.search(r"Confidence:\s*\*{0,2}\[?(HIGH|MEDIUM|LOW)\]?\*{0,2}", full_output, re.IGNORECASE)
        if confidence_match:
            state.confidence = confidence_match.group(1).upper()

        # Prefer the judge's independent answer; fall back to Verified Answer only
        # for older prompt outputs. Never recover an uncertain verifier answer from
        # the solver, because that caused false accepts in the 425-row shard.
        independent_answer = _extract_labeled_choice(full_output, "Independent Answer")
        verified_answer = _extract_labeled_choice(full_output, "Verified Answer")
        state.verifier_answer = independent_answer if independent_answer != "INCONCLUSIVE" else verified_answer

        # Multiple-choice answer adjudication: if the verifier names a different
        # answer than the solver, force a rejection even if the free-text verdict
        # says VERIFIED. This guards against permissive judge language.
        if (
            state.verifier_answer != "INCONCLUSIVE"
            and solver_answer != "INCONCLUSIVE"
            and state.verifier_answer != solver_answer
        ):
            if state.verdict == "VERIFIED":
                print(f"  [ANSWER MISMATCH] verifier={state.verifier_answer} solver={solver_answer}; forcing REJECTED")
            state.verdict = "REJECTED"
            state.confidence = "HIGH" if state.confidence == "HIGH" else state.confidence

        # If the verifier claims VERIFIED but did not provide an answer letter,
        # treat it as an untrusted low-confidence rejection for retry/analysis.
        if state.verdict == "VERIFIED" and state.verifier_answer == "INCONCLUSIVE":
            state.verdict = "REJECTED"
            state.confidence = "LOW"
            print("  [INCONCLUSIVE VERIFIED] verifier accepted without an independent answer; forcing REJECTED/LOW")

        halluc_match = re.search(
            r"Hallucination Check:\s*\[?(NONE DETECTED|MINOR HALLUCINATIONS|MAJOR HALLUCINATIONS)\]?",
            full_output,
        )
        if halluc_match:
            state.hallucination = halluc_match.group(1)

        # Parse detailed hallucination information
        state.hallucination_details = parse_hallucination_details(full_output)

        # Generate targeted feedback using hallucination details
        if state.verdict != "VERIFIED":
            state.verifier_feedback = generate_targeted_feedback(
                hallucination_type=state.hallucination,
                hallucination_details=state.hallucination_details,
                question=state.question,
                previous_queries=state.subqueries,
                choices=state.choices,
            )
        else:
            state.verifier_feedback = None

        # Log this attempt BEFORE incrementing retry_count so that
        # attempt_history[i].retry_count == i (0-based attempt index).
        log_attempt(state)

        # Increment retry_count so should_retry_planning() can enforce the
        # max-retries limit.  After this point retry_count == number of
        # verifier runs completed.
        state.retry_count += 1

        # Track metrics
        verifier_span.set_attribute("verifier.verdict", state.verdict)
        verifier_span.set_attribute("verifier.hallucination_count", len(state.hallucination_details))
        verifier_span.set_attribute("verifier.retry_count", state.retry_count)
        verifier_span.set_attribute("verifier.confidence", state.confidence)

    # If the verifier output is unparseable, do not auto-accept. Mark it as a
    # low-confidence rejection so the retry loop can attempt recovery and the
    # metrics surface parser/model failures instead of hiding them as VERIFIED.
    if state.verdict == "UNKNOWN":
        state.verdict = "REJECTED"
        state.confidence = "LOW"
        state.verifier_answer = "INCONCLUSIVE"
        print("  [UNKNOWN FALLBACK] Verifier output unparseable; forcing REJECTED/LOW")

    return state


def should_retry_planning(state: State, max_retries: int = 3) -> bool:
    """
    Determine if we should retry planning based on verification results
    """
    # Never retry if verified
    if state.verdict == "VERIFIED":
        return False

    # Stop at max retries
    if state.retry_count >= max_retries:
        return False

    # Retry if major hallucinations (likely retrieval issue)
    if state.hallucination == "MAJOR HALLUCINATIONS":
        return True

    # Retry if minor hallucinations and medium-or-lower confidence.
    # MEDIUM confidence on a MINOR hallucination
    # finding is uncertain enough to warrant one more attempt — the verifier
    # may have misread the image or been misled by injected text citations.
    if state.hallucination == "MINOR HALLUCINATIONS":
        # and state.confidence in [
        #     "LOW",
        #     "MEDIUM",
        # ]:
        return True

    return False


def should_plan(state: State, max_retries: int = 3) -> str:
    """
    Improved routing logic using the helper function
    """
    # Use the helper function for cleaner logic
    if should_retry_planning(state, max_retries=max_retries):
        # retry_count has already been incremented by verifier_step, so
        # "retry_count" == number of verifier runs completed so far.
        print(f"Retry {state.retry_count}/{max_retries}: replanning after {state.hallucination}")
        if state.subqueries:
            print(f"   Previous queries: {state.subqueries[:2]}...")
        return "plan"

    if state.retry_count >= max_retries:
        print(f"Max retries ({max_retries}) reached - stopping")
        print(f"   Final verdict: {state.verdict}")
        print(f"   Attempt history: {len(state.attempt_history)} attempts")

    return END


def _should_skip_verifier(state: State) -> bool:
    """
    Return True when the verifier can be safely skipped.

    Two distinct skip conditions:

    Condition A — RETRIEVAL FAILURE:
    When the retriever found no usable evidence (no text chunks across all
    sub-queries), calling the verifier is pointless: the solver had nothing
    to cite, so the verifier will flag every reasoning step as a fake citation
    and REJECT with HIGH confidence — not because the answer is wrong, but
    because retrieval failed. This is not a hallucination; it is a retrieval
    gap. Firing a retry from this state wastes the retry budget without
    improving evidence coverage (the Extractor already ran, the KB simply
    doesn't contain the answer).
    Diagnosis: In the 21k ScienceQA run with Qwen2.5-8B extractor, 92.5% of
    REJECTED verdicts had a correct final answer, and 99.3% were flagged
    HIGH confidence — a clear sign the verifier was penalising retrieval
    failure rather than detecting genuine hallucinations.

    Condition B — TEXT-ONLY + WELL-CITED (original logic):
    No question images + solver produced ≥2 text citations + answer letter
    extracted successfully → verifier adds no information.
    Skip rate at n=100: ~40%. Expected time saving: ~12s × 0.40.
    Image questions always proceed to the full verifier because visual
    hallucinations cannot be detected from citation counts alone.
    """
    # Condition A: retrieval completely failed
    # Count total text chunks across all sub-queries in retrieved_chunks.
    # Keys starting with "_" are metadata, not query results; skip them.
    # Keys starting with "dk_" are DuckDuckGo web snippets — count them too.
    total_chunks = 0
    for key, val in (state.retrieved_chunks or {}).items():
        if key.startswith("_"):
            continue
        chunks = val.get("text_chunks", []) if isinstance(val, dict) else []
        total_chunks += len([c for c in chunks if isinstance(c, str) and len(c.strip()) > 10])

    if total_chunks == 0:
        # No evidence retrieved at all — verifier will spuriously REJECT.
        # Auto-verify instead so the solver's parametric answer is preserved.
        print(
            "  [VerifierSkip] Zero text chunks retrieved — "
            "skipping verifier to avoid spurious retrieval-failure rejection."
        )
        return True

    # Condition B: text-only + well-cited
    # Never skip for image questions
    has_images = any(p for p in (state.image_paths or []) if p and os.path.exists(p))
    if has_images:
        return False

    # Require a successfully extracted answer letter
    fa_match = re.search(r"\b([A-E])\b", state.final_answer or "")
    if not fa_match:
        return False

    # Require ≥2 text citations (the injector ran and found grounding evidence)
    reasoning = " ".join(state.reasoning_steps or [])
    text_cites = len(re.findall(r"\[Text Evidence \d+\]", reasoning))
    if text_cites < 2:
        return False

    return True


def _auto_verify_step(state: State) -> State:
    """
    Lightweight verifier substitute used when _should_skip_verifier() fires.
    Sets VERIFIED/HIGH/NONE DETECTED directly from the solver output so the
    rest of the graph (should_plan, logging) sees a fully populated state.

    Called for two reasons (see _should_skip_verifier):
      A) Zero chunks retrieved — verifier would spuriously reject due to
         missing citations that were never available.
      B) Text-only question with ≥2 citations — verifier adds no signal.
    """
    fa_match = re.search(r"\b([A-E])\b", state.final_answer or "")
    letter = fa_match.group(1) if fa_match else "INCONCLUSIVE"
    state.verdict = "VERIFIED"
    state.confidence = "HIGH"
    state.verifier_answer = letter
    state.hallucination = "NONE DETECTED"
    state.hallucination_details = []
    state.verifier_feedback = None
    log_attempt(state)
    state.retry_count += 1

    # Determine skip reason for logging
    total_chunks = sum(
        len(
            [
                c
                for c in (v.get("text_chunks", []) if isinstance(v, dict) else [])
                if isinstance(c, str) and len(c.strip()) > 10
            ]
        )
        for k, v in (state.retrieved_chunks or {}).items()
        if not k.startswith("_")
    )
    if total_chunks == 0:
        print(f"  [VerifierSkip/RetrievalFailure] No chunks → auto-VERIFIED ({letter})")
    else:
        reasoning_text = " ".join(state.reasoning_steps or [])
        cite_count = len(re.findall(r"\[Text Evidence \d+\]", reasoning_text))
        print(f"  [VerifierSkip/WellCited] Text-only + {cite_count} citations → auto-VERIFIED ({letter})")
    return state


# def _conditional_verifier_step(verifier_model, verifier_processor) -> callable:
#     """
#     Returns a node function that either runs the full VLM verifier or the
#     lightweight auto-verify stub, depending on _should_skip_verifier().
#     Wraps both paths in an OpenTelemetry span for consistent tracing.
#     """
#     _full_verifier = partial(verifier_step, model=verifier_model, processor=verifier_processor)
#     def _node(state: State) -> State:
#         with tracer.start_as_current_span(
#             "Verifier", openinference_span_kind="chain"
#         ) as span:
#             if _should_skip_verifier(state):
#                 span.set_attribute("verifier.skipped", True)
#                 return _auto_verify_step(state)
#             span.set_attribute("verifier.skipped", False)
#             # verifier_step opens its own child span internally; the outer
#             # span here just provides a consistent entry point for the graph.
#             return _full_verifier(state)
#     return _node


def _conditional_verifier_step(verifier_model, verifier_processor) -> callable:
    _full_verifier = partial(verifier_step, model=verifier_model, processor=verifier_processor)

    def _node(state: State) -> State:
        with tracer.start_as_current_span("Verifier", openinference_span_kind="chain") as span:
            span.set_attribute("verifier.skipped", False)
            return _full_verifier(state)

    return _node


def build_cave_vlm_cot_graph(
    planner_model,
    planner_tokenizer,
    planner_kwargs,
    text_index,
    data,
    solver_model,
    solver_processor,
    solver_kwargs,
    verifier_model,
    verifier_processor,
    retrieval_k=5,
):
    """
    Build the complete CaVe-VLM-CoT graph with all dependencies.
    Returns a compiled graph that can be invoked with just a State object.
    Question images are passed directly to the
    solver via state.image_paths and cited as [Question Image N].
    """
    # Create partial functions with models/parameters bound
    planner_node = partial(
        planner_step,
        model=planner_model,
        tokenizer=planner_tokenizer,
        kwargs=planner_kwargs,
    )

    retriever_node = partial(
        retriever_step,
        text_index=text_index,
        data=data,
        k=retrieval_k,
    )

    solver_node = partial(
        solver_step_with_citation_retry,
        model=solver_model,
        processor=solver_processor,
        kwargs=solver_kwargs,
    )

    # verifier_node = partial(
    #     verifier_step, model=verifier_model, processor=verifier_processor
    # )

    verifier_node = _conditional_verifier_step(verifier_model, verifier_processor)

    # Build the graph
    graph = StateGraph(State)
    graph.add_node("plan", planner_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("solve", solver_node)
    graph.add_node("inject_citations", inject_citations_step)
    graph.add_node("verify", verifier_node)

    graph.set_entry_point("plan")

    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "solve")
    graph.add_edge("solve", "inject_citations")
    graph.add_edge("inject_citations", "verify")

    graph.add_conditional_edges("verify", should_plan, {"plan": "plan", END: END})

    return graph.compile()


# Ablation graph builders
def build_retrieval_solver_graph(
    planner_model,
    planner_tokenizer,
    planner_kwargs,
    text_index,
    data,
    solver_model,
    solver_processor,
    solver_kwargs,
    retrieval_k=5,
):
    """
    Ablation 1 — Retrieval-Solver pipeline.
    Pipeline: Planner → Retriever → Solver → END
    Drops citation injector and verifier to isolate the contribution of
    post-hoc citation injection and verification to the final CaVeScore.
    """
    planner_node = partial(
        planner_step,
        model=planner_model,
        tokenizer=planner_tokenizer,
        kwargs=planner_kwargs,
    )
    retriever_node = partial(
        retriever_step,
        text_index=text_index,
        data=data,
        k=retrieval_k,
    )
    solver_node = partial(
        solver_step_with_citation_retry,
        model=solver_model,
        processor=solver_processor,
        kwargs=solver_kwargs,
    )
    graph = StateGraph(State)

    graph.add_node("plan", planner_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("solve", solver_node)

    graph.set_entry_point("plan")

    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "solve")
    graph.add_edge("solve", END)

    return graph.compile()


def build_solver_only_graph(
    solver_model,
    solver_processor,
    solver_kwargs,
):
    """
    Ablation 2 — Solver-only pipeline.
    Pipeline: Solver → END
    The solver receives no retrieved evidence (state.retrieved_chunks == {})
    and no citations are injected, so the model must answer from question text
    and question images alone.  Useful as a RAG-vs-no-RAG baseline.
    """
    solver_node = partial(
        solver_step_with_citation_retry,
        model=solver_model,
        processor=solver_processor,
        kwargs=solver_kwargs,
    )

    graph = StateGraph(State)
    graph.add_node("solve", solver_node)
    graph.set_entry_point("solve")
    graph.add_edge("solve", END)
    return graph.compile()


def build_pipeline_without_citation_injector(
    planner_model,
    planner_tokenizer,
    planner_kwargs,
    text_index,
    data,
    solver_model,
    solver_processor,
    solver_kwargs,
    verifier_model,
    verifier_processor,
    retrieval_k=5,
):
    """
    Ablation 3 — Full pipeline without citation injector.
    Pipeline: Planner → Retriever → Solver → Verifier → (retry loop)
    Skips the inject_citations step to measure how much post-hoc citation
    injection improves citation precision, recall, and the composite CaVeScore
    compared to the citations the solver produces on its own.
    """
    planner_node = partial(
        planner_step,
        model=planner_model,
        tokenizer=planner_tokenizer,
        kwargs=planner_kwargs,
    )
    retriever_node = partial(
        retriever_step,
        text_index=text_index,
        data=data,
        k=retrieval_k,
    )
    solver_node = partial(
        solver_step_with_citation_retry,
        model=solver_model,
        processor=solver_processor,
        kwargs=solver_kwargs,
    )
    verifier_node = _conditional_verifier_step(verifier_model, verifier_processor)
    graph = StateGraph(State)

    graph.add_node("plan", planner_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("solve", solver_node)
    graph.add_node("verify", verifier_node)

    graph.set_entry_point("plan")

    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "solve")
    # Solver output goes directly to verifier — no citation injection.
    graph.add_edge("solve", "verify")
    graph.add_conditional_edges("verify", should_plan, {"plan": "plan", END: END})
    return graph.compile()
