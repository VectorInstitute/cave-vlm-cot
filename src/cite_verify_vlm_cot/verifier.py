import json
import re

import torch
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoProcessor
from langgraph.graph import END, StateGraph
from tracer import tracer

from typing import List, Dict

import base64
import os
from io import BytesIO
from PIL import Image

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # pip install qwen-vl-utils

# Import State and step functions
from utils import State
from planner import planner_step
from retriever import retriever_step
from solver import solver_step, solver_step_with_citation_retry
from citation_injector import inject_citations_step


# VERIFIER_PROMPT_TEMPLATE = """
# You are an expert reasoning verifier. Your job is to detect hallucinations in the solver's reasoning by examining actual images and text evidence.
# QUESTION
# {question}

# ANSWER CHOICES
# {choices}

# SOLVER'S ANSWER
# {solver_answer}

# SOLVER'S REASONING
# {solver_reasoning}

# RETRIEVED TEXT EVIDENCE
# {text_evidence}

# QUESTION IMAGES (shown above in order)
# {visual_evidence}

# ---
# CONFIDENCE CALIBRATION GUIDELINES:
# - HIGH: Evidence directly and unambiguously supports the answer; citations exist and are correct.
#   Default to HIGH when the answer is clearly supported by retrieved text.
# - MEDIUM: Evidence partially supports the answer, or the answer relies on common knowledge.
#   Use MEDIUM when citations are present but reasoning has minor gaps.
# - LOW: No evidence was retrieved, citations are fabricated, or answer contradicts evidence.
#   Reserve LOW for cases where you find MAJOR HALLUCINATIONS.

# IMPORTANT: If the answer is factually correct and supported, output HIGH confidence.
# Do NOT default to LOW just because the question is difficult.

# YOUR TASK

# Check if the solver's reasoning contains hallucinations (fabricated information) by examining the actual images above.

# What counts as a HALLUCINATION?

# 1. Fake Citations
#    - Referencing [Text Evidence N] that doesn't exist (e.g., [Text Evidence 5] when only 3 exist)
#    - Referencing [Question Image N] that doesn't exist (e.g., [Question Image 3] when only 1 image)

# 2. Misrepresented Evidence
#    - Claiming evidence says something it doesn't
#    - Quoting evidence incorrectly
#    - Distorting the meaning of evidence
#    - Describing visual features not present in the actual images

# 3. Fabricated Facts
#    - Making specific claims not supported by evidence OR common knowledge
#    - Inventing statistics, dates, or technical details

# What is NOT a hallucination?

# - Common knowledge (e.g., "water is wet", "mammals are warm-blooded")
# - Obvious inferences (e.g., "since A>B and B>C, then A>C")
# - Minor paraphrasing of evidence (as long as meaning is preserved)
# - Semantic equivalences and restatements of the question itself.
#   For example: "The question asks which question this experiment can answer"
#   is identical in meaning to "Identify the question that the experiment can
#   best answer" — these are NOT misrepresentations.  Do NOT flag a solver
#   claim as a hallucination merely because it restates or paraphrases the
#   question, the choices, or the task description in different words.
# - Describing the experimental setup, apparatus, or procedure in ways that
#   are consistent with the image, even if not verbatim from the evidence.
# - Comparative and "also" claims: if the question asks "which animal ALSO does X",
#   the solver only needs to show the chosen animal does X. It does NOT need to show
#   that no other animal does X. Do NOT flag "Animal A does X" as a hallucination
#   because "the text doesn't say Animal A is the only one that does X."
# - Multi-image questions: before checking any claim, identify WHICH specific animal,
#   object, or subject the question is asking about. Only evaluate claims about that
#   subject — do not reject the answer because a different object in the image has
#   different properties.
# - Partial evidence support: if the solver's conclusion is consistent with common
#   knowledge AND the retrieved text provides supporting context (even indirect),
#   treat this as VERIFIED. Only REJECT when a claim directly contradicts the
#   retrieved text or cites a non-existent source.

# Before verifying, identify:
# Subject of question: [the specific thing being asked about]
# Claim to verify: [the solver's specific answer claim]

# Then check only whether that specific claim contains hallucinations.

# ---
# OUTPUT FORMAT

# Provide your verification in EXACTLY this format:

# Hallucination Check: [NONE DETECTED] or [MINOR HALLUCINATIONS] or [MAJOR HALLUCINATIONS]

# [If hallucinations detected, list each one:]
# Claim: "[exact quote from solver]"
# Issue: [fake citation / not in evidence / misrepresented / fabricated]
# Evidence: [what you actually see in the image/text, or "doesn't exist"]

# [If no hallucinations:]
# All claims properly supported by evidence or common knowledge

# Final Verdict: [VERIFIED] or [REJECTED]

# Confidence: [HIGH] or [MEDIUM] or [LOW]

# Verified Answer: [A/B/C/D/E or INCONCLUSIVE]

# Now verify the solver's reasoning above using this exact format.
# """

VERIFIER_PROMPT_TEMPLATE = """
You are a strict fact-checker. Your ONLY job: does the evidence actually support the solver's claims?

QUESTION
{question}

ANSWER CHOICES
{choices}

SOLVER'S ANSWER
{solver_answer}

SOLVER'S REASONING
{solver_reasoning}

RETRIEVED TEXT EVIDENCE
{text_evidence}

QUESTION IMAGES (shown above in order)
{visual_evidence}

---
TASK: Verify whether the solver's reasoning is hallucination-free using the following
structured chain-of-thought. Work through each step in order — do not skip steps.

STEP 1 — IDENTIFY THE KEY CLAIM
State the single most important factual claim the solver makes to reach its answer.
Key claim: [one sentence]

STEP 2 — LOCATE SUPPORTING EVIDENCE
For each citation used anywhere in the solver's reasoning, quote what the evidence actually says.
- [Text Evidence N] says: "[exact relevant quote or 'does not exist']"
- [Question Image N] shows: "[what you actually see, or 'not visible']"

STEP 3 — CHECK EACH CITATION
For every citation in the solver's reasoning (not just the key claim):
- Does [Text Evidence N] contain the stated fact? YES / NO / PARTIALLY
- Does [Question Image N] actually show what is described? YES / NO / PARTIALLY
List any mismatch as a candidate hallucination.

STEP 4 — CLASSIFY HALLUCINATIONS
Based on Step 3, classify:
- NONE DETECTED: all citations check out; any uncited claims are common knowledge or logical inference
- MINOR HALLUCINATIONS: citation slightly misrepresents evidence but conclusion is still plausible
- MAJOR HALLUCINATIONS: citation is fabricated, contradicts evidence, or conclusion depends on a false claim

STEP 5 — RENDER VERDICT
Given the hallucination classification, decide:
- VERIFIED: the solver's answer is well-supported and hallucination-free (or only minor)
- REJECTED: the solver's answer depends on a major hallucination

---
OUTPUT FORMAT (use EXACTLY this format after completing the steps above):

Hallucination Check: [NONE DETECTED] or [MINOR HALLUCINATIONS] or [MAJOR HALLUCINATIONS]

[If hallucinations detected, list each one:]
Claim: "[exact quote from solver]"
Issue: [fake citation / not in evidence / misrepresented / fabricated]
Evidence: [what the evidence actually says, or "doesn't exist"]

[If no hallucinations:]
All claims properly supported by evidence or common knowledge.

Final Verdict: [VERIFIED] or [REJECTED]

Confidence: [HIGH] or [MEDIUM] or [LOW]

Verified Answer: [copy the answer letter here — see rule below]

VERIFIED ANSWER RULE (mandatory):
- If Final Verdict is VERIFIED → copy the letter from SOLVER'S ANSWER (e.g. if solver said "The answer is B", write: Verified Answer: B)
- If Final Verdict is REJECTED and you know the correct answer → write that letter (e.g. Verified Answer: C)
- If Final Verdict is REJECTED and you cannot determine the correct answer → write: Verified Answer: INCONCLUSIVE

EXAMPLE — do not copy, just follow the pattern:
  SOLVER'S ANSWER: The answer is B: Container B has higher concentration
  Final Verdict: [VERIFIED]
  Confidence: [HIGH]
  Verified Answer: B       ← copied from solver since VERIFIED
"""

# VERIFIER_PROMPT_TEMPLATE = """
# You are a strict fact-checker. Your ONLY job: does the evidence actually support the solver's claims?

# QUESTION
# {question}

# ANSWER CHOICES
# {choices}

# SOLVER'S ANSWER
# {solver_answer}

# SOLVER'S REASONING
# {solver_reasoning}

# RETRIEVED TEXT EVIDENCE
# {text_evidence}

# QUESTION IMAGES (shown above in order)
# {visual_evidence}

# ---
# TASK: Check each factual claim in the solver's reasoning.

# A claim is SUPPORTED if:
# - A specific [Text Evidence N] contains the stated fact, OR
# - A [Question Image N] visually shows what is described

# A claim is a HALLUCINATION if:
# - It cites [Text Evidence N] but that evidence says something different or unrelated
# - It cites [Text Evidence N] or [Question Image N] that does not exist
# - It describes image details not visible in the actual image
# - It makes specific factual assertions with no evidence

# What is NOT a hallucination:
# - Obvious logical inferences (e.g., "since A>B and B>C, then A>C")
# - Paraphrasing or restating the question/choices in different words

# Before verifying, identify:
# Subject of question: [the specific thing being asked about]
# Claim to verify: [the solver's specific answer claim]

# Then check only whether that specific claim contains hallucinations.

# ---
# OUTPUT FORMAT (use EXACTLY this format):

# Hallucination Check: [NONE DETECTED] or [MINOR HALLUCINATIONS] or [MAJOR HALLUCINATIONS]

# [If hallucinations detected, list each one:]
# Claim: "[exact quote from solver]"
# Issue: [fake citation / not in evidence / misrepresented / fabricated]
# Evidence: [what the evidence actually says, or "doesn't exist"]

# [If no hallucinations:]
# All claims properly supported by evidence or common knowledge.

# Final Verdict: [VERIFIED] or [REJECTED]

# Confidence: [HIGH] or [MEDIUM] or [LOW]

# Verified Answer: [A/B/C/D/E or INCONCLUSIVE]
# """

def extract_topic_from_claim(claim: str) -> str:
    """Extract the topic being discussed from a claim"""
    # Remove citation markers
    clean = re.sub(r'\[.*?\]', '', claim)
    # Get main subject (simple heuristic)
    words = clean.split()
    return ' '.join(words[:10]) + '...' if len(words) > 10 else clean

def extract_key_terms(question: str, choices: List[str]) -> List[str]:
    """Extract important terms from question and choices"""
    key_terms = set()
    # Extract capitalized terms (likely proper nouns or important concepts)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
    key_terms.update(capitalized)
    # Extract numbers and units
    numbers_units = re.findall(r'\d+(?:\.\d+)?\s*(?:km|m|cm|kg|g|°C|°F|mph|%)?', question)
    key_terms.update(numbers_units)
    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', question)
    key_terms.update(quoted)
    # Extract important words from choices (nouns, likely)
    for choice in choices:
        words = [w for w in choice.split() if len(w) > 3 and w.lower() not in 
                {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'what', 'when'}]
        key_terms.update(words[:2])
    return list(key_terms)

def generate_targeted_feedback(
    hallucination_type: str,
    hallucination_details: List[Dict[str, str]],
    question: str,
    previous_queries: List[str],
    choices: List[str]
) -> str:
    """
    Generate specific, actionable feedback based on hallucination analysis
    """
    if hallucination_type == "NONE DETECTED":
        return None

    feedback_parts = []

    # Analyze what went wrong
    fake_citations = [h for h in hallucination_details if 'fake citation' in h.get('issue', '').lower()]
    misrepresented = [h for h in hallucination_details if 'misrepresented' in h.get('issue', '').lower() or 'not in evidence' in h.get('issue', '').lower()]
    fabricated = [h for h in hallucination_details if 'fabricated' in h.get('issue', '').lower()]

    # Header
    feedback_parts.append(" VERIFICATION FAILED\n")

    if fake_citations:
        feedback_parts.append("PROBLEM: Solver cited evidence that doesn't exist")
        feedback_parts.append("\nMissing evidence types:")
        for h in fake_citations[:3]:  # Show top 3
            if 'Text Evidence' in h['claim']:
                feedback_parts.append(f"  • Needed text about: {extract_topic_from_claim(h['claim'])}")
            # elif 'Image ROI' in h['claim']:
            elif 'Question Image' in h['claim']:
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
            feedback_parts.append(f"  • \"{h['claim'][:80]}...\"")

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
        feedback_parts.append(f"\n AVOID similar queries to these (didn't retrieve what we need):")
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
        'retry_count': state.retry_count,
        'subqueries': state.subqueries.copy() if state.subqueries else [],
        'verdict': state.verdict,
        'confidence': state.confidence,
        'hallucination': state.hallucination,
        'num_hallucinations': len(state.hallucination_details),
        'final_answer': state.final_answer

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
    # images = []
    image_sources = []
    descriptions = []

    MAX_IMAGES = 4  # keep low for consistent processor token counts
    # TARGET_SIZE = (448, 448)  # CRITICAL: Resize all images to same size for uniform grid
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
                # images.append(Image.open(img_path).convert('RGB'))
                # img = Image.open(img_path).convert('RGB')
                # img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)  # Resize for uniform grid
                # images.append(img)
                image_sources.append("file://" + os.path.abspath(img_path))
                descriptions.append(f"[Question Image {idx + 1}]")
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
    print(f"Verifier: Prepared {len(image_sources)} question images (max allowed: {MAX_IMAGES})")

    # Retrieved image ROIs: use base64 (already stored in roi.image_patch or cache)
    # all_image_rois = []
    # if state.retrieved_chunks and len(image_sources) < MAX_IMAGES:
    #     for query, chunk_info in state.retrieved_chunks.items():
    #         # chunk_info is a ChunkInfo Pydantic model - use attribute access
    #         all_image_rois.extend(chunk_info.image_rois)
    # for idx, roi in enumerate(all_image_rois):
    #     if len(image_sources) >= MAX_IMAGES:
    #         print(f"Warning: Reached max {MAX_IMAGES} images, skipping remaining ROIs")
    #         break

    #     try:
    #         # roi is a RoiInfo Pydantic model - use attribute access.
    #         # Resolve patch from cache if not embedded directly on the ROI.
    #         patch = roi.image_patch or state.image_patch_cache.get(roi.roi_id, "")
    #         if patch:
    #             image_sources.append("data:image/png;base64," + patch)
    #             caption = roi.caption if roi.caption else 'No caption'
    #             descriptions.append(f"[Image ROI {idx+1}]: {caption}")
    #     except Exception as e:
    #         print(f"Warning: Could not load ROI image {idx}: {e}")
    
    # print(f"Verifier: Prepared {len(image_sources)} images for processing (max allowed: {MAX_IMAGES})")
    return image_sources, descriptions

def verifier_step(state: State, model, processor) -> State:
    """VLM verifier that examines actual images"""
    with tracer.start_as_current_span(
        "Verifier", openinference_span_kind="chain"
    ) as verifier_span:
        # Prepare image sources and descriptions (file:// paths or data:image/png;base64,...)
        image_sources, image_descriptions = prepare_images_for_verifier(state)

        # Format text evidence — MUST mirror solver.format_evidence exactly so that
        # [Text Evidence N] labels in the solver's reasoning correspond to the same
        # chunk displayed here. solver.format_evidence:
        #   1. skips query keys starting with "_"
        #   2. filters chunks with len(stripped) <= 10
        #   3. caps each query at 5 chunks
        #   4. caps total at 10 entries
        # Applying different rules here causes the verifier to see a different chunk
        # under the same label → it flags correct citations as fake/misrepresented.
        # Build the evidence list via the canonical helper so [Text Evidence N]
        # labels seen by the verifier are identical to what the solver was shown
        # and what citation_injector used for injection.
        from citation_injector import _build_text_chunk_list
        all_text_chunks = _build_text_chunk_list(
            state.retrieved_chunks, total_cap=15, truncate=500
        )

        text_evidence = "\n".join([
            f"[Text Evidence {i+1}]: {chunk}"
            for i, chunk in enumerate(all_text_chunks)
        ]) if all_text_chunks else "No text evidence retrieved."

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
            if num_images_total > num_images_shown else ""
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
        choices_str = "\n".join(
            f"{chr(ord('A') + i)}: {c}" for i, c in enumerate(state.choices or [])
        )
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            question=state.question,
            choices=choices_str,
            solver_reasoning=solver_reasoning_str,
            text_evidence=text_evidence,
            visual_evidence=visual_evidence,
            solver_answer=state.final_answer
        )

        # messages = [{"role": "user", "content": prompt}]

        # Log input attributes
        verifier_span.set_attribute("verifier.question", state.question)
        verifier_span.set_attribute("verifier.choices", json.dumps(state.choices))
        verifier_span.set_attribute("verifier.retry_attempt", state.retry_count)
        verifier_span.set_attribute("verifier.num_images", len(image_sources))

        with tracer.start_as_current_span("Qwen2-VL-Verifier", openinference_span_kind="llm") as vlm_span:
            vlm_span.set_attribute("vlm.model_name", "Qwen-2.5-VL-7B")

            # Official Qwen2.5-VL flow (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct):
            # 1) Messages with image sources in content; 2) apply_chat_template; 3) process_vision_info(messages); 4) processor(text=..., images=image_inputs, videos=video_inputs)
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
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            if process_vision_info is None:
                raise ImportError(
                    "qwen_vl_utils is required for Qwen2.5-VL verifier. Install with: pip install qwen-vl-utils"
                )
            image_inputs, video_inputs = process_vision_info(messages)

            # Process inputs - Qwen2-VL specific format
            inputs = processor(
                text=[text],  # text as list for Qwen2-VL
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                max_length=8192,
                return_tensors="pt"
            ).to(model.device)

            # DEBUG: Uncomment to check what's being passed
            print(f"DEBUG Verifier - Num images: {len(image_sources)}")
            print(f"DEBUG Verifier - Input keys: {inputs.keys()}")
            if "pixel_values" in inputs:
                print(f"DEBUG Verifier - Pixel values shape: {inputs['pixel_values'].shape}")

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
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract just the assistant's response (after the prompt).
            # Split on the role boundary marker (\nassistant\n) rather than
            # the bare word "assistant", which also appears in the system
            # prompt ("You are an expert reasoning assistant") and would cause
            # a mis-split that drops most of the verification output.
            parts = re.split(r'\nassistant\n', decoded, flags=re.IGNORECASE)
            if len(parts) > 1:
                full_output = parts[-1].strip()
            elif "assistant" in decoded:
                # Fallback for chat templates that omit the surrounding newlines
                full_output = decoded.split("assistant")[-1].strip()
            else:
                full_output = decoded

            print(f"Generated output length: {len(full_output)} characters")
            print(f"Full output:\n{full_output[:500]}...")  # Print first 500 chars

            vlm_span.set_attribute("llm.output", full_output)
            vlm_span.set_attribute("llm.full_output", decoded)

        # Initialize with defaults
        state.verdict = "UNKNOWN"
        state.confidence = "LOW"
        state.verifier_answer = "INCONCLUSIVE"
        state.hallucination = "UNKNOWN"

        # Parse the response and update state.
        # Use IGNORECASE throughout and handle markdown bold (**VERIFIED**) that
        # Qwen2.5-VL sometimes emits.
        final_verdict = re.search(
            r"Final\s+Verdict:\s*\*{0,2}\[?(VERIFIED|REJECTED)\]?\*{0,2}",
            full_output, re.IGNORECASE
        )
        if final_verdict:
            state.verdict = final_verdict.group(1).upper()

        # The 5-step CoT model often outputs "Hallucination Check:" and
        # "Verified Answer:" but silently skips the "Final Verdict:" line.
        # Derive verdict from hallucination classification so downstream logic works.
        if state.verdict == "UNKNOWN":
            halluc_hint = re.search(
                r"Hallucination Check:\s*\[?(NONE DETECTED|MINOR HALLUCINATIONS|MAJOR HALLUCINATIONS)\]?",
                full_output, re.IGNORECASE
            )
            if halluc_hint:
                hval = halluc_hint.group(1).upper()
                state.verdict = "VERIFIED" if hval == "NONE DETECTED" else "REJECTED"
                print(f"  [VERDICT DERIVED] Final Verdict line missing — "
                      f"derived {state.verdict} from Hallucination Check: {hval}")

        confidence_match = re.search(
            r"Confidence:\s*\*{0,2}\[?(HIGH|MEDIUM|LOW)\]?\*{0,2}",
            full_output, re.IGNORECASE
        )
        if confidence_match:
            state.confidence = confidence_match.group(1).upper()

        # Verified Answer: handle brackets, markdown bold, and IGNORECASE label.
        answer_match = re.search(
            r"Verified\s+Answer:\s*\*{0,2}\[?([A-E]|INCONCLUSIVE)\]?\*{0,2}",
            full_output, re.IGNORECASE
        )
        if answer_match:
            state.verifier_answer = answer_match.group(1).upper()

        # verdict=VERIFIED but verifier still emitted INCONCLUSIVE —
        # extract the solver's letter from state.final_answer directly.
        if state.verdict == "VERIFIED" and state.verifier_answer == "INCONCLUSIVE":
            fa_match = re.search(r'\b([A-E])\b', state.final_answer or "")
            if fa_match:
                print(f"  [VERIFIER FALLBACK] verdict=VERIFIED but Verified Answer=INCONCLUSIVE "
                      f"— recovering letter {fa_match.group(1)} from solver final_answer")
                state.verifier_answer = fa_match.group(1)
        
        # verdict=REJECTED but verifier couldn't name the correct answer
        # (Verified Answer=INCONCLUSIVE). The verifier prompt says to write INCONCLUSIVE
        # when it detects a hallucination but doesn't know what the right answer is.
        # At n=1000, 76/82 of these are false rejections — the solver was right.
        # A rejection without an alternative answer is not strong enough evidence to
        # override the solver. Recover the solver's letter and downgrade to VERIFIED/LOW.
        if state.verdict == "REJECTED" and state.verifier_answer == "INCONCLUSIVE":
            fa_match = re.search(r'\b([A-E])\b', state.final_answer or "")
            if fa_match:
                state.verdict = "VERIFIED"
                state.confidence = "LOW"
                state.verifier_answer = fa_match.group(1)
                print(f"  [INCONCLUSIVE REJECT FALLBACK] verdict=REJECTED but Verified Answer=INCONCLUSIVE "
                      f"— verifier uncertain, recovering letter {fa_match.group(1)} from solver final_answer")

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
        verifier_span.set_attribute(
            "verifier.hallucination_count", len(state.hallucination_details)
        )
        verifier_span.set_attribute("verifier.retry_count", state.retry_count)
        verifier_span.set_attribute("verifier.confidence", state.confidence)

    # If the verifier loop exits with verdict still UNKNOWN
    # (regex failed on Final Verdict AND Hallucination Check lines), recover
    # the solver's letter directly so downstream logic gets a usable state.
    # Confidence is forced LOW to signal parse uncertainty.
    if state.verdict == "UNKNOWN":
        fa_match = re.search(r'\b([A-E])\b', state.final_answer or "")
        if fa_match:
            state.verdict = "VERIFIED"
            state.confidence = "LOW"
            state.verifier_answer = fa_match.group(1)
            print(f"  [UNKNOWN FALLBACK] Verifier output unparseable — "
                  f"recovering letter {state.verifier_answer} from solver final_answer")
        else:
            print(f"  [UNKNOWN FALLBACK] Verifier output unparseable and solver "
                  f"final_answer yielded no letter — leaving INCONCLUSIVE")

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
    # Previously only LOW confidence triggered a retry here, meaning a
    # MINOR + MEDIUM verdict (like the PID 655 false rejection) always
    # finalized without retrying.  MEDIUM confidence on a MINOR hallucination
    # finding is uncertain enough to warrant one more attempt — the verifier
    # may have misread the image or been misled by injected text citations.
    if state.hallucination == "MINOR HALLUCINATIONS" and state.confidence in [
        "LOW",
        "MEDIUM",
    ]:
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

# In build_cave_vlm_cot_graph, make verifier optional
# def should_plan(state: State) -> str:
#     # Skip retry logic entirely
#     return END

def _should_skip_verifier(state: State) -> bool:
    """
    Return True when the verifier can be safely skipped.

    Motivation: 80% of verifier calls in the n=100 SLURM run produced the
    minimal 160-char output (NONE DETECTED / VERIFIED / HIGH), meaning the
    verifier added no information and cost ~12s of VLM inference for nothing.
    These cases share three properties:
      1. No question images (text-only question — no visual hallucination risk).
      2. Solver produced multiple text citations (answer is grounded in evidence).
      3. Answer extraction succeeded (a letter A–E is in final_answer).

    We require ALL three conditions to fire, which keeps the skip rate
    conservative.  Image questions always proceed to the full verifier
    because visual hallucinations (misread diagrams, wrong particle counts,
    incorrect label readings) are the failure mode the verifier is best at
    catching, and they cannot be detected from citation counts alone.
    Skip rate at n=100: ~40% (text-only questions with ≥2 citations).
    Expected time saving: ~12s × 0.40 × 5000 = ~67 GPU-hours.
    """
    # Never skip for image questions
    has_images = any(
        p for p in (state.image_paths or []) if p and os.path.exists(p)
    )
    if has_images:
        return False
    # Require a successfully extracted answer letter
    fa_match = re.search(r'\b([A-E])\b', state.final_answer or "")
    if not fa_match:
        return False
    # Require ≥2 text citations (the injector ran and found grounding evidence)
    reasoning = ' '.join(state.reasoning_steps or [])
    text_cites = len(re.findall(r'\[Text Evidence \d+\]', reasoning))
    if text_cites < 2:
        return False
    return True

def _auto_verify_step(state: State) -> State:
    """
    Lightweight verifier substitute used when _should_skip_verifier() fires.
    Sets VERIFIED/HIGH/NONE DETECTED directly from the solver output so the
    rest of the graph (should_plan, logging) sees a fully populated state.
    """
    fa_match = re.search(r'\b([A-E])\b', state.final_answer or "")
    letter = fa_match.group(1) if fa_match else "INCONCLUSIVE"
    state.verdict          = "VERIFIED"
    state.confidence       = "HIGH"
    state.verifier_answer  = letter
    state.hallucination    = "NONE DETECTED"
    state.hallucination_details = []
    state.verifier_feedback = None
    log_attempt(state)
    state.retry_count += 1
    reasoning_text = ' '.join(state.reasoning_steps or [])
    cite_count = len(re.findall(r'\[Text Evidence \d+\]', reasoning_text))
    print(f"  [VerifierSkip] Text-only + {cite_count} citations → auto-VERIFIED ({letter})")
    return state

def _conditional_verifier_step(verifier_model, verifier_processor) -> callable:
    """
    Returns a node function that either runs the full VLM verifier or the
    lightweight auto-verify stub, depending on _should_skip_verifier().
    Wraps both paths in an OpenTelemetry span for consistent tracing.
    """
    _full_verifier = partial(verifier_step, model=verifier_model, processor=verifier_processor)
    def _node(state: State) -> State:
        with tracer.start_as_current_span(
            "Verifier", openinference_span_kind="chain"
        ) as span:
            if _should_skip_verifier(state):
                span.set_attribute("verifier.skipped", True)
                return _auto_verify_step(state)
            span.set_attribute("verifier.skipped", False)
            # verifier_step opens its own child span internally; the outer
            # span here just provides a consistent entry point for the graph.
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
    # Kept for backward compatibility but no longer used for retrieval:
    image_index=None,
    roi_metadata=None,
):
    """
    Build the complete CaVe-VLM-CoT graph with all dependencies.
    Returns a compiled graph that can be invoked with just a State object.
    ROI retrieval has been removed. Question images are passed directly to the
    solver via state.image_paths and cited as [Question Image N].
    """
    from functools import partial
    from retriever import (
        BM25Retriever,
        text_to_embedding,
        rerank_with_cross_encoder,
    )

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

    verifier_node = partial(
        verifier_step, model=verifier_model, processor=verifier_processor
    )

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
    # graph.add_edge("solve", "verify")
    graph.add_edge("solve", "inject_citations")
    graph.add_edge("inject_citations", "verify")

    graph.add_conditional_edges("verify", should_plan, {"plan": "plan", END: END})

    return graph.compile()