import json
import re

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from langgraph.graph import END, StateGraph
from tracer import tracer

from typing import List, Dict

import base64
import os
from io import BytesIO
from PIL import Image

# Import State and step functions
from planner import State, planner_step
from retriever import retriever_step
from solver import solver_step

# VERIFIER_PROMPT_TEMPLATE = """Verify if this answer is correct and well-supported.

# QUESTION: {question}
# ANSWER: {solver_answer}
# EVIDENCE: {evidence}

# Check:
# 1. Is answer CORRECT?
# 2. Are citations VALID?
# 3. Any HALLUCINATIONS?

# Verdict: [VERIFIED/REJECTED]
# Confidence: [HIGH/MEDIUM/LOW]
# Verified Answer: [A/B/C/D]"""

VERIFIER_PROMPT_TEMPLATE = """

You are an expert reasoning verifier. Your job is to detect hallucinations in the solver's reasoning by examining actual images and text evidence.
QUESTION
{question}

ANSWER CHOICES
{choices}

SOLVER'S REASONING
{solver_reasoning}

RETRIEVED TEXT EVIDENCE
{text_evidence}

RETRIEVED VISUAL EVIDENCE
{visual_evidence}

---

YOUR TASK

Check if the solver's reasoning contains hallucinations (fabricated information) by examining the actual images above.

What counts as a HALLUCINATION?

1. Fake Citations
   - Referencing evidence that doesn't exist (e.g., [Text Evidence 5] when only 3 exist)
   - Referencing image ROIs that don't exist

2. Misrepresented Evidence
   - Claiming evidence says something it doesn't
   - Quoting evidence incorrectly
   - Distorting the meaning of evidence
   - Describing visual features not present in the actual images

3. Fabricated Facts
   - Making specific claims not supported by evidence OR common knowledge
   - Inventing statistics, dates, or technical details

What is NOT a hallucination?

- Common knowledge (e.g., "water is wet", "mammals are warm-blooded")
- Obvious inferences (e.g., "since A>B and B>C, then A>C")
- Minor paraphrasing of evidence (as long as meaning is preserved)

---
OUTPUT FORMAT

Provide your verification in EXACTLY this format:

Hallucination Check: [NONE DETECTED] or [MINOR HALLUCINATIONS] or [MAJOR HALLUCINATIONS]

[If hallucinations detected, list each one:]
Claim: "[exact quote from solver]"
Issue: [fake citation / not in evidence / misrepresented / fabricated]
Evidence: [what you actually see in the image/text, or "doesn't exist"]

[If no hallucinations:]
All claims properly supported by evidence or common knowledge

Final Verdict: [VERIFIED] or [REJECTED]

Confidence: [HIGH] or [MEDIUM] or [LOW]

Verified Answer: [A/B/C/D or INCONCLUSIVE]

Now verify the solver's reasoning above using this exact format.
"""

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
            elif 'Image ROI' in h['claim']:
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
    images = []
    descriptions = []

    MAX_IMAGES = 6  # CRITICAL: Limit to prevent Qwen2-VL batching issues with too many images
    TARGET_SIZE = (448, 448)  # CRITICAL: Resize all images to same size for uniform grid
    
    # Question images (from paths)
    for idx, img_path in enumerate(state.image_paths or []):
        if len(images) >= MAX_IMAGES:
            print(f"Warning: Reached max {MAX_IMAGES} images, skipping remaining question images")
            break

        if img_path and os.path.exists(img_path):
            try:
                # images.append(Image.open(img_path).convert('RGB'))
                img = Image.open(img_path).convert('RGB')
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)  # Resize for uniform grid
                images.append(img)
                descriptions.append(f"Question Image {idx+1}")
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")

    # Add retrieved image ROIs (from stored patches) - only if we have room
    # Note: state.retrieved_chunks is a dict with queries as keys, ChunkInfo as values
    if state.retrieved_chunks and len(images) < MAX_IMAGES:
        all_image_rois = []
        for query, chunk_info in state.retrieved_chunks.items():
            # chunk_info is a ChunkInfo Pydantic model - use attribute access
            all_image_rois.extend(chunk_info.image_rois)
    for idx, roi in enumerate(all_image_rois):
        if len(images) >= MAX_IMAGES:
            print(f"Warning: Reached max {MAX_IMAGES} images, skipping remaining ROIs")
            break

        try:
            # roi is a RoiInfo Pydantic model - use attribute access
            if roi.image_patch:
                # Decode base64 to PIL image - no disk I/O!
                img_data = base64.b64decode(roi.image_patch)
                img = Image.open(BytesIO(img_data)).convert('RGB')
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)  # Resize for uniform grid
                images.append(img)
                caption = roi.caption if roi.caption else 'No caption'
                descriptions.append(f"[Image ROI {idx+1}]: {caption}")
        except Exception as e:
            print(f"Warning: Could not load ROI image {idx}: {e}")
    
    print(f"Verifier: Prepared {len(images)} images for processing (max allowed: {MAX_IMAGES})")
    return images, descriptions

def verifier_step(state: State, model, processor) -> State:
    """VLM verifier that examines actual images"""
    with tracer.start_as_current_span(
        "Verifier", openinference_span_kind="chain"
    ) as verifier_span:
        # Prepare images and descriptions
        images, image_descriptions = prepare_images_for_verifier(state)

        # Format text evidence
        all_text_chunks = []
        if state.retrieved_chunks:
            for query, chunk_info in state.retrieved_chunks.items():
                # chunk_info is a ChunkInfo Pydantic model - use attribute access
                all_text_chunks.extend(chunk_info.text_chunks)

        text_evidence = "\n".join([
            f"[Text Evidence {i+1}]: {chunk}"
            for i, chunk in enumerate(all_text_chunks)
        ]) if all_text_chunks else "No text evidence retrieved."

        # Format visual evidence descriptions (not with image tokens)
        if images and image_descriptions:
            visual_evidence = "\n".join([
                f"{desc}" for desc in image_descriptions
            ])
        else:
            visual_evidence = "No visual evidence retrieved."

        # Create prompt (WITHOUT manual image tokens - let processor handle it)
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            question=state.question,
            choices=state.choices,
            solver_reasoning=state.reasoning_steps,
            text_evidence=text_evidence,
            visual_evidence=visual_evidence
        )

        # messages = [{"role": "user", "content": prompt}]

        # Log input attributes
        verifier_span.set_attribute("verifier.question", state.question)
        verifier_span.set_attribute("verifier.choices", json.dumps(state.choices))
        verifier_span.set_attribute("verifier.retry_attempt", state.retry_count)
        verifier_span.set_attribute("verifier.num_images", len(images))

        with tracer.start_as_current_span("Qwen2-VL-Verifier", openinference_span_kind="llm") as vlm_span:
            vlm_span.set_attribute("vlm.model_name", "Qwen-2.5-VL-7B")

            # For Qwen2-VL: Create proper conversation structure
            if images:
                # Multi-modal conversation with images
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image"} for _ in images],  # Image placeholders
                            {"type": "text", "text": prompt}       # Text content
                        ]
                    }
                ]
            else:
                # Text-only conversation
                conversation = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            # Apply chat template
            text = processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # Process inputs - Qwen2-VL specific format
            inputs = processor(
                text=[text],  # text as list for Qwen2-VL
                # text = prompt,
                images=images if images else None,
                padding=True,
                max_length=8192,
                return_tensors="pt"
            ).to(model.device)

            # DEBUG: Uncomment to check what's being passed
            print(f"DEBUG Verifier - Num images: {len(images) if images else 0}")
            print(f"DEBUG Verifier - Input keys: {inputs.keys()}")
            if 'pixel_values' in inputs:
                print(f"DEBUG Verifier - Pixel values shape: {inputs['pixel_values'].shape}")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.0,  # Deterministic for consistency
                    do_sample=False,
                )
                # outputs.shape = [batch_size, sequence_length]
                # e.g., tensor([[1, 2, 3, 4, 5, ...]])  # 2D tensor

            # Decode the output properly
            # outputs is a tensor of shape [batch_size, sequence_length]
            # Use batch_decode for proper decoding
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract just the assistant's response (after the prompt)
            # The output includes the full prompt + response, so we need to extract just the new part
            if "assistant" in decoded:
                # Split at the last occurrence of "assistant" to get the model's response
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
        state.gold_answer = "INCONCLUSIVE"
        state.hallucination = "UNKNOWN"

        # Parse the response and update state
        final_verdict = re.search(
            r"Final Verdict:\s*\[?(VERIFIED|REJECTED)\]?", full_output
        )
        if final_verdict:
            state.verdict = final_verdict.group(1)

        confidence_match = re.search(
            r"Confidence:\s*\[?(HIGH|MEDIUM|LOW)\]?", full_output
        )
        if confidence_match:
            state.confidence = confidence_match.group(1)

        answer_match = re.search(
            r"Verified Answer:\s*([A-D]|INCONCLUSIVE)", full_output
        )
        if answer_match:
            state.gold_answer = answer_match.group(1)

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

        # Log this attempt
        log_attempt(state)

        # Track metrics
        verifier_span.set_attribute("verifier.verdict", state.verdict)
        verifier_span.set_attribute(
            "verifier.hallucination_count", len(state.hallucination_details)
        )
        verifier_span.set_attribute("verifier.retry_count", state.retry_count)
        verifier_span.set_attribute("verifier.confidence", state.confidence)

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

    # Retry if minor hallucinations and low confidence
    if state.hallucination == "MINOR HALLUCINATIONS" and state.confidence in [
        "LOW",
        "MEDIUM",
    ]:
        return True

    return False


def should_plan(state: State) -> str:
    """
    Improved routing logic using the helper function
    """
    # Use the helper function for cleaner logic
    if should_retry_planning(state, max_retries=3):
        print(f"Retry {state.retry_count + 1}/3: {state.hallucination}")
        if state.subqueries:
            print(f"   Previous queries: {state.subqueries[:2]}...")
        return "plan"

    if state.retry_count >= 3:
        print("⚠️ Max retries reached - stopping")
        print(f"   Final verdict: {state.verdict}")
        print(f"   Attempt history: {len(state.attempt_history)} attempts")

    return END


def build_cave_vlm_cot_graph(
    planner_model,
    planner_tokenizer,
    planner_kwargs,
    text_index,
    image_index,
    roi_metadata,
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
    """
    from functools import partial

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
        image_index=image_index,
        roi_metadata=roi_metadata,
        data=data,
        k=retrieval_k,
    )

    solver_node = partial(
        solver_step,
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
    graph.add_node("verify", verifier_node)

    graph.set_entry_point("plan")

    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "solve")
    graph.add_edge("solve", "verify")

    graph.add_conditional_edges("verify", should_plan, {"plan": "plan", END: END})

    return graph.compile()

# graph = StateGraph(State)
# graph.add_node("plan", planner_step)
# graph.add_node("retrieve", retriever_step)
# graph.add_node("solve", solver_step)
# graph.add_node("verify", verifier_step)

# graph.set_entry_point("plan")

# graph.add_edge("plan", "retrieve")
# graph.add_edge("retrieve", "solve")
# graph.add_edge("solve", "verify")

# graph.add_conditional_edges(
#     "verify",
#     should_plan,
#     {"plan":"plan", END:END}
# )

# app = graph.compile()
# print(app)
