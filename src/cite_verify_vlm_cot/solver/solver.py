"""
solver.py - Visual reasoning solver with:
1. Forced visual observation (OBSERVATIONS section in image prompt)
2. Answer consistency check (cross-encoder verifies reasoning matches conclusion)
3. Citation examples restored (recovers citation quality lost without examples)
4. Citation retry REMOVED (fired on 54.5% of questions, added ~8-10s per sample
   with no quality gain; inject_citations_step in citation_injector.py is now
   the authoritative citation-recovery mechanism)
Key fix: The VLM sometimes reasons correctly but picks the wrong letter.
Example: "Solution B has more particles... The answer is A" (WRONG!)

check_answer_consistency() detects this mismatch via cross-encoder scoring and
corrects the stated letter to match what the reasoning actually supports.

Design evolution of image handling:
  Early versions / LLaVA-CoT (Xkev/Llama-3.2V-11B-cot):
    Passed both question images AND retrieved image ROI patches to the VLM,
    labelled [Image ROI N].
    The entire visual-evidence / ROI pipeline was removed
    in favour of the simpler approach: pass only question images, labelled
    [Question Image N].
    
    Llama-3.2-Vision (MLlama) requires image tokens embedded in the message
    content — passing images=... to the processor alone is not enough; the text
    must contain exactly one <|image|> placeholder per image. This is why
    solver_step builds a multimodal content_parts list and lets
    apply_chat_template insert the tokens, rather than embedding them manually.

  Dual prompt templates introduced: text-only questions use
  SOLVER_PROMPT_TEMPLATE (no OBSERVATIONS section); image questions use
  SOLVER_PROMPT_TEMPLATE_IMAGE (forces explicit visual grounding via OBSERVATIONS).

Pydantic v2 models don't support dictionary-style access:
model['field'] → AttributeError
model.get('field') → AttributeError
'field' in model → Wrong behavior
model.field → Correct!
"""

import gc
import re
from typing import List, Tuple
import numpy as np

import torch
from utils import State
from tracer import tracer

from PIL import Image  # NEEDED for Image.open()
import os  # NEEDED for os.path.exists()

from retriever.retriever import cross_encoder
from citation_injector.citation_injector import _build_text_chunk_list
from prompts import SOLVER_PROMPT_TEMPLATE, SOLVER_PROMPT_TEMPLATE_IMAGE

MAX_QUESTION_IMAGES = 5
CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E']

def prepare_question_images(state: State) -> Tuple[List[Image.Image], List[str]]:
    """
    Prepare ALL question images for the VLM.
    No categorization - just numbered sequentially.
    
    Returns: (images list, descriptions list)
    """
    images = []
    descriptions = []
    
    for idx, img_path in enumerate(state.image_paths or []):
        if len(images) >= MAX_QUESTION_IMAGES:
            print(f"Warning: Reached max {MAX_QUESTION_IMAGES} images, skipping remaining")
            break
            
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                
                # Get caption if available
                img_filename = os.path.basename(img_path)
                caption = (state.img_captions or {}).get(img_filename, "")
                
                # Simple sequential numbering
                if caption:
                    descriptions.append(f"[Question Image {len(images)}]: {caption}")
                else:
                    descriptions.append(f"[Question Image {len(images)}]: Image from the question")
                    
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
    
    return images, descriptions

def format_evidence(state: State, image_descriptions: List[str]) -> Tuple[str, str]:
    """
    Format text and image evidence for the prompt.
    Delegates chunk ordering, filtering, and capping to
    `citation_injector._build_text_chunk_list` — the single source of truth —
    so [Text Evidence N] labels in the solver prompt exactly match what
    build_evidence_index, verifier_step, and get_text_evidence_by_id use.
    Cap raised to 15 (from 10) to match citation_injector.build_evidence_index.
    Truncation kept at 400 chars for solver context-length safety.
    """

    chunks = _build_text_chunk_list(
        state.retrieved_chunks, total_cap=15, truncate=400
    )
    text_evidence = [
        f"[Text Evidence {i+1}]: {chunk}" for i, chunk in enumerate(chunks)
    ]
    text_str  = "\n".join(text_evidence) if text_evidence else "[No text evidence retrieved]"
    image_str = "\n".join(image_descriptions) if image_descriptions else "[No question images available]"
    
    print(f"Evidence: {len(text_evidence)} text, {len(image_descriptions)} images")
    
    return text_str, image_str

def format_labeled_choices(choices: List[str]) -> str:
    return '\n'.join([f'{CHOICE_LABELS[i]}: {c}' for i, c in enumerate(choices)])

def _build_prompt(question_text, answer_choices, image_evidence, text_evidence, has_images):
    """Select the appropriate prompt template based on whether images are present.
    Two templates exist because purely textual and visual questions need different
    grounding strategies:
    - Text-only (SOLVER_PROMPT_TEMPLATE): no OBSERVATIONS section; tighter citation
      rules that push the model toward "According to [Text Evidence N]" or
      "From domain knowledge" for every step.
    - Image (SOLVER_PROMPT_TEMPLATE_IMAGE): adds a mandatory OBSERVATIONS section
      so the VLM explicitly describes visual content before reasoning, reducing the
      "reasoning says B but conclusion says A" flip. Includes a worked example
      (do-not-copy framing) to anchor citation format without inducing copying.
    """
    if has_images:
        return SOLVER_PROMPT_TEMPLATE_IMAGE.format(
            question_text=question_text,
            answer_choices=answer_choices,
            image_evidence=image_evidence,
            text_evidence=text_evidence,
        )
    else:
        return SOLVER_PROMPT_TEMPLATE.format(
            question_text=question_text,
            answer_choices=answer_choices,
            text_evidence=text_evidence,
        )

def check_answer_consistency(
    reasoning: str, 
    stated_answer: str, 
    choices: list,
    encoder=None,
    threshold: float = 0.5,
) -> Tuple[bool, str]:
    """
    Use cross-encoder to verify reasoning supports the stated answer.
    The approach scores all choices against the full reasoning
    block, making it robust to wording variation and able to catch cases where the
    conclusion letter is plausible-sounding but contradicted by the reasoning.

    Args:
        reasoning: The reasoning/observations text
        stated_answer: The conclusion text containing the stated answer
        choices: List of answer choices
        encoder: Pre-loaded CrossEncoder (from retriever.cross_encoder)
        threshold: Minimum score gap required to trigger a correction
        
    Returns:
        (is_consistent, corrected_answer) — corrected_answer is "" if consistent
    """
    if not reasoning or not stated_answer or not choices:
        return True, ""
    
    # Extract stated letter
    letter_match = re.search(r'answer is\s*([a-e])', stated_answer.lower())
    if not letter_match:
        return True, ""
    
    stated_letter = letter_match.group(1).upper()
    stated_idx = ord(stated_letter) - ord('A')
    
    if stated_idx >= len(choices):
        return True, ""
    
    if encoder is None:
        return True, ""

    # Score each choice against the reasoning.
    # Note: earlier versions called _get_cross_encoder() here as a lazy-loading
    # fallback so the function could spin up its own encoder if none was passed.
    # That fallback was removed: the module-level cross_encoder (from retriever.py)
    # is always available at import time, so a missing encoder now means something
    # went wrong upstream — failing gracefully (return True, "") is safer than
    # silently loading a second encoder instance.

    pairs = [
        [reasoning, f"The correct answer is {CHOICE_LABELS[i]}: {choice}"]
        for i, choice in enumerate(choices)
    ]
    
    try:
        scores = encoder.predict(pairs)
    except Exception as e:
        print(f"  [CONSISTENCY CHECK] Error: {e}")
        return True, ""
    
    # Find best supported choice
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    stated_score = float(scores[stated_idx])
    
    # If reasoning clearly supports a different choice
    if best_idx != stated_idx and best_score > stated_score + threshold:
        best_letter = CHOICE_LABELS[best_idx]
        print(f"  [CONSISTENCY CHECK] Reasoning supports {best_letter} "
              f"(score={best_score:.2f}) > {stated_letter} (score={stated_score:.2f})")
        return False, f"The answer is {best_letter}: {choices[best_idx]}"
    
    return True, ""

def parse_solver_output(state, full_output: str, choices: list, encoder=None):
    """
    Parse solver output with cross-encoder consistency checking.
    
    Args:
        state: Pipeline state object
        full_output: Raw VLM output
        choices: List of answer choices
        encoder: Pre-loaded CrossEncoder (from retriever.cross_encoder)
        
    Returns:
        Updated state with final_answer set
    """
    state.reasoning_steps = [full_output]
    
    # STEP 1: Extract answer from CONCLUSION block
    conclusion_match = re.search(
        r'<CONCLUSION>(.*?)</CONCLUSION>', full_output, re.IGNORECASE | re.DOTALL
    )
    search_text = conclusion_match.group(1).strip() if conclusion_match else full_output

    answer_patterns = [
        r'[Tt]he answer is\s*([A-E])[:\s]*([^\n]+)',
        r'[Aa]nswer[:\s]+([A-E])[:\s]*([^\n]*)',
        r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',
        r'(?:choose|select|pick)\s*([A-E])',
    ]
    
    answer_text = ""
    for pattern in answer_patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(choices):
                answer_text = f"The answer is {letter}: {choices[idx]}"
                break
    
    # Fallback: choice text in the tail
    if not answer_text:
        tail = search_text[-300:] if len(search_text) > 300 else search_text
        for i, choice in enumerate(choices):
            if choice.lower() in tail.lower():
                answer_text = f"The answer is {CHOICE_LABELS[i]}: {choice}"
                break
    
    state.final_answer = answer_text if answer_text else full_output[-200:]
    
    # STEP 2: Consistency check using cross-encoder
    # Extract reasoning from REASONING, CAPTION, or OBSERVATIONS tags
    reasoning_parts = []
    
    reasoning_match = re.search(
        r'<REASONING>(.*?)</REASONING>', full_output, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning_parts.append(reasoning_match.group(1).strip())
    
    visual_match = re.search(
        r'<(?:CAPTION|OBSERVATIONS)>(.*?)</(?:CAPTION|OBSERVATIONS)>', 
        full_output, re.IGNORECASE | re.DOTALL
    )
    if visual_match:
        reasoning_parts.append(visual_match.group(1).strip())
    
    combined_reasoning = " ".join(reasoning_parts)
    
    # Only check consistency if we have both reasoning and a conclusion
    if combined_reasoning and conclusion_match and len(combined_reasoning) > 50:
        is_consistent, corrected = check_answer_consistency(
            reasoning=combined_reasoning,
            stated_answer=conclusion_match.group(1),
            choices=choices,
            encoder=cross_encoder,
            threshold=0.5,  # Require 0.5 score difference to correct
        )
        
        if not is_consistent and corrected:
            print(f"  [CORRECTED] {state.final_answer} → {corrected}")
            state.final_answer = corrected
    
    return state

def solver_step(state: State, model, processor, kwargs) -> State:
    """
    Solver with [Question Image N] citations and consistency checking.
    """
    with tracer.start_as_current_span("Solver", openinference_span_kind="chain") as solver_span:
        
        # Prepare question images (simple sequential numbering)
        images, image_descriptions = prepare_question_images(state)
        
        # Format evidence
        text_evidence, image_evidence = format_evidence(state, image_descriptions)
        
        # Format choices
        labeled_choices = format_labeled_choices(state.choices)
        
        # Build prompt
        prompt = _build_prompt(
            question_text=state.question,
            answer_choices=labeled_choices,
            image_evidence=image_evidence,
            text_evidence=text_evidence,
            has_images=bool(images),
        )
        
        # Build messages
        # Llama-3.2-Vision (MLlama) requires image tokens to be embedded in the
        # message content — passing images=... to the processor alone is not
        # sufficient. The text must contain exactly one <|image|> placeholder per
        # image, which apply_chat_template inserts automatically when it sees a
        # {"type": "image"} dict in the content list. Manually embedding the tokens
        # or relying on processor(images=...) without the content dicts produces
        # mismatched cross-attention and garbled output.

        # For VLM, describe retrieved images as text captions in the prompt.
        # Do NOT inject <|image|> tokens manually here — the processor inserts them
        # when images are passed to processor(images=..., text=...)

        if images:
            # Each image becomes a {"type": "image"} dict; text follows at the end.
            content_parts = [{"type": "image"} for _ in images]
            content_parts.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content_parts}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Log input attributes
        solver_span.set_attribute("solver.question", state.question)
        solver_span.set_attribute("solver.num_images", len(images))
        
        print(f"Solver: {len(images)} question images")
        
        with tracer.start_as_current_span("VLM-Inference", openinference_span_kind="llm") as vlm_span:
            
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process inputs (note: first arg is images, None for text-only)
            # Pass actual images to processor
            inputs = processor(
                images=images if images else None,
                text=text,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, **kwargs)
                # outputs.shape = [batch_size, sequence_length]
                # e.g., tensor([[1, 2, 3, 4, 5, ...]])  # 2D tensor

                # Add these lines to free memory
                torch.cuda.empty_cache()
                gc.collect()
            
            # Decode the output properly
            # outputs is a tensor of shape [batch_size, sequence_length]
            # Use batch_decode for proper decoding
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the assistant's response. Split on the role
            # boundary marker (\nassistant\n) to avoid mis-splitting on the
            # word "assistant" inside the system prompt text.
            parts = re.split(r'\nassistant\n', decoded, flags=re.IGNORECASE)
            if len(parts) > 1:
                full_output = parts[-1].strip()
            # Extract just the assistant's response (after the prompt)
            # The output includes the full prompt + response, so we need to extract just the new part
            elif "assistant" in decoded.lower():
                # Split at the last occurrence of "assistant" to get the model's response
                full_output = decoded.split("assistant")[-1].strip()
            else:
                full_output = decoded

            # Check for observations/caption
            has_observations = bool(re.search(r'<(?:OBSERVATIONS|CAPTION)>', full_output, re.IGNORECASE))
            
            # Check citations
            text_cites = re.findall(r'\[Text Evidence \d+\]', full_output)
            image_cites = re.findall(r'\[Question Image \d+\]', full_output)
            
            print(f"Solver output: {len(full_output)} chars")
            print(f"  Has observations: {has_observations}")
            print(f"  Citations: {len(text_cites)} text, {len(image_cites)} image "
                  f"({'cited' if image_cites else 'not cited'})")            
            if images and not image_cites and not has_observations:
                print(f"  [Solver] Image passed but not cited — citation injector will handle.")
            
            vlm_span.set_attribute("llm.output", full_output[:2000])
        
        # Parse output with consistency check (pass cross_encoder)
        state = parse_solver_output(state, full_output, state.choices, encoder=cross_encoder)
        solver_span.set_attribute("solver.final_answer", state.final_answer or "")
        solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps or "")
    
    return state

def solver_step_with_citation_retry(state: State, model, processor, kwargs) -> State:
    """
    Previously retried the solver when citations were missing.

    When active, the retry logic worked as follows:
    - Text retry  : fired when text evidence was retrieved but no [Text Evidence N]
      appeared in the output AND the retrieved text was substantive (>150 chars) AND
      relevant (shared keywords with the question). Skipped if the solver had already
      grounded its answer in visual observations — retrying for missing text citations
      when the answer is purely visual just forces the model to fabricate text support.
    - Image retry : fired when question images were passed to the VLM but no
      [Question Image N] citations or <OBSERVATIONS> block appeared in the output.
    - Priority: text retry was checked first (more common failure mode). Each retry
      was independent — at most ONE retry fired per sample, avoiding a third full
      VLM inference when both conditions were met simultaneously.
    - Retry temperature: 0.1 (lower than default to reduce stochastic non-compliance).

    REMOVED (latency fix): The retry fired on 54.5% of questions (all image
    questions and many text-only ones), adding a full second VLM inference
    (~12–15s) that almost never produced citations either — the citation
    injector then added them post-hoc regardless.  Running the solver twice
    to get the same uncited output and then fixing it with the injector saved
    nothing and cost ~8–10s per question on average.

    The citation injector (inject_citations_step) is the authoritative
    citation recovery mechanism.  This function is kept as a thin passthrough
    so call sites in verifier.build_cave_vlm_cot_graph don't need changes.
    """
    return solver_step(state, model, processor, kwargs)