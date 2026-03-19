"""
solver.py - Visual reasoning solver with:
1. Forced visual observation (OBSERVATIONS section)
2. Answer consistency check (reasoning must match conclusion)
3. Citation examples restored (to recover citation quality)
Key fix: The VLM sometimes reasons correctly but picks the wrong letter.
Example: "Solution B has more particles... The answer is A" (WRONG!)
This version adds explicit answer mapping and consistency checking.

Pydantic v2 models don't support dictionary-style access:
model['field'] → AttributeError
model.get('field') → AttributeError
'field' in model → Wrong behavior
model.field → Correct!
"""
import json
import re
from typing import List, Tuple
import numpy as np

import torch
from utils import State
from tracer import tracer

import base64  # NEEDED for base64.b64decode()
from io import BytesIO  # NEEDED for BytesIO(img_data)
from PIL import Image  # NEEDED for Image.open()
import os  # NEEDED for os.path.exists()

from retriever import cross_encoder

# Prompt for text-only questions (no images)
SOLVER_PROMPT_TEMPLATE = """
You are an expert reasoning assistant. Provide systematic, evidence-based answers.

Question: {question_text}

Choices:
{answer_choices}

Retrieved Text Evidence:
{text_evidence}

CITATION RULES:
- Use [Text Evidence N] for any claim drawn from the retrieved text above.
- If evidence is insufficient, rely on general knowledge without a citation.

Answer using this structured format:

<SUMMARY>
State the core problem in 1-2 sentences.
</SUMMARY>

<REASONING>
Step-by-step reasoning with citations:
- Step 1: According to [Text Evidence 1], ...
- Step 2: ...
</REASONING>

<CONCLUSION>
The answer is [LETTER]: [option text]
</CONCLUSION>
"""

# Question Images (shown above in order):
# {image_evidence}

# SOLVER_PROMPT_TEMPLATE = """
# You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

# Question: {question_text}

# Choices:
# {answer_choices}

# Retrieved Text Evidence:
# {text_evidence}

# CITATION RULES:
# - Use [Text Evidence N] for any claim drawn from the retrieved text above.
# - Use [Question Image N] ONLY when that specific image provides visual evidence
#   that directly supports your claim (e.g. a label, diagram feature, colour, or
#   spatial relationship visible in the image). Do NOT cite an image just because
#   the question has one — only cite it when you are actually using what you see
#   in it to support a specific reasoning step.
# - If neither text nor images provide relevant evidence for a claim, rely on
#   general knowledge and do not add a citation.

# Answer using this structured format:

# <SUMMARY>
# State the core problem in 1-2 sentences.
# </SUMMARY>

# <CAPTION>
# Briefly describe any visual information in the Question Images that is directly
# relevant to answering the question. If the images are not informative for this
# question, state that explicitly.
# </CAPTION>

# <REASONING>
# Step-by-step reasoning with selective citations:
# - Step 1: According to [Text Evidence 1], ...
# - Step 2: Looking at [Question Image 1], I can see that... (only if the image
#   shows something relevant — e.g. a diagram, map, or chart you are reading)
# - Step 3: Combining [Text Evidence 2] and [Question Image 1], ...
# </REASONING>

# <CONCLUSION>
# The answer is [LETTER]: [option text]
# </CONCLUSION>

# CRITICAL RULES:
# 1. Cite [Text Evidence N] for every claim drawn from the retrieved text.
# 2. Cite [Question Image N] only when that image is genuinely useful evidence
#    for the specific claim — not as a formality.
# 3. It is correct to have zero image citations if the images do not help answer
#    the question.
# 4. If evidence is insufficient, state "Based on available evidence, I cannot
#    determine..."
# """

# Prompt for questions WITH images - forces visual observation
# SOLVER_PROMPT_TEMPLATE_IMAGE = """
# You are an expert visual reasoning assistant. Examine the images above carefully.

# Question: {question_text}

# Choices:
# {answer_choices}

# Question Images (shown above):
# {image_evidence}

# Retrieved Text Evidence:
# {text_evidence}

# ---
# INSTRUCTIONS - Complete these steps IN ORDER:

# STEP 1: OBSERVE THE IMAGES (Required - do not skip)
# Look at each image carefully and describe what you actually see.
# Be specific: labels, numbers, colors, arrows, text, patterns, structures.

# STEP 2: REASON WITH EVIDENCE  
# Connect your visual observations to the text evidence.
# Explain how they help answer the question.

# STEP 3: CONCLUDE
# State your final answer. Make sure it matches your reasoning!

# ---
# FORMAT:

# <OBSERVATIONS>
# Image 1: [What type of image? What specific details do you see?]
# Image 2: [If present - what do you see?]
# </OBSERVATIONS>

# <REASONING>
# Based on my observations:
# - [What I see in the image tells me...]
# - According to the text evidence, [relevant fact]...
# - Therefore...
# </REASONING>

# <CONCLUSION>
# The answer is [LETTER]: [full answer text]
# </CONCLUSION>

# ---
# CRITICAL REMINDERS:
# - You MUST fill in the <OBSERVATIONS> section with specific visual details
# - Your final answer MUST match your reasoning (don't contradict yourself)
# - Include BOTH the letter AND the full answer text in your conclusion
# """

SOLVER_PROMPT_TEMPLATE_IMAGE = """
You are an expert visual reasoning assistant. Examine the images above carefully.

Question: {question_text}

Choices:
{answer_choices}

Question Images (shown above in order):
{image_evidence}

Retrieved Text Evidence:
{text_evidence}

---
CITATION RULES:
- Use [Text Evidence N] for claims from retrieved text
- Use [Question Image N] when describing what you see in an image

---
EXAMPLE (do NOT copy — answer YOUR question):

<OBSERVATIONS>
[Question Image 1] shows a diagram with two containers. Container A has 3 particles, Container B has 5 particles. Both have 40mL volume labeled.
</OBSERVATIONS>

<REASONING>
- Looking at [Question Image 1], I can count the particles: Container A has 3, Container B has 5.
- According to [Text Evidence 1], concentration = particles / volume.
- Since both containers have equal volume (40mL), Container B has higher concentration.
</REASONING>

<CONCLUSION>
The answer is B: Container B
</CONCLUSION>

---
NOW ANSWER YOUR QUESTION:

<OBSERVATIONS>
For each image, describe what you see. Use [Question Image N] citations.
</OBSERVATIONS>

<REASONING>
Step-by-step reasoning using your observations and text evidence.
- Use [Question Image N] when referencing visual details
- Use [Text Evidence N] when referencing retrieved text
</REASONING>

<CONCLUSION>
The answer is [LETTER]: [full answer text]
</CONCLUSION>

---
CRITICAL:
- You MUST cite [Question Image N] when describing visual observations
- You MUST cite [Text Evidence N] when using retrieved facts
- Your CONCLUSION must match your REASONING
- MANDATORY: Every observation MUST include [Question Image N] citation.
- MANDATORY: Every fact from text MUST include [Text Evidence N] citation.
"""

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
    """
    text_evidence = []
    text_idx = 1
    
    if state.retrieved_chunks:
        for query, chunk_info in state.retrieved_chunks.items():
            # Skip special keys
            if query.startswith("_"):
                continue
                
            for chunk in chunk_info.text_chunks[:5]:
                if chunk and len(chunk.strip()) > 10:
                    chunk_text = chunk.strip()[:400]
                    text_evidence.append(f"[Text Evidence {text_idx}]: {chunk_text}")
                    text_idx += 1
    
    # Limit to avoid context overflow
    text_evidence = text_evidence[:10]
    
    text_str = "\n".join(text_evidence) if text_evidence else "[No text evidence retrieved]"
    image_str = "\n".join(image_descriptions) if image_descriptions else "[No question images available]"
    
    print(f"Evidence: {len(text_evidence)} text, {len(image_descriptions)} images")
    
    return text_str, image_str

def format_labeled_choices(choices: List[str]) -> str:
    return '\n'.join([f'{CHOICE_LABELS[i]}: {c}' for i, c in enumerate(choices)])

def _build_prompt(question_text, answer_choices, image_evidence, text_evidence, has_images):
    """Select the appropriate prompt template based on whether images are present."""
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
    
    Args:
        reasoning: The reasoning/observations text
        stated_answer: The conclusion text containing the stated answer
        choices: List of answer choices
        cross_encoder: Optional pre-loaded CrossEncoder
        threshold: Minimum score difference to trigger correction
        
    Returns:
        (is_consistent, corrected_answer) - corrected_answer is empty if consistent
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
    
    # Get cross-encoder
    # encoder = cross_encoder or _get_cross_encoder()
    
    # Score each choice against the reasoning
    # Query: "Based on this reasoning, which answer is correct?"
    pairs = [
        [reasoning, f"The correct answer is {CHOICE_LABELS[i]}: {choice}"]
        for i, choice in enumerate(choices)
    ]
    
    if encoder is None:
        return True, ""
    
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
        cross_encoder: Optional pre-loaded CrossEncoder for consistency check
        
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
        if images:
            content_parts = [{"type": "image"} for _ in images]
            content_parts.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content_parts}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Log
        solver_span.set_attribute("solver.question", state.question)
        solver_span.set_attribute("solver.num_images", len(images))
        
        print(f"Solver: {len(images)} question images")
        
        with tracer.start_as_current_span("VLM-Inference", openinference_span_kind="llm") as vlm_span:
            
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = processor(
                images=images if images else None,
                text=text,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, **kwargs)
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the assistant's response. Split on the role
            # boundary marker (\nassistant\n) to avoid mis-splitting on the
            # word "assistant" inside the system prompt text.
            parts = re.split(r'\nassistant\n', decoded, flags=re.IGNORECASE)
            if len(parts) > 1:
                full_output = parts[-1].strip()
            elif "assistant" in decoded.lower():
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
                print(f"  [DEBUG] Image passed but not cited. Full output:\n{full_output}\n{'='*60}")
            
            vlm_span.set_attribute("llm.output", full_output[:2000])
        
        # Parse output with consistency check (pass cross_encoder)
        state = parse_solver_output(state, full_output, state.choices, encoder=cross_encoder)
        solver_span.set_attribute("solver.final_answer", state.final_answer or "")
        solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps or "")
    
    return state

def solver_step_with_citation_retry(state: State, model, processor, kwargs) -> State:
    """
    Run the solver, then retry with lower temperature if citations are missing.

    Text retry  : fires when text evidence was retrieved but not cited.
    Image retry : fires when images were passed but not described/cited.
    """
    state = solver_step(state, model, processor, kwargs)
    reasoning = ' '.join(state.reasoning_steps or [])

    # Text citation retry
    has_text_citations = '[Text Evidence' in reasoning
    has_observations = bool(re.search(r'<(?:OBSERVATIONS|CAPTION)>', reasoning, re.IGNORECASE))
    has_image_citations = '[Question Image' in reasoning

    text_was_retrieved = bool(
        state.retrieved_chunks and any(
            ci.text_chunks for ci in state.retrieved_chunks.values()
        )
    )
    text_is_substantive = any(
        len(chunk) > 150
        for ci in state.retrieved_chunks.values()
        for chunk in ci.text_chunks[:3]
        if not chunk.strip().lower().startswith(
            ("natural science", "social science", "language science")
        )
    )

    # Also require retrieved text to be typically relevant to the question,
    # not just long. Purely visual questions (e.g. "which solution has more
    # green particles?") retrieve generic concentration/chemistry text that is
    # long but irrelevant — retrying for missing text citations forces the
    # solver to invent text-citation support for visual observations.
    question_keywords = {
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', state.question or '')
    }
    text_is_relevant = text_is_substantive and any(
        any(kw in chunk.lower() for kw in question_keywords)
        for ci in state.retrieved_chunks.values()
        for chunk in ci.text_chunks[:3]
        if len(chunk.strip()) > 150
    )
    # If solver already used visual reasoning, don't retry just for missing
    # text citations — the answer is grounded in the image, not text.
    solver_used_visual = bool(re.search(
        r'<(?:OBSERVATIONS|CAPTION)>', reasoning, re.IGNORECASE
    ))
    # Retry if no text citations despite having substantive, relevant text evidence,
    # and the solver didn't already anchor its answer in visual observations.
    if (not has_text_citations and text_was_retrieved
            and text_is_substantive and text_is_relevant
            and not solver_used_visual):
        print("No text citations despite retrieved evidence — retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        return solver_step(state, model, processor, retry_kwargs)

    # Retry if images exist but no observations/citations
    has_image_citations = '[Question Image' in reasoning
    images_were_passed = bool(
        any(p for p in (state.image_paths or []) if p and os.path.exists(p))
    )

    if images_were_passed and not has_observations and not has_image_citations:
        print("No image observations or citations — retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        return solver_step(state, model, processor, retry_kwargs)

    return state