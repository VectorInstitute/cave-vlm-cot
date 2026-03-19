"""
solver.py - Visual reasoning solver with:
- NO citation example in prompt (reduces model copying behavior)
- NO check_answer_consistency function
- Simpler prompt structure
- Lower hallucination rate due to less "confident" responses

Pydantic v2 models don't support dictionary-style access:
model['field'] → AttributeError
model.get('field') → AttributeError
'field' in model → Wrong behavior
model.field → Correct!
"""
import json
import re
from typing import List, Tuple

import torch
from utils import State
from tracer import tracer

import base64
from io import BytesIO
from PIL import Image
import os

# V17 SOLVER PROMPT - NO EXAMPLE (key difference from V22)
SOLVER_PROMPT_TEMPLATE = """
You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

Question: {question_text}

Choices:
{answer_choices}

Question Images (shown above in order):
{image_evidence}

Retrieved Text Evidence:
{text_evidence}

CITATION RULES:
- Use [Text Evidence N] for any claim drawn from the retrieved text above.
- Use [Question Image N] ONLY when that specific image provides visual evidence
  that directly supports your claim (e.g. a label, diagram feature, colour, or
  spatial relationship visible in the image). Do NOT cite an image just because
  the question has one — only cite it when you are actually using what you see
  in it to support a specific reasoning step.
- If neither text nor images provide relevant evidence for a claim, rely on
  general knowledge and do not add a citation.

Answer using this structured format:

<SUMMARY>
State the core problem in 1-2 sentences.
</SUMMARY>

<CAPTION>
Briefly describe any visual information in the Question Images that is directly
relevant to answering the question. If the images are not informative for this
question, state that explicitly.
</CAPTION>

<REASONING>
Step-by-step reasoning with selective citations:
- Step 1: According to [Text Evidence 1], ...
- Step 2: Looking at [Question Image 1], I can see that... (only if the image
  shows something relevant — e.g. a diagram, map, or chart you are reading)
- Step 3: Combining [Text Evidence 2] and [Question Image 1], ...
</REASONING>

<CONCLUSION>
The answer is [LETTER]: [option text]
</CONCLUSION>

CRITICAL RULES:
1. Cite [Text Evidence N] for every claim drawn from the retrieved text.
2. Cite [Question Image N] only when that image is genuinely useful evidence
   for the specific claim — not as a formality.
3. It is correct to have zero image citations if the images do not help answer
   the question.
4. If evidence is insufficient, state "Based on available evidence, I cannot
   determine..."
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

def solver_step(state: State, model, processor, kwargs) -> State:
    """
    V17 Solver - simpler prompt, no example, better grounding.
    """
    with tracer.start_as_current_span("Solver", openinference_span_kind="chain") as solver_span:
        
        # Prepare question images (simple sequential numbering)
        images, image_descriptions = prepare_question_images(state)
        
        # Format evidence
        text_evidence, image_evidence = format_evidence(state, image_descriptions)
        
        # Format choices
        labeled_choices = format_labeled_choices(state.choices)
        
        # Build prompt
        prompt = SOLVER_PROMPT_TEMPLATE.format(
            question_text=state.question,
            answer_choices=labeled_choices,
            image_evidence=image_evidence,
            text_evidence=text_evidence,
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
            
            # Decode
            full_output = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "assistant" in full_output.lower():
                parts = full_output.split("assistant")
                full_output = parts[-1].strip()
            
            vlm_span.set_attribute("vlm.output_length", len(full_output))
        
        # Parse output
        state = parse_solver_output(state, full_output, state.choices)
        
        solver_span.set_attribute("solver.final_answer", state.final_answer or "")
    
    return state

def parse_solver_output(state: State, full_output: str, choices: list) -> State:
    """Parse the solver output and extract answer + reasoning."""
    
    # Store full reasoning
    state.reasoning_steps = [full_output]
    
    # Try multiple patterns to find the answer
    answer_patterns = [
        r'[Tt]he answer is\s*([A-E])[\s:]*([^\n\[\]]+)',  # "The answer is A: option"
        r'[Aa]nswer[\s:]+([A-E])[\s:]*([^\n]*)',         # "Answer: A"
        r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',  # "A is correct"
        r'(?:choose|select|pick)\s*([A-E])',             # "choose A"
    ]
    
    answer_text = ""
    for pattern in answer_patterns:
        match = re.search(pattern, full_output, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(choices):
                answer_text = f"The answer is {letter}: {choices[idx]}"
                break
    
    # Fallback: Look for choice text mentioned
    if not answer_text:
        for i, choice in enumerate(choices):
            choice_lower = choice.lower()
            if choice_lower in full_output.lower()[-200:]:
                answer_text = f"The answer is {CHOICE_LABELS[i]}: {choice}"
                print(f"   Found answer by choice text: {CHOICE_LABELS[i]}")
                break
    
    if answer_text:
        state.final_answer = answer_text
    else:
        # Last resort
        state.final_answer = full_output[-200:]
        print(f"   Could not extract answer, using last 200 chars")
    
    return state

def solver_step_with_citation_retry(state: State, model, processor, kwargs) -> State:
    """
    V17 version - no retry logic needed since we don't have example to copy.
    Just call solver_step directly.
    """
    return solver_step(state, model, processor, kwargs)