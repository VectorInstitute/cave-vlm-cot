"""
solver.py — InternVL2.5-8B solver for CaVe-VLM-CoT pipeline.
Replaces LLaVA-CoT with InternVL2.5-8B which has significantly better
structured output and citation following capabilities.
Key differences from LLaVA-CoT:
  - Uses model.chat() API instead of processor.apply_chat_template + generate
  - Images are passed as PIL objects to model.chat(); <image> tags in prompt
  - AutoTokenizer instead of AutoProcessor
  - Returns response string directly (no "assistant" splitting needed)

InternVL — you must manually preprocess images into tensors before passing them to model.chat(). This is why we added three new functions:
dynamic_preprocess() — InternVL uses "dynamic resolution." Instead of forcing all images to one size, it tiles the image into 448×448 patches that best preserve the original aspect ratio. A wide panoramic image might become 2×1 patches, a tall image 1×2, etc. This gives the model better visual understanding than fixed-size resizing.
build_transform() — Standard ImageNet normalization (the same mean/std values used during InternVL's training). Converts PIL images to tensors.
load_image_for_internvl() — Combines the above: tile the image, normalize each tile, stack into one tensor.
"""
import json

import torch
from planner import State
from tracer import tracer

import base64  # NEEDED for base64.b64decode()
from io import BytesIO  # NEEDED for BytesIO(img_data)
from PIL import Image  # NEEDED for Image.open()
import os  # NEEDED for os.path.exists()

import gc
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import re

# InternVL preprocessing helpers
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    ), key=lambda x: x[0] * x[1])
    # Find closest aspect ratio
    best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0]/r[1]))
    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    resized = image.resize((target_width, target_height))
    processed = [
        resized.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        ))
        for i in range(best_ratio[0] * best_ratio[1])
    ]
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed

def load_image_for_internvl(image, input_size=448, max_num=12):
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])

# SOLVER_PROMPT_TEMPLATE = """Answer this question using ONLY the evidence below. You MUST cite evidence.

# Question: {question_text}

# Choices:
# {answer_choices}

# Evidence:
# {text_evidence}
# {visual_evidence}

# Instructions:
# - Read each piece of evidence carefully
# - Cite evidence using [Text Evidence 1], [Text Evidence 2], [Image ROI 1], etc.
# - Every claim needs a citation

# Example format:
# "According to [Text Evidence 1], the latitude is 40N. From [Image ROI 1], I can see the map shows..."

# Now answer step by step with citations:

# Step 1: Looking at the evidence, [Text Evidence 1] states that"""

SOLVER_PROMPT_TEMPLATE = """
You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

Question: {question_text}
Choices: {answer_choices}

Retrieved Text Evidence:
{text_evidence}

Retrieved Visual Evidence:
{visual_evidence}

IMPORTANT: You MUST cite the evidence using exact labels (e.g., [Text Evidence 1], [Image ROI 2]) in your reasoning.

Answer using this structured format:
Answer this question using a structured chain-of-thought approach:

<SUMMARY>
State the core problem and your approach in 1-2 sentences.
</SUMMARY>

<CAPTION>
Describe key visual elements from the images above that are relevant to the question.
</CAPTION>

<REASONING>
Provide step-by-step reasoning. REQUIRED: Cite specific evidence using [Text Evidence N] and [Image ROI N] labels for each claim you make. Example:
- Step 1: According to [Text Evidence 1], ...
- Step 2: From [Image ROI 1], I can observe that...
- Step 3: Combining [Text Evidence 2] and [Image ROI 2], ...
</REASONING>

<CONCLUSION>
State your final answer as: "The answer is [LETTER]: [option text]"
</CONCLUSION>

---
EXAMPLE OUTPUT FORMAT:
<REASONING>
- Step 1: According to [Text Evidence 1], the map shows state locations.
- Step 2: Based on [Image ROI 1], I can see the compass rose indicating north.
- Step 3: Combining [Text Evidence 2] and [Image ROI 2], West Virginia is northernmost.
</REASONING>

Choices:
A: West Virginia
B: Kentucky
C: Tennessee
<CONCLUSION>
The answer is A: West Virginia [Text Evidence 1] [Image ROI 1]
</CONCLUSION>

CRITICAL: Answer the SPECIFIC question provided above. Do NOT copy or reference unrelated examples.
Use the format: <SUMMARY>...</SUMMARY> <CAPTION>...</CAPTION> <REASONING>...</REASONING> <CONCLUSION>The answer is [LETTER]: [option]</CONCLUSION>
REMEMBER: Cite evidence using [Text Evidence N] and [Image ROI N] labels in your reasoning.
"""

MAX_TOTAL_IMAGES = 4 # was 2 — InternVL handles multi-image better 
def prepare_images_for_solver(state):
    images = []
    image_descriptions = []

    # Step 1: Add question image ONLY if it exists
    for idx, img_path in enumerate(state.image_paths or []):
        if img_path and os.path.exists(img_path):
            try:
                images.append(Image.open(img_path).convert('RGB'))
                image_descriptions.append(f'Question Image {idx+1}')
            except Exception as e:
                print(f'Warning: {img_path}: {e}')

    # Step 2: Add ROI patches from retrieved chunks (any source image, not just question images)
    # old filter `if roi.source_image not in question_image_set` discarded all
    # externally-retrieved ROIs before the model ever saw them, making [Image ROI N]
    # citations impossible. We now include any ROI up to MAX_TOTAL_IMAGES.
    if state.retrieved_chunks:
        for subquery, chunk_info in state.retrieved_chunks.items():
            if len(images) >= MAX_TOTAL_IMAGES:
                break
            for roi in chunk_info.image_rois:
                if len(images) >= MAX_TOTAL_IMAGES:
                    break
                if roi.image_patch:
                    try:
                        img_data = base64.b64decode(roi.image_patch)
                        img = Image.open(BytesIO(img_data)).convert('RGB')
                        images.append(img)
                        image_descriptions.append(f'[Image ROI]: {roi.caption or "No caption"}')
                    except Exception as e:
                        print(f'Warning ROI: {e}')
    return images, image_descriptions
    
def format_retrieved_evidence(state):
    """
    Format evidence with clear labels that match what we ask the model to cite.
    """
    text_evidence = []
    visual_evidence = []
    
    text_idx = 1
    roi_idx = 1
    
    if state.retrieved_chunks:
        for query, chunk_info in state.retrieved_chunks.items():
            # Text chunks - limit to avoid overwhelming
            for chunk in chunk_info.text_chunks[:5]:  # Max 5 per query
                if chunk and len(chunk.strip()) > 10:
                    # Truncate very long chunks
                    chunk_text = chunk.strip()[:500]
                    text_evidence.append(f"[Text Evidence {text_idx}]: {chunk_text}")
                    text_idx += 1
            
            # Image ROI captions
            for roi in chunk_info.image_rois[:3]:  # Max 3 per query
                caption = roi.caption if roi.caption else "Visual region from image"
                visual_evidence.append(f"[Image ROI {roi_idx}]: {caption}")
                roi_idx += 1

    # Limit total evidence to avoid context overflow
    text_evidence = text_evidence[:10]
    visual_evidence = visual_evidence[:6]

    text_str = "\n".join(text_evidence) if text_evidence else "[No text evidence retrieved]"
    visual_str = "\n".join(visual_evidence) if visual_evidence else "[No visual evidence retrieved]"
    
    # Debug
    print(f"Evidence for solver: {len(text_evidence)} text, {len(visual_evidence)} visual")
    
    return text_str, visual_str

CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E']
def format_labeled_choices(choices):
    return '\n'.join([f'{CHOICE_LABELS[i]}: {c}' for i, c in enumerate(choices)])

def validate_and_correct_answer(conclusion_text, choices):
    """Cross-check letter against stated choice text. Return corrected text."""
    import re
    match = re.search(r'[Tt]he answer is ([A-Ea-e])[:\s]+([^\[\n]+)', conclusion_text)
    if not match:
        return conclusion_text, False  # Can't parse

    letter = match.group(1).upper()
    stated_text = match.group(2).strip().lower().rstrip('.')
    idx = ord(letter) - ord('A')

    # Validate: stated text should appear in choices[idx]
    if idx < len(choices):
        actual = choices[idx].lower()
        if stated_text in actual or actual in stated_text:
            return conclusion_text, True  # Consistent

    # Mismatch — find which choice matches the stated text
    for i, choice in enumerate(choices):
        if any(w in choice.lower() for w in stated_text.split()[:3] if len(w) > 3):
            corrected = f'The answer is {CHOICE_LABELS[i]}: {choices[i]}'
            print(f'Corrected: {letter}→{CHOICE_LABELS[i]} ({stated_text!r} → {choice!r})')
            return corrected, True

    return conclusion_text, False  # Could not correct

def solver_step(state: State, model, tokenizer, kwargs) -> State:
    with tracer.start_as_current_span(
        "Solver", openinference_span_kind="chain"
    ) as solver_span:
        
        # Prepare images
        images, image_descriptions = prepare_images_for_solver(state)
        
        # Format evidence
        text_evidence, visual_evidence = format_retrieved_evidence(state)
        
        # Format choices
        labeled_choices = format_labeled_choices(state.choices)
        
        # Build prompt
        prompt = SOLVER_PROMPT_TEMPLATE.format(
            question_text=state.question,
            answer_choices=labeled_choices,
            text_evidence=text_evidence,
            visual_evidence=visual_evidence
        )
        
        # Build image prefix and pixel_values for InternVL
        pixel_values_list = []
        image_prefix = ""
        if images:
            for img in images:
                pv = load_image_for_internvl(img, max_num=6)
                pixel_values_list.append(pv)
                image_prefix += "<image>\n"
            pixel_values = torch.cat(pixel_values_list, dim=0).to(
                dtype=torch.bfloat16, device=model.device
            )
        else:
            pixel_values = None

        full_prompt = image_prefix + prompt

        solver_span.set_attribute("solver.question", state.question)
        solver_span.set_attribute("solver.num_images", len(images))

        with tracer.start_as_current_span(
            "InternVL2.5-8B", openinference_span_kind="llm"
        ) as vlm_span:

            generation_config = {
                "max_new_tokens": kwargs.get("max_new_tokens", 1024),
                "do_sample": kwargs.get("do_sample", True),
                "temperature": kwargs.get("temperature", 0.3),
                "top_p": kwargs.get("top_p", 0.95),
            }

            with torch.no_grad():
                full_output = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=full_prompt,
                    generation_config=generation_config,
                )
            torch.cuda.empty_cache()
            gc.collect()

            # Debug citations
            text_cites = re.findall(r'\[Text Evidence \d+\]', full_output)
            roi_cites = re.findall(r'\[Image ROI \d+\]', full_output)
            print(f"Solver output: {len(full_output)} chars")
            print(f"  Citations: {len(text_cites)} text, {len(roi_cites)} ROI")
            if not text_cites and not roi_cites:
                print(f"  NO CITATIONS! Preview: {full_output[:400]}")
            vlm_span.set_attribute("llm.output", full_output[:2000])
        # Parse output
        state = parse_solver_output(state, full_output, state.choices)
        solver_span.set_attribute("solver.final_answer", state.final_answer or "")
    return state

# def solver_step_with_retry(state, model, processor, kwargs):
#     state = solver_step(state, model, processor, kwargs)
    
#     # Check for citations
#     reasoning = state.reasoning_steps[0] if state.reasoning_steps else ''
#     has_citations = '[Text Evidence' in reasoning or '[Image ROI' in reasoning
    
#     if not has_citations:
#         print("No citations, retrying...")
#         kwargs_retry = {**kwargs, 'temperature': 0.1}
#         state = solver_step(state, model, processor, kwargs_retry)
    
#     return state

def solver_step_with_citation_retry(state, model, tokenizer, kwargs):
    # First attempt
    state = solver_step(state, model, tokenizer, kwargs)
    
    # Check citations
    reasoning = ' '.join(state.reasoning_steps or [])
    has_citations = '[Text Evidence' in reasoning or '[Image ROI' in reasoning
    
    if not has_citations and state.retrieved_chunks:
        print("No citations, retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        state = solver_step(state, model, tokenizer, kwargs)
    
    return state

def parse_solver_output(state: State, full_output: str, choices: list) -> State:
    """Parse the solver output and extract answer + reasoning."""
    import re
    
    # Store full reasoning
    state.reasoning_steps = [full_output]
    
    # Try multiple patterns to find the answer
    answer_patterns = [
        r'[Tt]he answer is\s*([A-E])[:\s]*([^\n]+)',  # "The answer is A: option"
        r'[Aa]nswer[:\s]+([A-E])[:\s]*([^\n]*)',       # "Answer: A"
        r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',  # "A is correct"
        r'(?:choose|select|pick)\s*([A-E])',           # "choose A"
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
            # Check if choice text appears near end of output
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