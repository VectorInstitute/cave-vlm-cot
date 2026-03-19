"""
solver.py - Visual reasoning solver with:
1. Forced visual observation (OBSERVATIONS section)
2. Answer consistency check (reasoning must match conclusion)
3. Clearer answer format to prevent A/B confusion

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

# SOLVER_PROMPT_TEMPLATE = """
# You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

# Question: {question_text}
# Choices: {answer_choices}

# Retrieved Text Evidence:
# {text_evidence}

# Retrieved Visual Evidence:
# {visual_evidence}

# IMPORTANT: You MUST cite the evidence using exact labels (e.g., [Text Evidence 1], [Image ROI 2]) in your reasoning.

# Answer using this structured format:
# Answer this question using a structured chain-of-thought approach:

# <SUMMARY>
# State the core problem and your approach in 1-2 sentences.
# </SUMMARY>

# <CAPTION>
# Describe key visual elements from the images above that are relevant to the question.
# </CAPTION>

# <REASONING>
# Provide step-by-step reasoning. REQUIRED: Cite specific evidence using [Text Evidence N] and [Image ROI N] labels for each claim you make. Example:
# - Step 1: According to [Text Evidence 1], ...
# - Step 2: From [Image ROI 1], I can observe that...
# - Step 3: Combining [Text Evidence 2] and [Image ROI 2], ...
# </REASONING>

# <CONCLUSION>
# State your final answer as: "The answer is [LETTER]: [option text]"
# </CONCLUSION>

# ---
# EXAMPLE OUTPUT FORMAT:
# <REASONING>
# - Step 1: According to [Text Evidence 1], the map shows state locations.
# - Step 2: Based on [Image ROI 1], I can see the compass rose indicating north.
# - Step 3: Combining [Text Evidence 2] and [Image ROI 2], West Virginia is northernmost.
# </REASONING>

# Choices:
# A: West Virginia
# B: Kentucky
# C: Tennessee
# <CONCLUSION>
# The answer is A: West Virginia [Text Evidence 1] [Image ROI 1]
# </CONCLUSION>

# CRITICAL: Answer the SPECIFIC question provided above. Do NOT copy or reference unrelated examples.
# Use the format: <SUMMARY>...</SUMMARY> <CAPTION>...</CAPTION> <REASONING>...</REASONING> <CONCLUSION>The answer is [LETTER]: [option]</CONCLUSION>
# REMEMBER: Cite evidence using [Text Evidence N] and [Image ROI N] labels in your reasoning.
# """

# - If neither text nor images provide relevant evidence for a claim, rely on
#   general knowledge and do not add a citation.

# SOLVER_PROMPT_TEMPLATE = """
# You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

# Question: {question_text}

# Choices:
# {answer_choices}

# Question Images (shown above in order):
# {image_evidence}

# Retrieved Text Evidence:
# {text_evidence}

# CITATION RULES:
# - Use [Text Evidence N] for any claim drawn from the retrieved text above.
# - Use [Question Image N] ONLY when that specific image provides visual evidence
#   that directly supports your claim (e.g. a label, diagram feature, colour, or
#   spatial relationship visible in the image). Do NOT cite an image just because
#   the question has one — only cite it when you are actually using what you see
#   in it to support a specific reasoning step.

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

# SOLVER_PROMPT_TEMPLATE_IMAGE = """
# You are an expert visual reasoning assistant. Examine the images above carefully \
# and provide systematic, evidence-based answers.

# Question: {question_text}

# Choices:
# {answer_choices}

# Question Images (shown above in order):
# {image_evidence}

# Retrieved Text Evidence:
# {text_evidence}

# CITATION RULES:
# - Use [Text Evidence N] for any claim drawn from the retrieved text above.
# - Use [Question Image N] when that image provides direct visual evidence for \
# your claim — e.g. you are reading a label, diagram, chart, map, or colour \
# from it. You MUST use this format; do not write "the image shows" without \
# the bracket citation.

# ---
# EXAMPLE (do NOT copy this answer — answer YOUR question above):

# Question Images:
# [Question Image 1]: A diagram of a fish with a pointed downward-facing mouth
# [Question Image 2]: A diagram of a bird with a wide flat beak

# Question: Which animal is adapted for filter feeding?
# Choices:
# A: shark
# B: flamingo

# <SUMMARY>
# I need to identify which animal uses filter feeding based on visual and text evidence.
# </SUMMARY>

# <CAPTION>
# [Question Image 1] shows a fish with a narrow downward mouth — suited for \
# bottom feeding, not filtering.
# [Question Image 2] shows a bird with a wide, curved beak — consistent with \
# filter feeding anatomy.
# </CAPTION>

# <REASONING>
# - Step 1: Looking at [Question Image 1], the fish has a pointed downward mouth, \
# which is adapted for bottom feeding rather than filtering particles from water.
# - Step 2: Looking at [Question Image 2], the bird has a wide, curved beak. \
# According to [Text Evidence 2], flamingos use a specialised beak to filter \
# small organisms from water.
# - Step 3: Combining [Question Image 2] and [Text Evidence 2], the flamingo \
# is the filter feeder.
# </REASONING>

# <CONCLUSION>
# The answer is B: flamingo
# </CONCLUSION>
# ---

# Now answer YOUR question using the same format. \
# CRITICAL: use [Question Image N] citations in <CAPTION> and <REASONING> \
# whenever you describe something you see in one of the images above. \
# Do NOT write "the image shows …" without the bracket citation.

# <SUMMARY>
# State the core problem in 1-2 sentences.
# </SUMMARY>

# <CAPTION>
# Describe what you see in each Question Image and how it relates to the question.
# Use [Question Image N] for every visual observation.
# </CAPTION>

# <REASONING>
# Step-by-step reasoning with citations:
# - Step 1: Looking at [Question Image 1], I can see that ...
# - Step 2: According to [Text Evidence N], ...
# - Step 3: Combining [Question Image N] and [Text Evidence N], ...
# </REASONING>

# <CONCLUSION>
# The answer is [LETTER]: [option text]
# </CONCLUSION>
# """

# Prompt for questions WITH images - forces visual observation
SOLVER_PROMPT_TEMPLATE_IMAGE = """
You are an expert visual reasoning assistant.

Question: {question_text}

Choices:
{answer_choices}

Question Images (shown above):
{image_evidence}

Retrieved Text Evidence:
{text_evidence}

---
INSTRUCTIONS - Complete these steps IN ORDER:

STEP 1: OBSERVE THE IMAGES (Required - do not skip)
Look at each image carefully and describe what you actually see.
Be specific: labels, numbers, colors, arrows, text, patterns, structures.

STEP 2: REASON WITH EVIDENCE  
Connect your visual observations to the text evidence.
Explain how they help answer the question.

STEP 3: CONCLUDE
State your final answer. Make sure it matches your reasoning!

---
FORMAT:

<OBSERVATIONS>
Image 1: [What type of image? What specific details do you see?]
Image 2: [If present - what do you see?]
</OBSERVATIONS>

<REASONING>
Based on my observations:
- [What I see in the image tells me...]
- According to the text evidence, [relevant fact]...
- Therefore...
</REASONING>

<CONCLUSION>
The answer is [LETTER]: [full answer text]
</CONCLUSION>

---
CRITICAL REMINDERS:
- You MUST fill in the <OBSERVATIONS> section with specific visual details
- Your final answer MUST match your reasoning (don't contradict yourself)
- Include BOTH the letter AND the full answer text in your conclusion
"""

# def prepare_images_for_solver(state):
#     images = []
#     image_descriptions = []

#     # Step 1: Add question image ONLY if it exists
#     for idx, img_path in enumerate(state.image_paths or []):
#         if img_path and os.path.exists(img_path):
#             try:
#                 images.append(Image.open(img_path).convert('RGB'))
#                 image_descriptions.append(f'Question Image {idx+1}')
#             except Exception as e:
#                 print(f'Warning: {img_path}: {e}')

#     # Step 2: Add ROI patches from retrieved chunks (any source image, not just question images)
#     # old filter `if roi.source_image not in question_image_set` discarded all
#     # externally-retrieved ROIs before the model ever saw them, making [Image ROI N]
#     # citations impossible. We now include any ROI up to MAX_TOTAL_IMAGES.
#     if state.retrieved_chunks:
#         for subquery, chunk_info in state.retrieved_chunks.items():
#             if len(images) >= MAX_TOTAL_IMAGES:
#                 break
#             for roi in chunk_info.image_rois:
#                 if len(images) >= MAX_TOTAL_IMAGES:
#                     break
#                 # Resolve patch from cache if not embedded directly on the ROI.
#                 patch = roi.image_patch or state.image_patch_cache.get(roi.roi_id, "")
#                 if patch:
#                     try:
#                         img_data = base64.b64decode(patch)
#                         img = Image.open(BytesIO(img_data)).convert('RGB')
#                         images.append(img)
#                         image_descriptions.append(f'[Image ROI]: {roi.caption or "No caption"}')
#                     except Exception as e:
#                         print(f'Warning ROI: {e}')
#     return images, image_descriptions
    
# def format_retrieved_evidence(state, actual_image_descriptions=None):
#     """
#     Format evidence with clear labels that match what we ask the model to cite.
#     `actual_image_descriptions` must be the list returned by prepare_images_for_solver
#     so that [Image ROI N] labels in the prompt exactly match the images physically
#     passed to the VLM.  Passing more descriptions than images would let the model
#     cite ROIs it never saw.
#     """
#     text_evidence = []
#     visual_evidence = []
    
#     text_idx = 1
#     # roi_idx = 1
    
#     if state.retrieved_chunks:
#         for query, chunk_info in state.retrieved_chunks.items():
#             # Text chunks - limit to avoid overwhelming
#             for chunk in chunk_info.text_chunks[:5]:  # Max 5 per query
#                 if chunk and len(chunk.strip()) > 10:
#                     # Truncate very long chunks
#                     chunk_text = chunk.strip()[:300]
#                     text_evidence.append(f"[Text Evidence {text_idx}]: {chunk_text}")
#                     text_idx += 1
            
#     # Limit text to avoid context overflow
#     text_evidence = text_evidence[:8]
#     # Build visual evidence directly from the descriptions of images that were
#     # actually prepared.  Skip question-image slots (those aren't ROIs the model
#     # needs to cite); only label entries that came from retrieved ROIs.
#     if actual_image_descriptions:
#         roi_idx = 1
#         for desc in actual_image_descriptions:
#             if desc.startswith('[Image ROI]'):
#                 # Strip the generic prefix; the caption is after ': '
#                 caption = desc[len('[Image ROI]: '):]
#                 visual_evidence.append(f"[Image ROI {roi_idx}]: {caption}")
#                 roi_idx += 1
#     else:
#         # Fallback path (no image list provided): mirror old behaviour but cap
#         # at MAX_TOTAL_IMAGES so we never describe more ROIs than the VLM sees.
#         roi_idx = 1
#         roi_cap = MAX_TOTAL_IMAGES  # never exceed what prepare_images allows
#         if state.retrieved_chunks:
#             for query, chunk_info in state.retrieved_chunks.items():
#                 for roi in chunk_info.image_rois[:3]:
#                     if roi_idx > roi_cap:
#                         break
#                     caption = roi.caption if roi.caption else "Visual region from image"
#                     visual_evidence.append(f"[Image ROI {roi_idx}]: {caption}")
#                     roi_idx += 1

#     # Limit total evidence to avoid context overflow
#     text_evidence = text_evidence[:8]
#     visual_evidence = visual_evidence[:4]

#     text_str = "\n".join(text_evidence) if text_evidence else "[No text evidence retrieved]"
#     visual_str = "\n".join(visual_evidence) if visual_evidence else "[No visual evidence retrieved]"
    
#     # Debug
#     print(f"Evidence for solver: {len(text_evidence)} text, {len(visual_evidence)} visual"
#           f" (images sent to VLM: {len(actual_image_descriptions) if actual_image_descriptions else '?'})")
    
#     return text_str, visual_str

# def format_labeled_choices(choices):
#     return '\n'.join([f'{CHOICE_LABELS[i]}: {c}' for i, c in enumerate(choices)])

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

# def solver_step(state: State, model, processor, kwargs) -> State:
#     with tracer.start_as_current_span(
#         "Solver", openinference_span_kind="chain"
#     ) as solver_span:
        
#         # Prepare images — must happen BEFORE format_retrieved_evidence so the
#         # visual-evidence labels in the prompt match exactly what the VLM receives.
#         images, image_descriptions = prepare_images_for_solver(state)
        
#         # Format evidence (pass image_descriptions so ROI labels are in sync)
#         text_evidence, visual_evidence = format_retrieved_evidence(state, image_descriptions)
        
#         # Format choices
#         labeled_choices = format_labeled_choices(state.choices)
        
#         # Build prompt
#         prompt = SOLVER_PROMPT_TEMPLATE.format(
#             question_text=state.question,
#             answer_choices=labeled_choices,
#             text_evidence=text_evidence,
#             visual_evidence=visual_evidence
#         )

#         # Build messages for VLM
#         if images:
#             content_parts = [{"type": "image"} for _ in images]
#             content_parts.append({"type": "text", "text": prompt})
#             messages = [{"role": "user", "content": content_parts}]
#         else:
#             messages = [{"role": "user", "content": prompt}]

#         # Log
#         solver_span.set_attribute("solver.question", state.question)
#         solver_span.set_attribute("solver.num_images", len(images))

#         with tracer.start_as_current_span(
#             "Llama-3.2V", openinference_span_kind="llm"
#         ) as vlm_span:
            
#             text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
#             inputs = processor(
#                 images=images if images else None, 
#                 text=text,
#                 return_tensors="pt",
#             ).to(model.device)

#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     **kwargs
#                 )
#                 torch.cuda.empty_cache()
#                 import gc
#                 gc.collect()

#             decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

#             # Extract assistant response
#             if "assistant" in decoded.lower():
#                 full_output = decoded.split("assistant")[-1].strip()
#             else:
#                 full_output = decoded

#             # Debug: Check for citations
#             import re
#             text_cites = re.findall(r'\[Text Evidence \d+\]', full_output)
#             roi_cites = re.findall(r'\[Image ROI \d+\]', full_output)
            
#             print(f"Solver output: {len(full_output)} chars")
#             print(f" Citations: {len(text_cites)} text, {len(roi_cites)} ROI")
#             if not text_cites and not roi_cites:
#                 print(f"   NO CITATIONS! Output preview:")
#                 print(f"   {full_output[:400]}")

#             vlm_span.set_attribute("llm.output", full_output[:2000])
#         # Parse output"""
"""
import json
import re
from typing import List, Tuple

import torch
from utils import State
from tracer import tracer

import base64  # NEEDED for base64.b64decode()
from io import BytesIO  # NEEDED for BytesIO(img_data)
from PIL import Image  # NEEDED for Image.open()
import os  # NEEDED for os.path.exists()

from retriever import cross_encoder

# SOLVER_PROMPT_TEMPLATE = """
# You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

# Question: {question_text}
# Choices: {answer_choices}

# Retrieved Text Evidence:
# {text_evidence}

# Retrieved Visual Evidence:
# {visual_evidence}

# IMPORTANT: You MUST cite the evidence using exact labels (e.g., [Text Evidence 1], [Image ROI 2]) in your reasoning.

# Answer using this structured format:
# Answer this question using a structured chain-of-thought approach:

# <SUMMARY>
# State the core problem and your approach in 1-2 sentences.
# </SUMMARY>

# <CAPTION>
# Describe key visual elements from the images above that are relevant to the question.
# </CAPTION>

# <REASONING>
# Provide step-by-step reasoning. REQUIRED: Cite specific evidence using [Text Evidence N] and [Image ROI N] labels for each claim you make. Example:
# - Step 1: According to [Text Evidence 1], ...
# - Step 2: From [Image ROI 1], I can observe that...
# - Step 3: Combining [Text Evidence 2] and [Image ROI 2], ...
# </REASONING>

# <CONCLUSION>
# State your final answer as: "The answer is [LETTER]: [option text]"
# </CONCLUSION>

# ---
# EXAMPLE OUTPUT FORMAT:
# <REASONING>
# - Step 1: According to [Text Evidence 1], the map shows state locations.
# - Step 2: Based on [Image ROI 1], I can see the compass rose indicating north.
# - Step 3: Combining [Text Evidence 2] and [Image ROI 2], West Virginia is northernmost.
# </REASONING>

# Choices:
# A: West Virginia
# B: Kentucky
# C: Tennessee
# <CONCLUSION>
# The answer is A: West Virginia [Text Evidence 1] [Image ROI 1]
# </CONCLUSION>

# CRITICAL: Answer the SPECIFIC question provided above. Do NOT copy or reference unrelated examples.
# Use the format: <SUMMARY>...</SUMMARY> <CAPTION>...</CAPTION> <REASONING>...</REASONING> <CONCLUSION>The answer is [LETTER]: [option]</CONCLUSION>
# REMEMBER: Cite evidence using [Text Evidence N] and [Image ROI N] labels in your reasoning.
# """

# - If neither text nor images provide relevant evidence for a claim, rely on
#   general knowledge and do not add a citation.

# SOLVER_PROMPT_TEMPLATE = """
# You are an expert visual reasoning assistant. Examine the images above and provide systematic, evidence-based answers.

# Question: {question_text}

# Choices:
# {answer_choices}

# Question Images (shown above in order):
# {image_evidence}

# Retrieved Text Evidence:
# {text_evidence}

# CITATION RULES:
# - Use [Text Evidence N] for any claim drawn from the retrieved text above.
# - Use [Question Image N] ONLY when that specific image provides visual evidence
#   that directly supports your claim (e.g. a label, diagram feature, colour, or
#   spatial relationship visible in the image). Do NOT cite an image just because
#   the question has one — only cite it when you are actually using what you see
#   in it to support a specific reasoning step.

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

# SOLVER_PROMPT_TEMPLATE_IMAGE = """
# You are an expert visual reasoning assistant. Examine the images above carefully \
# and provide systematic, evidence-based answers.

# Question: {question_text}

# Choices:
# {answer_choices}

# Question Images (shown above in order):
# {image_evidence}

# Retrieved Text Evidence:
# {text_evidence}

# CITATION RULES:
# - Use [Text Evidence N] for any claim drawn from the retrieved text above.
# - Use [Question Image N] when that image provides direct visual evidence for \
# your claim — e.g. you are reading a label, diagram, chart, map, or colour \
# from it. You MUST use this format; do not write "the image shows" without \
# the bracket citation.

# ---
# EXAMPLE (do NOT copy this answer — answer YOUR question above):

# Question Images:
# [Question Image 1]: A diagram of a fish with a pointed downward-facing mouth
# [Question Image 2]: A diagram of a bird with a wide flat beak

# Question: Which animal is adapted for filter feeding?
# Choices:
# A: shark
# B: flamingo

# <SUMMARY>
# I need to identify which animal uses filter feeding based on visual and text evidence.
# </SUMMARY>

# <CAPTION>
# [Question Image 1] shows a fish with a narrow downward mouth — suited for \
# bottom feeding, not filtering.
# [Question Image 2] shows a bird with a wide, curved beak — consistent with \
# filter feeding anatomy.
# </CAPTION>

# <REASONING>
# - Step 1: Looking at [Question Image 1], the fish has a pointed downward mouth, \
# which is adapted for bottom feeding rather than filtering particles from water.
# - Step 2: Looking at [Question Image 2], the bird has a wide, curved beak. \
# According to [Text Evidence 2], flamingos use a specialised beak to filter \
# small organisms from water.
# - Step 3: Combining [Question Image 2] and [Text Evidence 2], the flamingo \
# is the filter feeder.
# </REASONING>

# <CONCLUSION>
# The answer is B: flamingo
# </CONCLUSION>
# ---

# Now answer YOUR question using the same format. \
# CRITICAL: use [Question Image N] citations in <CAPTION> and <REASONING> \
# whenever you describe something you see in one of the images above. \
# Do NOT write "the image shows …" without the bracket citation.

# <SUMMARY>
# State the core problem in 1-2 sentences.
# </SUMMARY>

# <CAPTION>
# Describe what you see in each Question Image and how it relates to the question.
# Use [Question Image N] for every visual observation.
# </CAPTION>

# <REASONING>
# Step-by-step reasoning with citations:
# - Step 1: Looking at [Question Image 1], I can see that ...
# - Step 2: According to [Text Evidence N], ...
# - Step 3: Combining [Question Image N] and [Text Evidence N], ...
# </REASONING>

# <CONCLUSION>
# The answer is [LETTER]: [option text]
# </CONCLUSION>
# """

# Prompt for questions WITH images - forces visual observation
SOLVER_PROMPT_TEMPLATE_IMAGE = """
You are an expert visual reasoning assistant.

Question: {question_text}

Choices:
{answer_choices}

Question Images (shown above):
{image_evidence}

Retrieved Text Evidence:
{text_evidence}

---
INSTRUCTIONS - Complete these steps IN ORDER:

STEP 1: OBSERVE THE IMAGES (Required - do not skip)
Look at each image carefully and describe what you actually see.
Be specific: labels, numbers, colors, arrows, text, patterns, structures.

STEP 2: REASON WITH EVIDENCE  
Connect your visual observations to the text evidence.
Explain how they help answer the question.

STEP 3: CONCLUDE
State your final answer.  Make sure it matches your reasoning!

---
FORMAT:

<OBSERVATIONS>
Image 1: [What type of image? What specific details do you see?]
Image 2: [If present - what do you see?]
</OBSERVATIONS>

<REASONING>
Based on my observations:
- [What I see in the image tells me...]
- According to the text evidence, [relevant fact]...
- Therefore...
</REASONING>

<CONCLUSION>
The answer is [LETTER]: [full answer text]
</CONCLUSION>

---
CRITICAL REMINDERS:
- You MUST fill in the <OBSERVATIONS> section with specific visual details
- Your final answer MUST match your reasoning (don't contradict yourself)
- Include BOTH the letter AND the full answer text in your conclusion
"""

# def prepare_images_for_solver(state):
#     images = []
#     image_descriptions = []

#     # Step 1: Add question image ONLY if it exists
#     for idx, img_path in enumerate(state.image_paths or []):
#         if img_path and os.path.exists(img_path):
#             try:
#                 images.append(Image.open(img_path).convert('RGB'))
#                 image_descriptions.append(f'Question Image {idx+1}')
#             except Exception as e:
#                 print(f'Warning: {img_path}: {e}')

#     # Step 2: Add ROI patches from retrieved chunks (any source image, not just question images)
#     # old filter `if roi.source_image not in question_image_set` discarded all
#     # externally-retrieved ROIs before the model ever saw them, making [Image ROI N]
#     # citations impossible. We now include any ROI up to MAX_TOTAL_IMAGES.
#     if state.retrieved_chunks:
#         for subquery, chunk_info in state.retrieved_chunks.items():
#             if len(images) >= MAX_TOTAL_IMAGES:
#                 break
#             for roi in chunk_info.image_rois:
#                 if len(images) >= MAX_TOTAL_IMAGES:
#                     break
#                 # Resolve patch from cache if not embedded directly on the ROI.
#                 patch = roi.image_patch or state.image_patch_cache.get(roi.roi_id, "")
#                 if patch:
#                     try:
#                         img_data = base64.b64decode(patch)
#                         img = Image.open(BytesIO(img_data)).convert('RGB')
#                         images.append(img)
#                         image_descriptions.append(f'[Image ROI]: {roi.caption or "No caption"}')
#                     except Exception as e:
#                         print(f'Warning ROI: {e}')
#     return images, image_descriptions
    
# def format_retrieved_evidence(state, actual_image_descriptions=None):
#     """
#     Format evidence with clear labels that match what we ask the model to cite.
#     `actual_image_descriptions` must be the list returned by prepare_images_for_solver
#     so that [Image ROI N] labels in the prompt exactly match the images physically
#     passed to the VLM.  Passing more descriptions than images would let the model
#     cite ROIs it never saw.
#     """
#     text_evidence = []
#     visual_evidence = []
    
#     text_idx = 1
#     # roi_idx = 1
    
#     if state.retrieved_chunks:
#         for query, chunk_info in state.retrieved_chunks.items():
#             # Text chunks - limit to avoid overwhelming
#             for chunk in chunk_info.text_chunks[:5]:  # Max 5 per query
#                 if chunk and len(chunk.strip()) > 10:
#                     # Truncate very long chunks
#                     chunk_text = chunk.strip()[:300]
#                     text_evidence.append(f"[Text Evidence {text_idx}]: {chunk_text}")
#                     text_idx += 1
            
#     # Limit text to avoid context overflow
#     text_evidence = text_evidence[:8]
#     # Build visual evidence directly from the descriptions of images that were
#     # actually prepared.  Skip question-image slots (those aren't ROIs the model
#     # needs to cite); only label entries that came from retrieved ROIs.
#     if actual_image_descriptions:
#         roi_idx = 1
#         for desc in actual_image_descriptions:
#             if desc.startswith('[Image ROI]'):
#                 # Strip the generic prefix; the caption is after ': '
#                 caption = desc[len('[Image ROI]: '):]
#                 visual_evidence.append(f"[Image ROI {roi_idx}]: {caption}")
#                 roi_idx += 1
#     else:
#         # Fallback path (no image list provided): mirror old behaviour but cap
#         # at MAX_TOTAL_IMAGES so we never describe more ROIs than the VLM sees.
#         roi_idx = 1
#         roi_cap = MAX_TOTAL_IMAGES  # never exceed what prepare_images allows
#         if state.retrieved_chunks:
#             for query, chunk_info in state.retrieved_chunks.items():
#                 for roi in chunk_info.image_rois[:3]:
#                     if roi_idx > roi_cap:
#                         break
#                     caption = roi.caption if roi.caption else "Visual region from image"
#                     visual_evidence.append(f"[Image ROI {roi_idx}]: {caption}")
#                     roi_idx += 1

#     # Limit total evidence to avoid context overflow
#     text_evidence = text_evidence[:8]
#     visual_evidence = visual_evidence[:4]

#     text_str = "\n".join(text_evidence) if text_evidence else "[No text evidence retrieved]"
#     visual_str = "\n".join(visual_evidence) if visual_evidence else "[No visual evidence retrieved]"
    
#     # Debug
#     print(f"Evidence for solver: {len(text_evidence)} text, {len(visual_evidence)} visual"
#           f" (images sent to VLM: {len(actual_image_descriptions) if actual_image_descriptions else '?'})")
    
#     return text_str, visual_str

# def format_labeled_choices(choices):
#     return '\n'.join([f'{CHOICE_LABELS[i]}: {c}' for i, c in enumerate(choices)])

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

# def solver_step(state: State, model, processor, kwargs) -> State:
#     with tracer.start_as_current_span(
#         "Solver", openinference_span_kind="chain"
#     ) as solver_span:
        
#         # Prepare images — must happen BEFORE format_retrieved_evidence so the
#         # visual-evidence labels in the prompt match exactly what the VLM receives.
#         images, image_descriptions = prepare_images_for_solver(state)
        
#         # Format evidence (pass image_descriptions so ROI labels are in sync)
#         text_evidence, visual_evidence = format_retrieved_evidence(state, image_descriptions)
        
#         # Format choices
#         labeled_choices = format_labeled_choices(state.choices)
        
#         # Build prompt
#         prompt = SOLVER_PROMPT_TEMPLATE.format(
#             question_text=state.question,
#             answer_choices=labeled_choices,
#             text_evidence=text_evidence,
#             visual_evidence=visual_evidence
#         )

#         # Build messages for VLM
#         if images:
#             content_parts = [{"type": "image"} for _ in images]
#             content_parts.append({"type": "text", "text": prompt})
#             messages = [{"role": "user", "content": content_parts}]
#         else:
#             messages = [{"role": "user", "content": prompt}]

#         # Log
#         solver_span.set_attribute("solver.question", state.question)
#         solver_span.set_attribute("solver.num_images", len(images))

#         with tracer.start_as_current_span(
#             "Llama-3.2V", openinference_span_kind="llm"
#         ) as vlm_span:
            
#             text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
#             inputs = processor(
#                 images=images if images else None, 
#                 text=text,
#                 return_tensors="pt",
#             ).to(model.device)

#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     **kwargs
#                 )
#                 torch.cuda.empty_cache()
#                 import gc
#                 gc.collect()

#             decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

#             # Extract assistant response
#             if "assistant" in decoded.lower():
#                 full_output = decoded.split("assistant")[-1].strip()
#             else:
#                 full_output = decoded

#             # Debug: Check for citations
#             import re
#             text_cites = re.findall(r'\[Text Evidence \d+\]', full_output)
#             roi_cites = re.findall(r'\[Image ROI \d+\]', full_output)
            
#             print(f"Solver output: {len(full_output)} chars")
#             print(f" Citations: {len(text_cites)} text, {len(roi_cites)} ROI")
#             if not text_cites and not roi_cites:
#                 print(f"   NO CITATIONS! Output preview:")
#                 print(f"   {full_output[:400]}")

#             vlm_span.set_attribute("llm.output", full_output[:2000])
#         # Parse output
#         state = parse_solver_output(state, full_output, state.choices)
#         solver_span.set_attribute("solver.final_answer", state.final_answer or "")
#         solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps or "")
#     return state

# def solver_step_with_citation_retry(state, model, processor, kwargs):
#     # First attempt
#     state = solver_step(state, model, processor, kwargs)
    
#     # Check citations
#     reasoning = ' '.join(state.reasoning_steps or [])
#     has_citations = '[Text Evidence' in reasoning or '[Image ROI' in reasoning
    
#     if not has_citations and state.retrieved_chunks:
#         print("No citations, retrying with lower temperature...")
#         retry_kwargs = {**kwargs, 'temperature': 0.1}
#         state = solver_step(state, model, processor, retry_kwargs)
    
#     return state

# def parse_solver_output(state: State, full_output: str, choices: list) -> State:
#     """Parse the solver output and extract answer + reasoning."""
#     import re
    
#     # Store full reasoning
#     state.reasoning_steps = [full_output]
    
#     # Try multiple patterns to find the answer
#     answer_patterns = [
#         r'[Tt]he answer is\s*([A-E])[:\s]*([^\n]+)',  # "The answer is A: option"
#         r'[Aa]nswer[:\s]+([A-E])[:\s]*([^\n]*)',       # "Answer: A"
#         r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',  # "A is correct"
#         r'(?:choose|select|pick)\s*([A-E])',           # "choose A"
#     ]
    
#     answer_text = ""
#     for pattern in answer_patterns:
#         match = re.search(pattern, full_output, re.IGNORECASE)
#         if match:
#             letter = match.group(1).upper()
#             idx = ord(letter) - ord('A')
#             if 0 <= idx < len(choices):
#                 answer_text = f"The answer is {letter}: {choices[idx]}"
#                 break
    
#     # Fallback: Look for choice text mentioned
#     if not answer_text:
#         for i, choice in enumerate(choices):
#             # Check if choice text appears near end of output
#             choice_lower = choice.lower()
#             if choice_lower in full_output.lower()[-200:]:
#                 answer_text = f"The answer is {CHOICE_LABELS[i]}: {choice}"
#                 print(f"   Found answer by choice text: {CHOICE_LABELS[i]}")
#                 break
    
#     if answer_text:
#         state.final_answer = answer_text
#     else:
#         # Last resort
#         state.final_answer = full_output[-200:]
#         print(f"   Could not extract answer, using last 200 chars")
    
#     return state

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

def solver_step(state: State, model, processor, kwargs) -> State:
    """
    Solver with [Question Image N] citations.
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
            
            if "assistant" in decoded.lower():
                full_output = decoded.split("assistant")[-1].strip()
            else:
                full_output = decoded
            
            # Check citations
            text_cites = re.findall(r'\[Text Evidence \d+\]', full_output)
            image_cites = re.findall(r'\[Question Image \d+\]', full_output)
            
            print(f"Solver output: {len(full_output)} chars")
            print(f"  Citations: {len(text_cites)} text, {len(image_cites)} image "
                  f"({'image cited as useful evidence' if image_cites else 'image not cited (not useful or no image)'})")
            
            if images and not image_cites:
                print(f"  [DEBUG] Image passed but not cited. Full output:\n{full_output}\n{'='*60}")
            
            vlm_span.set_attribute("llm.output", full_output[:2000])
        
        state = parse_solver_output(state, full_output, state.choices, cross_encoder)
        solver_span.set_attribute("solver.final_answer", state.final_answer or "")
        solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps or "")
    
    return state

# def parse_solver_output(state: State, full_output: str, choices: list) -> State:
#     """Parse solver output, scoping answer extraction to <CONCLUSION> block."""
#     state.reasoning_steps = [full_output]
    
#     # Prefer the CONCLUSION block to avoid matching answer-like patterns that
#     # appear earlier in REASONING (e.g. "A is incorrect because…").
#     conclusion_match = re.search(
#         r'<CONCLUSION>(.*?)</CONCLUSION>', full_output, re.IGNORECASE | re.DOTALL
#     )
#     search_text = conclusion_match.group(1).strip() if conclusion_match else full_output

#     answer_patterns = [
#         r'[Tt]he answer is\s*([A-E])[:\s]*([^\n]+)',
#         r'[Aa]nswer[:\s]+([A-E])[:\s]*([^\n]*)',
#         r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',
#         r'(?:choose|select|pick)\s*([A-E])',
#     ]
    
#     answer_text = ""
#     for pattern in answer_patterns:
#         match = re.search(pattern, search_text, re.IGNORECASE)
#         if match:
#             letter = match.group(1).upper()
#             idx = ord(letter) - ord('A')
#             if 0 <= idx < len(choices):
#                 answer_text = f"The answer is {letter}: {choices[idx]}"
#                 break
    
#     # Fallback: choice text in the tail of the conclusion (or full output)
#     if not answer_text:
#         tail = search_text[-300:] if len(search_text) > 300 else search_text
#         for i, choice in enumerate(choices):
#             if choice.lower() in tail.lower():
#                 answer_text = f"The answer is {CHOICE_LABELS[i]}: {choice}"
#                 break
    
#     state.final_answer = answer_text if answer_text else full_output[-200:]
#     return state

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
    encoder = cross_encoder or _get_cross_encoder()
    
    # Score each choice against the reasoning
    # Query: "Based on this reasoning, which answer is correct?"
    pairs = [
        [reasoning, f"The correct answer is {CHOICE_LABELS[i]}: {choice}"]
        for i, choice in enumerate(choices)
    ]
    
    scores = encoder.predict(pairs)
    
    # Find best supported choice
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    stated_score = scores[stated_idx]
    
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
            cross_encoder=cross_encoder,
            threshold=0.5,  # Require 0.5 score difference to correct
        )
        
        if not is_consistent and corrected:
            print(f"  [CORRECTED] {state.final_answer} → {corrected}")
            state.final_answer = corrected
    
    return state

# def solver_step_with_citation_retry(state, model, processor, kwargs):
#     """Retry with lower temperature if text evidence was retrieved but not cited.
#     Image citations ([Question Image N]) are intentionally selective — the model
#     should cite an image only when it provides useful visual evidence.  A response
#     with zero image citations is perfectly valid, so missing image citations alone
#     do NOT trigger a retry.
#     """
#     state = solver_step(state, model, processor, kwargs)
#     reasoning = ' '.join(state.reasoning_steps or [])
#     has_text_citations = '[Text Evidence' in reasoning
#     # Only retry when the retriever found text evidence but the solver ignored it entirely.
#     text_was_retrieved = bool(
#         state.retrieved_chunks and any(
#             ci.text_chunks for ci in state.retrieved_chunks.values()
#         )
#     )
#     # if not has_text_citations and text_was_retrieved:
#     text_is_substantive = any(
#         len(chunk) > 150  # generic corpus entries are typically short
#         for ci in state.retrieved_chunks.values()
#         for chunk in ci.text_chunks[:3]
#         if not chunk.strip().startswith(("natural science", "social science", "language science"))
#     )
#     if not has_text_citations and text_was_retrieved and text_is_substantive:
#         print("No text citations despite retrieved evidence — retrying with lower temperature...")
#         retry_kwargs = {**kwargs, 'temperature': 0.1}
#         state = solver_step(state, model, processor, retry_kwargs)
    
#     return state

def solver_step_with_citation_retry(state: State, model, processor, kwargs) -> State:
    """
    Run the solver, then retry with lower temperature if citations are missing.

    Text retry  : fires when text evidence was retrieved but not cited.
    Image retry : fires when images were passed but [Question Image N] is absent.

    Each retry is independent — if both are needed, only one retry fires
    (whichever condition is checked first).  This avoids running the model
    three times per sample.  Text citation is checked first because it is
    the more common failure mode.
    """
    state = solver_step(state, model, processor, kwargs)
    reasoning = ' '.join(state.reasoning_steps or [])

    # Text citation retry
    has_text_citations = '[Text Evidence' in reasoning
    text_was_retrieved = bool(
        state.retrieved_chunks and any(
            ci.text_chunks for ci in state.retrieved_chunks.values()
        )
    )
    text_is_substantive = any(
        len(chunk) > 150
        for ci in state.retrieved_chunks.values()
        for chunk in ci.text_chunks[:3]
        if not chunk.strip().startswith(("natural science", "social science", "language science"))
    )

    if not has_text_citations and text_was_retrieved and text_is_substantive:
        print("No text citations despite retrieved evidence — retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        return solver_step(state, model, processor, retry_kwargs)

    # Image citation retry
    # Only retry when:
    #   1. Images were actually passed to the model (image_paths exist on disk)
    #   2. No [Question Image N] citations appear in the output
    # We do NOT retry if no images were available — that would be pointless.
    has_image_citations = '[Question Image' in reasoning
    images_were_passed = bool(
        any(p for p in (state.image_paths or []) if p and os.path.exists(p))
    )

    if not has_text_citations and not has_image_citations and images_were_passed:
        print("No image citations despite question images being present — retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        return solver_step(state, model, processor, retry_kwargs)

    return state
#         state = parse_solver_output(state, full_output, state.choices)
#         solver_span.set_attribute("solver.final_answer", state.final_answer or "")
#         solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps or "")
#     return state

# def solver_step_with_citation_retry(state, model, processor, kwargs):
#     # First attempt
#     state = solver_step(state, model, processor, kwargs)
    
#     # Check citations
#     reasoning = ' '.join(state.reasoning_steps or [])
#     has_citations = '[Text Evidence' in reasoning or '[Image ROI' in reasoning
    
#     if not has_citations and state.retrieved_chunks:
#         print("No citations, retrying with lower temperature...")
#         retry_kwargs = {**kwargs, 'temperature': 0.1}
#         state = solver_step(state, model, processor, retry_kwargs)
    
#     return state

# def parse_solver_output(state: State, full_output: str, choices: list) -> State:
#     """Parse the solver output and extract answer + reasoning."""
#     import re
    
#     # Store full reasoning
#     state.reasoning_steps = [full_output]
    
#     # Try multiple patterns to find the answer
#     answer_patterns = [
#         r'[Tt]he answer is\s*([A-E])[:\s]*([^\n]+)',  # "The answer is A: option"
#         r'[Aa]nswer[:\s]+([A-E])[:\s]*([^\n]*)',       # "Answer: A"
#         r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',  # "A is correct"
#         r'(?:choose|select|pick)\s*([A-E])',           # "choose A"
#     ]
    
#     answer_text = ""
#     for pattern in answer_patterns:
#         match = re.search(pattern, full_output, re.IGNORECASE)
#         if match:
#             letter = match.group(1).upper()
#             idx = ord(letter) - ord('A')
#             if 0 <= idx < len(choices):
#                 answer_text = f"The answer is {letter}: {choices[idx]}"
#                 break
    
#     # Fallback: Look for choice text mentioned
#     if not answer_text:
#         for i, choice in enumerate(choices):
#             # Check if choice text appears near end of output
#             choice_lower = choice.lower()
#             if choice_lower in full_output.lower()[-200:]:
#                 answer_text = f"The answer is {CHOICE_LABELS[i]}: {choice}"
#                 print(f"   Found answer by choice text: {CHOICE_LABELS[i]}")
#                 break
    
#     if answer_text:
#         state.final_answer = answer_text
#     else:
#         # Last resort
#         state.final_answer = full_output[-200:]
#         print(f"   Could not extract answer, using last 200 chars")
    
#     return state

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

def solver_step(state: State, model, processor, kwargs) -> State:
    """
    Solver with [Question Image N] citations.
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
            
            if "assistant" in decoded.lower():
                full_output = decoded.split("assistant")[-1].strip()
            else:
                full_output = decoded
            
            # Check citations
            text_cites = re.findall(r'\[Text Evidence \d+\]', full_output)
            image_cites = re.findall(r'\[Question Image \d+\]', full_output)
            
            print(f"Solver output: {len(full_output)} chars")
            print(f"  Citations: {len(text_cites)} text, {len(image_cites)} image "
                  f"({'image cited as useful evidence' if image_cites else 'image not cited (not useful or no image)'})")
            
            if images and not image_cites:
                print(f"  [DEBUG] Image passed but not cited. Full output:\n{full_output}\n{'='*60}")
            
            vlm_span.set_attribute("llm.output", full_output[:2000])
        
        state = parse_solver_output(state, full_output, state.choices, cross_encoder)
        solver_span.set_attribute("solver.final_answer", state.final_answer or "")
        solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps or "")
    
    return state

# def parse_solver_output(state: State, full_output: str, choices: list) -> State:
#     """Parse solver output, scoping answer extraction to <CONCLUSION> block."""
#     state.reasoning_steps = [full_output]
    
#     # Prefer the CONCLUSION block to avoid matching answer-like patterns that
#     # appear earlier in REASONING (e.g. "A is incorrect because…").
#     conclusion_match = re.search(
#         r'<CONCLUSION>(.*?)</CONCLUSION>', full_output, re.IGNORECASE | re.DOTALL
#     )
#     search_text = conclusion_match.group(1).strip() if conclusion_match else full_output

#     answer_patterns = [
#         r'[Tt]he answer is\s*([A-E])[:\s]*([^\n]+)',
#         r'[Aa]nswer[:\s]+([A-E])[:\s]*([^\n]*)',
#         r'\b([A-E])\s*is\s*(?:the\s*)?(?:correct|right|best)',
#         r'(?:choose|select|pick)\s*([A-E])',
#     ]
    
#     answer_text = ""
#     for pattern in answer_patterns:
#         match = re.search(pattern, search_text, re.IGNORECASE)
#         if match:
#             letter = match.group(1).upper()
#             idx = ord(letter) - ord('A')
#             if 0 <= idx < len(choices):
#                 answer_text = f"The answer is {letter}: {choices[idx]}"
#                 break
    
#     # Fallback: choice text in the tail of the conclusion (or full output)
#     if not answer_text:
#         tail = search_text[-300:] if len(search_text) > 300 else search_text
#         for i, choice in enumerate(choices):
#             if choice.lower() in tail.lower():
#                 answer_text = f"The answer is {CHOICE_LABELS[i]}: {choice}"
#                 break
    
#     state.final_answer = answer_text if answer_text else full_output[-200:]
#     return state

def check_answer_consistency(
    reasoning: str, 
    stated_answer: str, 
    choices: list,
    cross_encoder=None,
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
    encoder = cross_encoder or _get_cross_encoder()
    
    # Score each choice against the reasoning
    # Query: "Based on this reasoning, which answer is correct?"
    pairs = [
        [reasoning, f"The correct answer is {CHOICE_LABELS[i]}: {choice}"]
        for i, choice in enumerate(choices)
    ]
    
    scores = encoder.predict(pairs)
    
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

def parse_solver_output(state, full_output: str, choices: list, cross_encoder=None):
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
            cross_encoder=cross_encoder,
            threshold=0.5,  # Require 0.5 score difference to correct
        )
        
        if not is_consistent and corrected:
            print(f"  [CORRECTED] {state.final_answer} → {corrected}")
            state.final_answer = corrected
    
    return state

# def solver_step_with_citation_retry(state, model, processor, kwargs):
#     """Retry with lower temperature if text evidence was retrieved but not cited.
#     Image citations ([Question Image N]) are intentionally selective — the model
#     should cite an image only when it provides useful visual evidence.  A response
#     with zero image citations is perfectly valid, so missing image citations alone
#     do NOT trigger a retry.
#     """
#     state = solver_step(state, model, processor, kwargs)
#     reasoning = ' '.join(state.reasoning_steps or [])
#     has_text_citations = '[Text Evidence' in reasoning
#     # Only retry when the retriever found text evidence but the solver ignored it entirely.
#     text_was_retrieved = bool(
#         state.retrieved_chunks and any(
#             ci.text_chunks for ci in state.retrieved_chunks.values()
#         )
#     )
#     # if not has_text_citations and text_was_retrieved:
#     text_is_substantive = any(
#         len(chunk) > 150  # generic corpus entries are typically short
#         for ci in state.retrieved_chunks.values()
#         for chunk in ci.text_chunks[:3]
#         if not chunk.strip().startswith(("natural science", "social science", "language science"))
#     )
#     if not has_text_citations and text_was_retrieved and text_is_substantive:
#         print("No text citations despite retrieved evidence — retrying with lower temperature...")
#         retry_kwargs = {**kwargs, 'temperature': 0.1}
#         state = solver_step(state, model, processor, retry_kwargs)
    
#     return state

def solver_step_with_citation_retry(state: State, model, processor, kwargs) -> State:
    """
    Run the solver, then retry with lower temperature if citations are missing.

    Text retry  : fires when text evidence was retrieved but not cited.
    Image retry : fires when images were passed but [Question Image N] is absent.

    Each retry is independent — if both are needed, only one retry fires
    (whichever condition is checked first).  This avoids running the model
    three times per sample.  Text citation is checked first because it is
    the more common failure mode.
    """
    state = solver_step(state, model, processor, kwargs)
    reasoning = ' '.join(state.reasoning_steps or [])

    # Text citation retry
    has_text_citations = '[Text Evidence' in reasoning
    text_was_retrieved = bool(
        state.retrieved_chunks and any(
            ci.text_chunks for ci in state.retrieved_chunks.values()
        )
    )
    text_is_substantive = any(
        len(chunk) > 150
        for ci in state.retrieved_chunks.values()
        for chunk in ci.text_chunks[:3]
        if not chunk.strip().startswith(("natural science", "social science", "language science"))
    )

    if not has_text_citations and text_was_retrieved and text_is_substantive:
        print("No text citations despite retrieved evidence — retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        return solver_step(state, model, processor, retry_kwargs)

    # Image citation retry
    # Only retry when:
    #   1. Images were actually passed to the model (image_paths exist on disk)
    #   2. No [Question Image N] citations appear in the output
    # We do NOT retry if no images were available — that would be pointless.
    has_image_citations = '[Question Image' in reasoning
    images_were_passed = bool(
        any(p for p in (state.image_paths or []) if p and os.path.exists(p))
    )

    if not has_text_citations and not has_image_citations and images_were_passed:
        print("No image citations despite question images being present — retrying with lower temperature...")
        retry_kwargs = {**kwargs, 'temperature': 0.1}
        return solver_step(state, model, processor, retry_kwargs)

    return state