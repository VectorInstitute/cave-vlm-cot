"""
Pydantic v2 models don't support dictionary-style access:

model['field'] → AttributeError
model.get('field') → AttributeError
'field' in model → Wrong behavior
model.field → Correct!
"""
import json

import torch
from planner import State
from tracer import tracer

import base64  # NEEDED for base64.b64decode()
from io import BytesIO  # NEEDED for BytesIO(img_data)
from PIL import Image  # NEEDED for Image.open()
import os  # NEEDED for os.path.exists()

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

# SOLVER_PROMPT_TEMPLATE = """You must answer using ONLY the provided evidence.

# Question: {question_text}
# Choices: {answer_choices}

# Evidence:
# {text_evidence}
# {visual_evidence}

# REQUIRED FORMAT - Follow exactly:

# <EVIDENCE_ANALYSIS>
# [Text Evidence 1] tells us: [summarize what this evidence says]
# [Text Evidence 2] tells us: [summarize what this evidence says]
# [Image ROI 1] shows: [describe what you see]
# [Image ROI 2] shows: [describe what you see]
# </EVIDENCE_ANALYSIS>

# <RELEVANT_EVIDENCE>
# For this question, the relevant evidence is:
# - [List which evidence items help answer the question]
# </RELEVANT_EVIDENCE>

# <REASONING>
# Based on [Evidence ID], [make a claim].
# Combining [Evidence ID] and [Evidence ID], [make another claim].
# </REASONING>

# <ANSWER>
# The answer is [LETTER]: [option]
# Supported by: [list evidence IDs used]
# </ANSWER>

# CRITICAL: You MUST complete the EVIDENCE_ANALYSIS section first!"""

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
You are an expert visual reasoning assistant. Answer the question using ONLY the provided evidence. You MUST cite specific evidence in every reasoning step.

EXAMPLE (showing required citation format)
Question: Which state is the most northern?
Choices:
A: West Virginia
B: Kentucky
C: Virginia

Text Evidence:
[Text Evidence 1]: West Virginia is located between 37N and 40.6N latitude.
[Text Evidence 2]: Kentucky spans from 36.5N to 39.1N latitude.

Visual Evidence:
[Image ROI 1]: Map showing state borders with compass rose pointing north.

<EVIDENCE_ANALYSIS>
[Text Evidence 1] tells us: West Virginia's northern boundary reaches 40.6N.
[Text Evidence 2] tells us: Kentucky's northern boundary only reaches 39.1N.
[Image ROI 1] shows: A compass rose and state outlines confirming relative positions.
</EVIDENCE_ANALYSIS>

<RELEVANT_EVIDENCE>
For this question, the relevant evidence is:
- [Text Evidence 1]: gives West Virginia's latitude range
- [Text Evidence 2]: gives Kentucky's latitude range
- [Image ROI 1]: visually confirms the northern position
</RELEVANT_EVIDENCE>

<REASONING>
Based on [Text Evidence 1], West Virginia reaches 40.6N, the highest latitude of any option.
Based on [Text Evidence 2], Kentucky only reaches 39.1N, which is south of West Virginia.
Combining [Text Evidence 1] and [Image ROI 1], West Virginia is confirmed as the most northern state.
</REASONING>

<ANSWER>
The answer is A: West Virginia
Supported by: [Text Evidence 1], [Text Evidence 2], [Image ROI 1]
</ANSWER>
END EXAMPLE

Now answer the following question using the same format:

Question: {question_text}
Choices:
{answer_choices}

Text Evidence:
{text_evidence}

Visual Evidence:
{visual_evidence}

INSTRUCTIONS:
1. In <EVIDENCE_ANALYSIS>, you MUST reference EVERY piece of evidence by its exact label (e.g., [Text Evidence 1], [Image ROI 2]).
2. In <REASONING>, EVERY sentence must cite at least one evidence label. Do not make claims without a citation.
3. In <ANSWER>, list ALL evidence labels you used.

<EVIDENCE_ANALYSIS>
"""

MAX_TOTAL_IMAGES = 2 
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
    Extract and format retrieved evidence from state for the solver prompt.
    Focus on quality over quantity - only include relevant, useful evidence.
    """
    text_evidence = []
    visual_evidence = []
    
    if state.retrieved_chunks:
        for query, chunk_info in state.retrieved_chunks.items():
            # chunk_info is a ChunkInfo Pydantic model - use attribute access
            # Access Pydantic model attributes directly, not with .get()
            # Text chunks
            for chunk in chunk_info.text_chunks:
                text_evidence.append(chunk)
            
            # Image ROI captions
            # roi is a RoiInfo Pydantic model - use attribute access
            for roi in chunk_info.image_rois:
                caption = roi.caption if roi.caption else 'No caption'
                visual_evidence.append(caption)

    # Format as numbered list
    text_str = "\n".join([
        f"[Text Evidence {i+1}]: {chunk}"
        for i, chunk in enumerate(text_evidence)
    ]) if text_evidence else "No text evidence retrieved."

    visual_str = "\n".join([
        f"[Image ROI {i+1}]: {caption}"
        for i, caption in enumerate(visual_evidence)
    ]) if visual_evidence else "No visual evidence retrieved."
    
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

def solver_step(state: State, model, processor, kwargs) -> State:
    with tracer.start_as_current_span(
        "Solver", openinference_span_kind="chain"
    ) as solver_span:
        # Load actual images
        images, image_descriptions = prepare_images_for_solver(state)
        # Format evidence
        text_evidence, visual_evidence_captions = format_retrieved_evidence(state)
        # For VLM, describe retrieved images as text captions in the prompt.
        # Do NOT inject <|image|> tokens manually here — the processor inserts them
        # when images are passed to processor(images=..., text=...).
        if images:
            visual_evidence = "\n\n".join([
                f"[{desc}]" for desc in image_descriptions
            ])
        else:
            visual_evidence = "No visual evidence retrieved."

        labeled_choices = format_labeled_choices(state.choices)
        prompt = SOLVER_PROMPT_TEMPLATE.format(
            question_text=state.question,
            answer_choices=labeled_choices,
            text_evidence=text_evidence,
            visual_evidence=visual_evidence
        )

        # Llama-3.2-Vision (MLlama) requires image tokens embedded in the message
        # content. Passing images=... to processor alone is not enough — the text
        # must contain exactly one <|image|> placeholder per image. The correct way
        # is to build a multimodal content list so apply_chat_template inserts them.
        if images:
            # Each image becomes a {"type": "image"} dict; text follows at the end.
            content_parts = [{"type": "image"} for _ in images]
            content_parts.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content_parts}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Log input attributes
        solver_span.set_attribute("solver.question", state.question)
        solver_span.set_attribute("solver.choices", json.dumps(state.choices))
        solver_span.set_attribute("solver.num_images", len(images))

        with tracer.start_as_current_span(
            "Llava-CoT", openinference_span_kind="llm"
        ) as vlm_span:
            vlm_span.set_attribute("vlm.model_name", "Xkev/Llama-3.2V-11B-cot")
            vlm_span.set_attribute("vlm.input_messages", str(messages))

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            # Process inputs (note: first arg is images, None for text-only)
            # Pass actual images to processor
            inputs = processor(
                images=images if images else None, 
                text=text,
                return_tensors="pt",
                max_num_tiles=2,          # caps cross-attn regardless of image count
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **kwargs)
                # outputs.shape = [batch_size, sequence_length]
                # e.g., tensor([[1, 2, 3, 4, 5, ...]])  # 2D tensor

                # Add these lines to free memory
                torch.cuda.empty_cache()
                import gc
                gc.collect()

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

        # Parse the response and update state
        # Extract the conclusion/answer from the response
        # conclusion_match = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', decoded, re.DOTALL)
        # if conclusion_match:
        #     conclusion = conclusion_match.group(1).strip()
        #     state.final_answer = conclusion
        #     solver_span.set_attribute("solver.final_answer", conclusion)

        # reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', decoded, re.DOTALL)
        # if reasoning_match:
        #     reasoning = reasoning_match.group(1).strip()
        #     # Split into steps if they're numbered
        #     steps = re.split(r'\nStep \d+:', reasoning)
        #     state.reasoning_steps = [s.strip() for s in steps if s.strip()]
        #     solver_span.set_attribute("solver.reasoning", state.reasoning_steps)

        # solver_span.set_attribute("solver.full_output", decoded)

        # Try to extract the final answer from the CONCLUSION section

        try:
            if "<CONCLUSION>" in full_output and "</CONCLUSION>" in full_output:
                conclusion_text = (
                    full_output.split("<CONCLUSION>")[1]
                    .split("</CONCLUSION>")[0]
                    .strip()
                )
                conclusion_text, valid = validate_and_correct_answer(conclusion_text, state.choices)
                if not valid:
                    print(f'Warning: Could not validate answer format for pid={state.pid}')
                state.final_answer = conclusion_text
            else:
                # Fallback: just use the full output
                state.final_answer = full_output.strip()
        except Exception as e:
            print(f"Error extracting conclusion: {e}")
            state.final_answer = full_output
        solver_span.set_attribute("solver.final_answer", state.final_answer)

        try:
            if "<REASONING>" in full_output and "</REASONING>" in full_output:
                reasoning_text = (
                    full_output.split("<REASONING>")[1].split("</REASONING>")[0].strip()
                )
                state.reasoning_steps = [reasoning_text] 
            else:
                # Fallback: just use the full output
                state.reasoning_steps = [full_output]
        except Exception as e:
            print(f"Error extracting reasoning_steps: {e}")
            state.reasoning_steps = [full_output]
        solver_span.set_attribute("solver.reasoning_steps", state.reasoning_steps)

    return state
