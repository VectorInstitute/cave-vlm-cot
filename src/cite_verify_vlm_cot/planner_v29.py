import ast
import json
import math
from typing import Dict, List, Optional

import pandas as pd
import torch
from evaluations import planner_coverage_score, planner_specificity_score
import re

# CLIPProcessor handles image pre-processing like resizing and normalization
from tracer import tracer
from utils import safe_str, safe_parse_json, RoiInfo, ChunkInfo, State

# Enable only the safe fallback backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# SDPA = Scaled Dot-Product Attention, core math behind attention
# PyTorch provides multiple implementations of this, called backends (FlashAttention, SDPA(default), Math)
# SDPA is fast but allocates large temporary buffers, especially bad for long sequences + big models
# Prefer FlashAttention (best) if your GPU supports it

# df = pd.read_csv("scienceqa_augmented.csv")
# states = []

# for idx, row in df.iterrows():
#     pid = safe_str(row.get("pid"))
#     question = safe_str(row.get("question"))
#     hint = safe_str(row.get("hint"))
#     lecture = safe_str(row.get("lecture"))
#     subject = safe_str(row.get("subject"))
#     topic = safe_str(row.get("topic"))
#     skill = safe_str(row.get("skill"))
#     category = safe_str(row.get("category"))

#     # Parse image_paths (JSON array)
#     image_paths = safe_parse_json(row.get("image_paths"), default=[])
#     if not isinstance(image_paths, list):
#         image_paths = []

#     # Parse choices (JSON array or string representation)
#     choices_raw = row.get("choices")
#     if isinstance(choices_raw, str):
#         choices = safe_parse_json(choices_raw, default=[])
#     else:
#         choices = choices_raw if choices_raw else []

#     # Parse answer
#     answer = int(row["answer"]) if pd.notna(row.get("answer")) else -1

#     # Parse img_captions and img_ocr (JSON objects)
#     img_captions = safe_parse_json(row.get("img_captions"), default={})
#     img_ocr = safe_parse_json(row.get("img_ocr"), default={})

#     # Build image context from captions and OCR
#     image_context_parts = []
#     if img_captions:
#         captions_text = "\n".join([f"{k}: {v}" for k, v in img_captions.items()])
#         image_context_parts.append(f"Image Captions:\n{captions_text}")

#     if img_ocr:
#         ocr_text = "\n".join([f"{k}: {v}" for k, v in img_ocr.items()])
#         image_context_parts.append(f"OCR Text:\n{ocr_text}")

#     image_context = "\n\n".join(image_context_parts) if image_context_parts else ""

#     # Determine gold answer
#     gold_answer = choices[answer] if 0 <= answer < len(choices) else ""

#     state = State(
#         pid=pid,
#         question=question,
#         hint=hint,
#         image_paths=image_paths,
#         lecture=lecture,
#         choices=choices,
#         answer=answer,
#         img_captions=img_captions,
#         img_ocr=img_ocr,
#         subqueries=[],
#         retrieved_chunks={},
#         reasoning_steps=[],
#         final_answer="",
#         verdict="",
#         gold_answer=gold_answer,
#         subject=subject,
#         topic=topic,
#         skill=skill,
#         category=category
#     )

#     states.append(state)

# print(f"Loaded {len(states)} states from CSV")

# # Print first state
# if states:
#     print("\nFirst state:")
#     print(f"  PID: {states[0].pid}")
#     print(f"  Question: {states[0].question[:100]}...")
#     print(f"  Image paths: {states[0].image_paths}")
#     print(f"  Choices: {states[0].choices}")
#     print(f"  Gold answer: {states[0].gold_answer}")

# PLANNER_PROMPT_TEMPLATE = """Generate search queries to help answer a question.

# Example 1:
# Question: Which state is farthest north?
# Choices: ["Texas", "Maine", "Florida"]
# Queries:
# 1. Maine latitude coordinates
# 2. Texas geographic location
# 3. northernmost US states list
# 4. state latitude comparison

# Example 2:
# Question: What tense is used? "She will dance tomorrow."
# Choices: ["past", "present", "future"]
# Queries:
# 1. future tense definition grammar
# 2. will auxiliary verb tense
# 3. English verb tenses examples
# 4. identifying future tense sentences

# Example 3:
# Question: Is this a physical or chemical change? Burning wood.
# Choices: ["physical change", "chemical change"]
# Queries:
# 1. burning wood chemical reaction
# 2. physical vs chemical change examples
# 3. combustion change type
# 4. irreversible changes chemistry

# Now generate queries:
# Question: {question}
# Choices: {choices}
# {image_context}
# Queries:
# 1."""


# Prompt template (v3 — fully generic, no question-metadata slots)
# Design principles:
#
#   1. ONLY question + choices + image-context are in the live prompt.
#      Hint, lecture, subject, topic live in the KB; the queries must be
#      phrased to surface them naturally.
#
#   2. Few-shot examples demonstrate THREE query families that together cover
#      a typical KB:
#        (a) Definitional / conceptual  -> retrieves lecture-style chunks
#        (b) Choice-specific            -> retrieves discriminating evidence
#        (c) Comparative / relational   -> retrieves ranking / comparison info
#
#   3. No metadata (Subject, Hint, Lecture) appears in the examples, so the
#      model learns to produce good queries without privileged context.

PLANNER_PROMPT_TEMPLATE = """Generate search queries to retrieve the evidence needed to answer a multiple-choice question.
Your queries will be run against a knowledge base AND a web search engine.

Good queries retrieve:
  - definitions and explanations of key concepts in the question
  - facts specific to each answer choice that distinguish it from the others
  - background knowledge relevant to the topic being tested

RULES:
- Output 4-6 numbered queries, one per line.
- Each query: 3-12 words, plain keywords or short phrases — no question marks.
- Cover at least: one conceptual/definitional query AND one query per distinct answer choice.
- Do NOT repeat the question verbatim.

---
Example 1 — comparative / geographic:
Question: Which of these states is farthest north?
Choices: ["West Virginia", "Louisiana", "Arizona", "Oklahoma"]
Queries:
1. West Virginia latitude geographic location northern United States
2. Louisiana latitude southern United States position
3. Arizona geographic coordinates latitude north
4. Oklahoma latitude map northern states
5. US states latitude ranking northernmost comparison
6. latitude West Virginia Louisiana Arizona Oklahoma

---
Example 2 — grammar / conceptual:
Question: What tense is used in this sentence? "She will dance tomorrow."
Choices: ["past tense", "present tense", "future tense"]
Queries:
1. future tense definition will auxiliary verb grammar
2. past tense definition examples English grammar
3. present tense definition form examples
4. identify verb tense in a sentence
5. will auxiliary verb future tense indicator

---
Example 3 — physical science / classification:
Question: Is a scarf a solid or a liquid?
Choices: ["a solid", "a liquid"]
Queries:
1. solid state matter definition definite shape volume
2. liquid state matter definition flow takes container shape
3. scarf fabric material solid liquid classification
4. states of matter everyday objects examples classify
5. solid vs liquid properties differences comparison

---
Example 4 — language / capitalization rules:
Question: Which correctly shows the title of a play?
Choices: ["A breath of Fresh Air", "A Breath of Fresh Air"]
Queries:
1. title case capitalization rules books plays movies
2. capitalize major words in a title grammar rule
3. title capitalization which words capitalized English
4. correct title formatting capitalize each word
5. title capitalization articles prepositions rules

---
Now generate queries for this question.
Question: {question}
Choices: {choices}
{image_context}Queries:
1."""

# def extract_key_terms(question: str, choices: List[str]) -> List[str]:
#     """
#     Extract important terms from question and choices
#     These should be included in search queries
#     """
#     key_terms = set()

#     # Extract capitalized terms (likely proper nouns or important concepts)
#     capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
#     key_terms.update(capitalized)

#     # Extract numbers and units
#     numbers_units = re.findall(
#         r"\d+(?:\.\d+)?\s*(?:km|m|cm|kg|g|°C|°F|mph|%)?", question
#     )
#     key_terms.update(numbers_units)

#     # Extract quoted terms
#     quoted = re.findall(r'"([^"]+)"', question)
#     key_terms.update(quoted)

#     # Extract important words from choices (nouns, likely)
#     for choice in choices:
#         # Get significant words (>3 chars, not common words)
#         words = [
#             w
#             for w in choice.split()
#             if len(w) > 3
#             and w.lower()
#             not in {
#                 "that",
#                 "this",
#                 "with",
#                 "from",
#                 "have",
#                 "been",
#                 "were",
#                 "what",
#                 "when",
#             }
#         ]
#         key_terms.update(words[:2])  # Take first 2 significant words per choice

#     return list(key_terms)

def parse_planner_output(raw_output: str, question: str, choices: List[str]) -> List[str]:
    """
    Parse planner output with comprehensive filtering.
    """
    raw_lines = raw_output.split("\n")
    clean = []
    
    # Patterns to skip entirely
    SKIP_PATTERNS = [
        'here are', 'following', 'search quer', 'to answer', 'these quer', 
        'i generate', 'based on', 'now generate', 'output format', 'example for',
        'each query should', 'generate exactly', 'be suitable', 'focus on one',
        'queries:', 'pattern:', 'choices:', 'question:', 'answer choices:',
        'image caption', 'critical rule', 'generate 3-4', 'generate 4',
        # Add more patterns that might leak from prompt
        'not repeat', 'one per line', 'be 3-8 words'
    ]
    
    # Get first 20 chars of question for comparison (used to skip parroted lines)
    question_start = question.lower()[:20].strip()
    
    for line in raw_lines:
        # Basic cleanup
        line = line.strip()
        
        # Remove numbering: "1. ", "1) ", "- ", "• ", "* "
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        line = line.strip()
        
        # Remove surrounding quotes and brackets
        line = line.strip('"\'[]')
        line = line.strip()
        
        # REJECTION FILTERS
        
        # Skip empty or too short/long
        if len(line) < 6 or len(line) > 150:
            continue
        
        # Skip template placeholders like "[query 1]", "[query 2]", "query 1"
        if re.match(r'^\[?query\s*\d+\]?$', line, re.IGNORECASE):
            continue
        
        # Skip if line IS a template placeholder pattern
        if re.match(r'^\[.*\]$', line) and len(line) < 15:
            continue
        
        # Skip meta-text patterns
        line_lower = line.lower()
        if any(p in line_lower for p in SKIP_PATTERNS):
            continue
        
        # Skip if it starts like the question (first 20 chars)
        if len(question_start) > 10 and line_lower.startswith(question_start):
            continue
        
        # Skip if it contains a JSON-like choices list
        if re.search(r'\[[\'\"].*[\'\"],', line_lower):
            continue
        
        # Skip lines that are just the question repeated
        if line_lower == question.lower().strip():
            continue
        
        # Skip lines that look like instruction remnants
        if line_lower.startswith(('note:', 'example:', 'output:', 'format:', 'rule:')):
            continue
        
        # VALIDATION
        
        # Word count check: 2-15 words
        words = line.split()
        if len(words) < 2 or len(words) > 15:
            continue
        
        clean.append(line)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in clean:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)
    
    return unique[:4]


def generate_fallback_queries(question: str, choices: List[str]) -> List[str]:
    """
    Generate fallback queries when parsing fails.
    """
    queries = []
    
    # Stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
        'how', 'this', 'that', 'does', 'do', 'will', 'can', 'be', 'been',
        'complete', 'sentence', 'following', 'use', 'uses', 'using',
        'example', 'does', 'have', 'has', 'had', 'would', 'could', 'should'
    }
    
    # Extract key terms from question
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    key_terms = [w for w in words if w not in stop_words][:5]
    
    # Query from question terms
    if len(key_terms) >= 2:
        queries.append(f"{' '.join(key_terms[:3])} definition")
    
    # Queries from choices (skip yes/no)
    for choice in choices[:3]:
        choice_clean = str(choice).strip().lower()
        if len(choice_clean) > 3 and choice_clean not in ['yes', 'no', 'true', 'false']:
            queries.append(f"{choice_clean} definition")
            if len(queries) < 4:
                queries.append(f"{choice_clean} examples")
    
    return queries[:4]


def planner_step(state: State, model, tokenizer, kwargs) -> State:
    with tracer.start_as_current_span(
        "Planner", openinference_span_kind="chain"
    ) as planner_span:
        # Save current queries to history before generating new ones
        if state.subqueries:
            state.query_history.append(state.subqueries.copy())

        # Build image context
        image_context = ""
        if state.img_captions or state.img_ocr:
            image_context_parts = []
            if state.img_captions:
                captions_text = "\n".join(
                    [f"{k}: {v}" for k, v in state.img_captions.items()]
                )
                image_context_parts.append(f"Image: {captions_text}")
            if state.img_ocr:
                ocr_text = "\n".join([f"{k}: {v}" for k, v in state.img_ocr.items()])
                image_context_parts.append(f"OCR: {ocr_text}")
            image_context = "\n".join(image_context_parts)

        # Build prompt - few-shot already includes instructions
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            question=state.question,
            image_context=image_context,
            choices=state.choices,
        )

        # Add feedback ONLY if this is a retry
        # On retries the verifier_feedback contains specific guidance (e.g. which claims
        # were hallucinated or unsupported). Incorporate it into the prompt BEFORE the
        # numbered list so the model has context when generating new queries.
        # Previously this block appended a second "1." after the template's "1.", creating
        # a malformed prompt, and it used generic key-term extraction instead of the
        # actual verifier feedback.
        if state.verifier_feedback and state.retry_count > 0:
            key_terms = extract_key_terms(state.question, state.choices)
            feedback_note = (
                f"Note: Previous search queries were insufficient. "
                f"Verifier feedback: {state.verifier_feedback.strip()} "
                f"Also consider searching for: {', '.join(key_terms[:5])}"
            )
            # Insert the note before the numbered list (which the template already opened with "1.")
            prompt = prompt.rstrip()
            # Remove the trailing "1." that the template appended, prepend the note, then restore it
            if prompt.endswith("1."):
                prompt = prompt[:-2].rstrip() + "\n" + feedback_note + "\n1."
            else:
                prompt = prompt + "\n" + feedback_note + "\n1."
        # else: prompt already ends with "1." from template

        planner_span.set_attribute("planner.has_feedback", bool(state.verifier_feedback))
        planner_span.set_attribute("planner.retry_attempt", state.retry_count)

        messages = [
            {"role": "user", "content": prompt},  # No system message needed
        ]

        planner_span.set_attribute("planner.question", state.question)
        planner_span.set_attribute("planner.choices", json.dumps(state.choices))

        with tracer.start_as_current_span(
            "Qwen2.5-7B", openinference_span_kind="llm"
        ) as llm_span:
            llm_span.set_attribute("llm.model_name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            llm_span.set_attribute("llm.input_messages", str(messages))

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **kwargs)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract only the generated part (after the prompt)
            if "Queries:" in decoded:
                decoded = decoded.split("Queries:")[-1]
            elif "1." in decoded:
                # Find where our completion starts
                decoded = "1." + decoded.split("1.")[-1]

            llm_span.set_attribute("llm.token_count.prompt", inputs["input_ids"].shape[1])
            llm_span.set_attribute("llm.token_count.completion", outputs.shape[1] - inputs["input_ids"].shape[1])
            llm_span.set_attribute("llm.output", decoded)

        # Parse output
        state.subqueries = parse_planner_output(decoded, state.question, state.choices)
        
        # Fallback if not enough valid queries
        if len(state.subqueries) < 2:
            print(f"Only {len(state.subqueries)} valid queries, adding fallbacks")
            fallback_queries = generate_fallback_queries(state.question, state.choices)
            combined = state.subqueries + fallback_queries
            seen = set()
            unique = []
            for q in combined:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique.append(q)
            state.subqueries = unique[:4]
        
        # Debug logging
        print(f"Planner output for PID {state.pid}:")
        for i, q in enumerate(state.subqueries):
            print(f"   {i+1}. {q}")

        # Planner Metrics
        planner_span.set_attribute("planner.subquery_count", len(state.subqueries))
        planner_span.set_attribute("planner.subqueries", json.dumps(state.subqueries))

        coverage = planner_coverage_score(state)
        planner_span.set_attribute("planner.coverage_score", coverage)

        specificity = planner_specificity_score(state)
        planner_span.set_attribute("planner.specificity_score", specificity)

        quality = (coverage + specificity) / 2
        planner_span.set_attribute("planner.quality_score", quality)

    return state