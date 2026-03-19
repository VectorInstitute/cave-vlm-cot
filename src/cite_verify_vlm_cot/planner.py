"""
This is planner.py — the first node in the pipeline. 
Its job is to turn a raw multiple-choice question into a list of search queries that the retriever will use to find evidence.
Input: a State object containing the question, answer choices, and optionally image captions/OCR text.
Output: state.subqueries — a list of up to 8 search query strings.
"""

# import ast
import json
# import math
from typing import Dict, List, Optional

# import pandas as pd
import torch
from evaluations import planner_coverage_score, planner_specificity_score
import re

# CLIPProcessor handles image pre-processing like resizing and normalization
from tracer import tracer
from utils import State

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


# Prompt template
# The few-shot examples are in JSON format so the model learns the required
# output structure from the examples themselves.

# Design principles:
#   1. ONLY question + choices + image-context in the live prompt.
#      Hint, lecture, subject, topic live in the KB and are surfaced via
#      well-phrased queries, not privileged context injection.
#   2. Three query families per example: definitional, choice-specific,
#      comparative — these cover the main KB document types.
#   3. No metadata (Subject, Hint, Lecture) in the examples.

PLANNER_PROMPT_TEMPLATE = """Generate search queries to retrieve the evidence needed to answer a multiple-choice question.
Your queries will be run against a knowledge base AND a web search engine.

Good queries retrieve:
  - definitions and explanations of key concepts in the question
  - facts specific to each answer choice that distinguish it from the others
  - background knowledge relevant to the topic being tested

OUTPUT FORMAT: a JSON array of query strings — nothing else.
Each query: 3-12 words, plain keywords or short phrases, no question marks.
Cover at least one conceptual/definitional query AND one query per distinct answer choice.

---
Example 1 — comparative / geographic:
Question: Which of these states is farthest north?
Choices: ["West Virginia", "Louisiana", "Arizona", "Oklahoma"]
[
  "West Virginia latitude geographic location northern United States",
  "Louisiana latitude southern United States position",
  "Arizona geographic coordinates latitude north",
  "Oklahoma latitude map northern states",
  "US states latitude ranking northernmost comparison",
  "latitude West Virginia Louisiana Arizona Oklahoma"
]

---
Example 2 — grammar / conceptual:
Question: What tense is used in this sentence? "She will dance tomorrow."
Choices: ["past tense", "present tense", "future tense"]
[
  "future tense definition will auxiliary verb grammar",
  "past tense definition examples English grammar",
  "present tense definition form examples",
  "identify verb tense in a sentence",
  "will auxiliary verb future tense indicator"
]

---
Example 3 — physical science / classification:
Question: Is a scarf a solid or a liquid?
Choices: ["a solid", "a liquid"]
[
  "solid state matter definition definite shape volume",
  "liquid state matter definition flow takes container shape",
  "scarf fabric material solid liquid classification",
  "states of matter everyday objects examples classify",
  "solid vs liquid properties differences comparison"
]

---
Example 4 — language / capitalization:
Question: Which correctly shows the title of a play?
Choices: ["A breath of Fresh Air", "A Breath of Fresh Air"]
[
  "title case capitalization rules books plays movies",
  "capitalize major words in a title grammar rule",
  "title capitalization which words capitalized English",
  "correct title formatting capitalize each word",
  "title capitalization articles prepositions rules"
]

---
Now generate queries for this question.
Question: {question}
Choices: {choices}
{image_context}["""

# JSON parsing + structural validation
def _is_valid_query(candidate: str, question: str) -> bool:
    """
    Structural validity check — no vocabulary patterns.

    A valid search query must:
      - be a non-empty string
      - have between 2 and 12 words
      - have between 6 and 150 characters
      - contain at least one alphabetic word of >= 3 characters
      - not be identical to the question (case-insensitive)
      - not end with a colon (structural marker for labels/headers, not queries)
    """
    if not isinstance(candidate, str):
        return False

    candidate = candidate.strip()

    if not (6 <= len(candidate) <= 150):
        return False

    # Labels and headers end with ":" (e.g. "Note:", "Here are the queries:")
    if candidate.endswith(":"):
        return False

    words = candidate.split()
    if not (2 <= len(words) <= 12):
        return False

    # Must contain at least one content-bearing alphabetic token
    if not any(re.search(r'[a-zA-Z]{3,}', w) for w in words):
        return False

    # Must not be a verbatim copy of the question
    if candidate.lower() == question.lower().strip():
        return False

    return True


def _extract_json_array(raw: str) -> Optional[list]:
    """
    Extract query strings from raw LLM output. Three strategies in order:

      1. Full JSON parse  — find "[...]" and parse with json.loads.
      2. Partial recovery — truncated array; salvage quoted strings via regex.
      3. Line extraction  — no JSON at all; treat stripped non-empty lines
                            as candidates (covers models that ignore the format
                            instruction and emit a numbered list instead).
    Strategy 3 is the last resort and is only reached if the model completely
    ignores the JSON format instruction. The prompt ends with "[" to make
    Strategy 1 the common path.
    """
    start = raw.find("[")

    if start != -1:
        # Strategy 1: walk to matching "]" and parse
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break

        # Strategy 2: truncated — salvage quoted strings from the fragment
        fragment = raw[start:]
        items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fragment)
        if items:
            return items

    # Strategy 3: no JSON found — split into lines, strip numbering/bullets,
    # return non-empty lines as raw string candidates.
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = line.lstrip("[")          # remove the generation-prefix "[" if present
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        line = line.strip().strip('"\'')
        if line:
            lines.append(line)
    return lines if lines else None

def parse_planner_output(raw_output: str, question: str) -> List[str]:
    """
    Parse LLM output into a clean, deduplicated list of search queries.

    Primary path: JSON array extraction + structural validation.
    No vocabulary-based filtering anywhere.
    """
    candidates = _extract_json_array(raw_output)

    if not candidates:
        return []

    seen: set = set()
    valid: List[str] = []
    for item in candidates:
        item = str(item).strip().strip('"').strip()
        if _is_valid_query(item, question) and item.lower() not in seen:
            seen.add(item.lower())
            valid.append(item)

    return valid[:6]


# Choice-discriminating query generator (deterministic, KB-agnostic)
_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
    'how', 'this', 'that', 'does', 'do', 'will', 'can', 'be', 'been',
    'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
}


def generate_choice_queries(question: str, choices: List[str]) -> List[str]:
    """
    Generate one targeted retrieval query per answer choice.

    Built from question text and choice labels only — no metadata.
    Queries are phrased to match KB documents and web snippets about each
    specific choice, enabling discrimination between options.
    """
    q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question)]
    anchor_words = [w for w in q_words if w not in _STOP_WORDS][:3]
    anchor = " ".join(anchor_words)

    queries = []
    for choice in choices:
        choice_str = str(choice).strip()
        if not choice_str or choice_str.lower() in ('yes', 'no', 'true', 'false'):
            if anchor:
                queries.append(f"{anchor} {choice_str}"[:80])
            continue

        choice_words = choice_str.split()
        if len(choice_words) <= 3:
            queries.append(f"{choice_str} definition properties characteristics"[:80])
        else:
            queries.append(" ".join(choice_words[:6])[:80])

    return [q for q in queries if _is_valid_query(q, question)]

# Retry: extract query directions from hallucination details
def queries_from_hallucination_feedback(
    hallucination_details: List[Dict],
    question: str,
    choices: List[str],
) -> List[str]:
    """
    Convert verifier hallucination details into new search query directions.
    No vocabulary patterns — works purely from the structured hallucination report.
    """
    queries = []

    for detail in hallucination_details[:4]:
        claim = detail.get('claim', '')
        issue = detail.get('issue', '').lower()
        evidence = detail.get('evidence', '')

        # Strip citation markers to get the raw claim text
        clean_claim = re.sub(
            r'\[(?:Text Evidence|Question Image)\s*\d+\]', '', claim
        ).strip().strip('"\'')

        if 'fake citation' in issue or 'not in evidence' in issue:
            words = [w for w in clean_claim.lower().split()
                     if w not in _STOP_WORDS and len(w) > 3]
            if words:
                queries.append(" ".join(words[:6]))

        elif 'misrepresented' in issue or 'fabricated' in issue:
            if evidence and evidence.lower() not in ("doesn't exist", "unknown"):
                words = [w for w in evidence.lower().split()
                         if w not in _STOP_WORDS and len(w) > 3]
                if words:
                    queries.append(" ".join(words[:6]))
            words = [w for w in clean_claim.lower().split()
                     if w not in _STOP_WORDS and len(w) > 3]
            if words:
                queries.append(" ".join(words[:5]) + " examples")

    return [q for q in queries if _is_valid_query(q, question)][:4]

# Fallback: generate queries when LLM parsing yields nothing usable
def generate_fallback_queries(question: str, choices: List[str]) -> List[str]:
    """
    Deterministic fallback when the LLM produces no parseable output.
    Uses question keywords and choice labels to construct minimal queries.
    """
    queries = []

    q_words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    key_terms = [w for w in q_words if w not in _STOP_WORDS][:5]

    if len(key_terms) >= 2:
        queries.append(f"{' '.join(key_terms[:3])} definition")

    for choice in choices[:4]:
        choice_clean = str(choice).strip().lower()
        if len(choice_clean) > 3 and choice_clean not in ('yes', 'no', 'true', 'false'):
            queries.append(f"{choice_clean} definition")
            if len(queries) < 6:
                queries.append(f"{choice_clean} examples")

    return [q for q in queries if _is_valid_query(q, question)][:6]


# Main planner step
def planner_step(state: State, model, tokenizer, kwargs) -> State:
    with tracer.start_as_current_span(
        "Planner", openinference_span_kind="chain"
    ) as planner_span:

        if state.subqueries:
            state.query_history.append(state.subqueries.copy())

        # Image context (derived from the image itself, not from metadata)
        image_context = ""
        if state.img_captions or state.img_ocr:
            parts = []
            if state.img_captions:
                parts.append("Image: " + " | ".join(
                    f"{k}: {v}" for k, v in state.img_captions.items()
                ))
            if state.img_ocr:
                parts.append("OCR: " + " | ".join(
                    f"{k}: {v}" for k, v in state.img_ocr.items()
                ))
            image_context = "\n".join(parts) + "\n"

        # Build prompt — ends with "[" to prime JSON array generation.
        # Hint / lecture / subject / topic are NOT passed here.
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            question=state.question,
            choices=state.choices,
            image_context=image_context,
        )

        # On retry: insert verifier feedback between the question block and
        # the opening "[".  The model still outputs a JSON array.
        if state.verifier_feedback and state.retry_count > 0:
            hd = state.hallucination_details or []
            feedback_queries = queries_from_hallucination_feedback(
                hd, state.question, state.choices
            )
            feedback_parts = [
                f"NOTE: Attempt {state.retry_count} was REJECTED ({state.hallucination}).",
                f"Verifier feedback: {state.verifier_feedback.strip()[:400]}",
            ]
            if feedback_queries:
                feedback_parts.append(
                    "Suggested query directions: " + "; ".join(feedback_queries)
                )
            if state.subqueries:
                failed_str = ", ".join(f'"{q}"' for q in state.subqueries[:4])
                feedback_parts.append(
                    f"Do not repeat these ineffective queries: {failed_str}"
                )
            feedback_note = "\n".join(feedback_parts)

            # Insert feedback between the last line of the template and the
            # opening "[" that starts the JSON array.
            if prompt.rstrip().endswith("["):
                prompt = prompt.rstrip()[:-1].rstrip() + "\n" + feedback_note + "\n["
            else:
                prompt = prompt.rstrip() + "\n" + feedback_note + "\n"

        planner_span.set_attribute("planner.has_feedback", bool(state.verifier_feedback))
        planner_span.set_attribute("planner.retry_attempt", state.retry_count)
        planner_span.set_attribute("planner.question", state.question)
        planner_span.set_attribute("planner.choices", json.dumps(state.choices))

        messages = [{"role": "user", "content": prompt}]

        # LLM inference
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

            # The model continues from "[" — prepend it so the parser sees a
            # complete JSON array regardless of whether the model echoes the
            # opening bracket.
            raw_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract the assistant turn (everything after the prompt)
            if "\nassistant\n" in raw_decoded:
                decoded = raw_decoded.split("\nassistant\n")[-1].strip()
            elif "assistant" in raw_decoded.lower():
                decoded = raw_decoded.split("assistant")[-1].strip()
            else:
                decoded = raw_decoded

            # The prompt ended with "["; prepend it so the parser sees a
            # complete array whether or not the model echoes the bracket.
            if not decoded.lstrip().startswith("["):
                decoded = "[" + decoded

            llm_span.set_attribute("llm.token_count.prompt", inputs["input_ids"].shape[1])
            llm_span.set_attribute(
                "llm.token_count.completion",
                outputs.shape[1] - inputs["input_ids"].shape[1],
            )
            llm_span.set_attribute("llm.output", decoded)

        # Parse: JSON array → structural validation → deduplication
        state.subqueries = parse_planner_output(decoded, state.question)

        # Augment with deterministic choice-discriminating queries
        choice_queries = generate_choice_queries(state.question, state.choices)

        seen: set = {q.lower() for q in state.subqueries}
        for cq in choice_queries:
            if cq.lower() not in seen and len(state.subqueries) < 8:
                seen.add(cq.lower())
                state.subqueries.append(cq)

        # Fallback if JSON parse produced nothing usable
        if len(state.subqueries) < 2:
            print(f"[Planner] Only {len(state.subqueries)} valid queries — using fallback")
            for fq in generate_fallback_queries(state.question, state.choices):
                if fq.lower() not in seen and len(state.subqueries) < 8:
                    seen.add(fq.lower())
                    state.subqueries.append(fq)

        state.subqueries = state.subqueries[:8]

        print(f"[Planner] PID {state.pid} — {len(state.subqueries)} queries:")
        for i, q in enumerate(state.subqueries):
            print(f"   {i + 1}. {q}")

        # Metrics
        planner_span.set_attribute("planner.subquery_count", len(state.subqueries))
        planner_span.set_attribute("planner.subqueries", json.dumps(state.subqueries))

        coverage = planner_coverage_score(state)
        planner_span.set_attribute("planner.coverage_score", coverage)

        specificity = planner_specificity_score(state)
        planner_span.set_attribute("planner.specificity_score", specificity)

        planner_span.set_attribute("planner.quality_score", (coverage + specificity) / 2)

    return state