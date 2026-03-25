"""
planner.py
----------
First node in the pipeline. Turns a raw multiple-choice question into a list
of search queries for the retriever.
Input:  State (question, choices, optional image captions/OCR)
Output: state.subqueries — up to 8 search query strings
"""
import json
from typing import Dict, List, Optional
import torch
from evaluations import planner_coverage_score, planner_specificity_score
import re

# CLIPProcessor handles image pre-processing like resizing and normalization
from tracer import tracer
from utils import State
from prompts import PLANNER_PROMPT_TEMPLATE

# Enable only the safe fallback backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# SDPA = Scaled Dot-Product Attention, core math behind attention
# PyTorch provides multiple implementations of this, called backends (FlashAttention, SDPA(default), Math)
# SDPA is fast but allocates large temporary buffers, especially bad for long sequences + big models
# Prefer FlashAttention (best) if your GPU supports it

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
    # current set
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
    'how', 'this', 'that', 'does', 'do', 'will', 'can', 'be', 'been',
    'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
    # additions that actually appear in ScienceQA question stems
    'has', 'have', 'had', 'not', 'from', 'with', 'would', 'could',
    'should', 'may', 'when', 'where', 'who', 'why', 'if', 'it',
    'its', 'they', 'them', 'their', 'most', 'some', 'each',
    'identify', 'following', 'best', 'describes', 'called',
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
    queries = []

    # Question-level anchor query (not produced by generate_choice_queries)
    q_words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    key_terms = [w for w in q_words if w not in _STOP_WORDS][:5]
    if len(key_terms) >= 2:
        queries.append(f"{' '.join(key_terms[:3])} definition")

    # Reuse choice query logic rather than duplicating it
    choice_queries = generate_choice_queries(question, choices)
    seen = {q.lower() for q in queries}
    for q in choice_queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

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