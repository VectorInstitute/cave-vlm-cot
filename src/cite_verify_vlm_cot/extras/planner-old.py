import ast
import json
import math
from typing import Dict, List, Optional

import pandas as pd
import torch
from evaluations import planner_coverage_score, planner_specificity_score
from pydantic import BaseModel
import re

# CLIPProcessor handles image pre-processing like resizing and normalization
from tracer import tracer


# Unsloth (efficient Llama inference)
# import unsloth
# from unsloth import FastLanguageModel


# Enable only the safe fallback backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# SDPA = Scaled Dot-Product Attention, core math behind attention
# PyTorch provides multiple implementations of this, called backends (FlashAttention, SDPA(default), Math)
# SDPA is fast but allocates large temporary buffers, especially bad for long sequences + big models
# Prefer FlashAttention (best) if your GPU supports it

# Phoenix is an application that can receive the traces that you're going to send from your agent here and then can visualize those in a UI
# import phoenix as px
# session = px.launch_app()
# px_client = px.Client()

# # Add Phoenix API Key for tracing
# os.environ["PHOENIX_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.z7I18JzH4dsxKikCBGg9deP05APuLC37t8Z0euNAy3I"
# os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/srao0996"

# # If you created your Phoenix Cloud instance before June 24th, 2025,
# # you also need to set the API key as a header
# # os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

# PROJECT_NAME = "cite-and-verify-vlm-cot-agent"
# tracer_provider = register(
#     project_name=PROJECT_NAME,
#     endpoint= "https://app.phoenix.arize.com/s/srao0996/v1/traces",
#     headers={
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.z7I18JzH4dsxKikCBGg9deP05APuLC37t8Z0euNAy3I"
#     }
# )
# OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)
# tracer = tracer_provider.get_tracer(__name__)


# Define RoiInfo with all fields used in retriever
class RoiInfo(BaseModel):
    roi_id: str
    bbox: List[int]  
    source_image: str  
    image_patch: str
    caption: str
    score: float
    subject: str = "" 
    topic: str = ""

class ChunkInfo(BaseModel):
    text_chunks: List[str] = []
    image_rois: List[RoiInfo] = []

class EvidenceChunk(BaseModel):
    evidence_id: str       # e.g. 'Text Evidence 1'
    text: str

class EvidenceROI(BaseModel):
    evidence_id: str       # e.g. 'Image ROI 1'
    roi_id: str
    bbox: List[int]
    source_image: str
    image_patch: str
    caption: str
    score: float

class State(BaseModel):
    pid: str
    question: str
    hint: Optional[str] = ""
    image_paths: Optional[List[str]] = []
    lecture: Optional[str] = ""
    choices: List[str]
    answer: int
    gold_answer: str = ""

    img_captions: Optional[Dict[str, str]] = {}
    img_ocr: Optional[Dict[str, str]] = {}

    # Planner output
    subqueries: Optional[List[str]] = []
    query_history: List[List[str]] = []  # Track all query attempts

    # Retriever output
    retrieved_chunks: Dict[str, ChunkInfo] = {}

    # Solver output
    reasoning_steps: Optional[List[str]] = []
    final_answer: Optional[str] = None
    # retrieved_evidence_with_citations: Optional[str] = None

    # Verifier output
    verdict: Optional[str] = None
    verifier_answer: Optional[str] = "INCONCLUSIVE"
    confidence: Optional[str] = "LOW"
    hallucination: Optional[str] = "UNKNOWN"
    hallucination_details: List[Dict[str, str]] = []  # Structured hallucination info

    # Feedback loop
    verifier_feedback: Optional[str] = None  # Feedback from verifier to planner
    retry_count: int = 0  # Track number of retries
    attempt_history: List[Dict] = []


def safe_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def safe_parse_json(x, default=None):
    """Safely parse JSON or Python literal string."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return default if default is not None else {}

    if isinstance(x, (dict, list)):
        return x

    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return default if default is not None else {}


df = pd.read_csv("scienceqa_augmented_100.csv")
states = []

for idx, row in df.iterrows():
    pid = safe_str(row.get("pid"))
    question = safe_str(row.get("question"))
    hint = safe_str(row.get("hint"))
    lecture = safe_str(row.get("lecture"))

    # Parse image_paths (JSON array)
    image_paths = safe_parse_json(row.get("image_paths"), default=[])
    if not isinstance(image_paths, list):
        image_paths = []

    # Parse choices (JSON array or string representation)
    choices_raw = row.get("choices")
    if isinstance(choices_raw, str):
        choices = safe_parse_json(choices_raw, default=[])
    else:
        choices = choices_raw if choices_raw else []

    # Parse answer
    answer = int(row["answer"]) if pd.notna(row.get("answer")) else -1

    # Parse img_captions and img_ocr (JSON objects)
    img_captions = safe_parse_json(row.get("img_captions"), default={})
    img_ocr = safe_parse_json(row.get("img_ocr"), default={})

    # Build image context from captions and OCR
    image_context_parts = []
    if img_captions:
        captions_text = "\n".join([f"{k}: {v}" for k, v in img_captions.items()])
        image_context_parts.append(f"Image Captions:\n{captions_text}")

    if img_ocr:
        ocr_text = "\n".join([f"{k}: {v}" for k, v in img_ocr.items()])
        image_context_parts.append(f"OCR Text:\n{ocr_text}")

    image_context = "\n\n".join(image_context_parts) if image_context_parts else ""

    # Determine gold answer
    gold_answer = choices[answer] if 0 <= answer < len(choices) else ""

    state = State(
        pid=pid,
        question=question,
        hint=hint,
        image_paths=image_paths,
        lecture=lecture,
        choices=choices,
        answer=answer,
        img_captions=img_captions,
        img_ocr=img_ocr,
        subqueries=[],
        retrieved_chunks={},
        reasoning_steps=[],
        final_answer="",
        verdict="",
        gold_answer=gold_answer,
    )

    states.append(state)

print(f"Loaded {len(states)} states from CSV")

# Example: Print first state
if states:
    print("\nFirst state:")
    print(f"  PID: {states[0].pid}")
    print(f"  Question: {states[0].question[:100]}...")
    print(f"  Image paths: {states[0].image_paths}")
    print(f"  Choices: {states[0].choices}")
    print(f"  Gold answer: {states[0].gold_answer}")

# PLANNER_PROMPT_TEMPLATE = """\
#   Question: {question}
#   Hint: {hint}
#   {image_context}
#   Lecture: {lecture}
#   Choices: {choices}
#   Task: Decompose the above into a chain of reasoning steps (sub-questions) i.e. Break down the question into 2–3 sub-queries or keywords useful for retrieval.
# """

# PLANNER_PROMPT_TEMPLATE = """\
# You are a retrieval planner. Produce 2–3 targeted sub-queries that a search engine or retriever could use.

# Context:
# Question: {question}
# Hint: {hint}
# {image_context}
# Lecture: {lecture}
# Choices: {choices}

# Constraints:
# - Do NOT solve the problem.
# - Each line must be a query fragment (keywords or a short question).
# - Include key nouns from the choices when relevant.
# - Prefer geography/science terms, units, definitions, or comparisons.
# - 2–3 lines only.

# Return ONLY the lines, no extra text.
# """


# PLANNER_PROMPT_TEMPLATE = """\
# You are generating search queries to answer a question.

# Question: {question}
# Answer Choices: {choices}

# {image_context}

# CRITICAL RULES:
# 1. Include terms from CHOICES in your queries
# 2. Extract SPECIFIC details from hints (names, numbers, objects)
# 3. DO NOT generate generic "definition of X" queries
# 4. Generate queries that could directly retrieve the answer
# 5. Avoid restating the full question
# 6. If image captions are provided, use specific visual terms from them in queries

# Generate 3-4 specific search queries:

# Follow these examples:

# ---
# Question: What is an example of a physical change?
# Choices: ["cutting paper", "burning wood", "rusting iron"]
# → Subqueries:
# - definition of physical change
# - is cutting paper a physical change
# - cutting paper vs burning

# ---
# Question: Why does wax make a snowboard faster?
# Choices: ["reduces friction", "adds weight", "changes shape"]
# → Subqueries:
# - how snowboard wax reduces friction
# - effect of wax on snowboard speed
# - snowboard wax experiment results

# ---
# Question: Which organelle is labeled X in the diagram?
# Choices: ["mitochondria", "nucleus", "ribosome"]
# Image Captions: {{diagram.png: "cell diagram with labeled organelles, arrow pointing to oval structure with inner folds"}}
# → Subqueries:
# - oval organelle inner membrane folds cell
# - mitochondria cristae structure diagram
# - cell organelle identification inner folds function
# """

# PLANNER_PROMPT_TEMPLATE = """\
# You are generating search queries to retrieve evidence for answering a question.

# Question: {question}
# Answer Choices: {choices}
# {image_context}

# CRITICAL RULES:
# 1. Query for facts about EACH answer choice individually
# 2. Query for the comparison criterion or key property
# 3. Use specific terms from choices and image captions
# 4. Keep queries short (3-7 words)
# 5. Focus on retrieving facts, not solving the problem

# Generate 3-5 search queries following these patterns:

# ---

# Pattern 1: COMPARISON QUESTIONS
# Question: Which of these states is farthest north?
# Choices: ["West Virginia", "Louisiana", "Arizona", "Oklahoma"]

# Queries:
# - West Virginia latitude coordinates
# - Louisiana latitude coordinates
# - Arizona latitude coordinates
# - Oklahoma latitude coordinates

# ---

# Pattern 2: PROPERTY IDENTIFICATION
# Question: What is an example of a physical change?
# Choices: ["cutting paper", "burning wood", "rusting iron"]

# Queries:
# - physical change characteristics
# - cutting paper reversible change
# - burning wood chemical reaction
# - rusting iron oxidation process

# ---

# Pattern 3: VISUAL IDENTIFICATION
# Question: Which organelle is labeled X in the diagram?
# Choices: ["mitochondria", "nucleus", "ribosome"]
# Image Captions: {{diagram.png: "cell diagram with labeled organelles, arrow pointing to oval structure with inner folds"}}

# Queries:
# - oval organelle inner membrane folds
# - mitochondria structure cristae folds
# - nucleus structure appearance cell
# - ribosome size shape cell

# ---

# Pattern 4: MECHANISM/CAUSATION
# Question: Why does wax make a snowboard faster?
# Choices: ["reduces friction", "adds weight", "changes shape"]

# Queries:
# - snowboard wax friction reduction
# - wax coating surface properties
# - friction effect speed movement
# - snowboard base wax purpose

# ---

# Now generate queries for this question:
# """

# PLANNER_PROMPT_TEMPLATE = """Generate search queries to answer this question.

# Question: {question}
# Choices: {choices}
# {image_context}

# Generate exactly 4 search queries. Each query should:
# - Be 3-8 words long
# - Focus on ONE concept
# - Be suitable for a search engine
# - NOT repeat the question verbatim

# Output format - just the queries, one per line:
# [query 1]
# [query 2]
# [query 3]
# [query 4]

# Example for "What type of rock isite?" with choices ["ignite", "sedimentary", "metamorphic"]:
# ignite rock definition
# sedimentary rock characteristics
# metamorphic rock formation process
# rock type classification geology

# Now generate queries for the question above:"""

# PLANNER_PROMPT_TEMPLATE = """\
#   Question: {question}
#   Hint: {hint}
#   {image_context}
#   Lecture: {lecture}
#   Choices: {choices}
#   Task: Decompose the above into a chain of reasoning steps (sub-questions) i.e. Break down the question into 2–3 sub-queries or keywords useful for retrieval.
# """

# PLANNER_PROMPT_TEMPLATE = """\
# You are a retrieval planner. Produce 2–3 targeted sub-queries that a search engine or retriever could use.

# Context:
# Question: {question}
# Hint: {hint}
# {image_context}
# Lecture: {lecture}
# Choices: {choices}

# Constraints:
# - Do NOT solve the problem.
# - Each line must be a query fragment (keywords or a short question).
# - Include key nouns from the choices when relevant.
# - Prefer geography/science terms, units, definitions, or comparisons.
# - 2–3 lines only.

# Return ONLY the lines, no extra text.
# """


# PLANNER_PROMPT_TEMPLATE = """\
# You are generating search queries to answer a question.

# Question: {question}
# Answer Choices: {choices}

# {image_context}

# CRITICAL RULES:
# 1. Include terms from CHOICES in your queries
# 2. Extract SPECIFIC details from hints (names, numbers, objects)
# 3. DO NOT generate generic "definition of X" queries
# 4. Generate queries that could directly retrieve the answer
# 5. Avoid restating the full question
# 6. If image captions are provided, use specific visual terms from them in queries

# Generate 3-4 specific search queries:

# Follow these examples:

# ---
# Question: What is an example of a physical change?
# Choices: ["cutting paper", "burning wood", "rusting iron"]
# → Subqueries:
# - definition of physical change
# - is cutting paper a physical change
# - cutting paper vs burning

# ---
# Question: Why does wax make a snowboard faster?
# Choices: ["reduces friction", "adds weight", "changes shape"]
# → Subqueries:
# - how snowboard wax reduces friction
# - effect of wax on snowboard speed
# - snowboard wax experiment results

# ---
# Question: Which organelle is labeled X in the diagram?
# Choices: ["mitochondria", "nucleus", "ribosome"]
# Image Captions: {{diagram.png: "cell diagram with labeled organelles, arrow pointing to oval structure with inner folds"}}
# → Subqueries:
# - oval organelle inner membrane folds cell
# - mitochondria cristae structure diagram
# - cell organelle identification inner folds function
# """

# PLANNER_PROMPT_TEMPLATE = """\
# You are generating search queries to retrieve evidence for answering a question.

# Question: {question}
# Answer Choices: {choices}
# {image_context}

# CRITICAL RULES:
# 1. Query for facts about EACH answer choice individually
# 2. Query for the comparison criterion or key property
# 3. Use specific terms from choices and image captions
# 4. Keep queries short (3-7 words)
# 5. Focus on retrieving facts, not solving the problem

# Generate 3-5 search queries following these patterns:

# ---

# Pattern 1: COMPARISON QUESTIONS
# Question: Which of these states is farthest north?
# Choices: ["West Virginia", "Louisiana", "Arizona", "Oklahoma"]

# Queries:
# - West Virginia latitude coordinates
# - Louisiana latitude coordinates
# - Arizona latitude coordinates
# - Oklahoma latitude coordinates

# ---

# Pattern 2: PROPERTY IDENTIFICATION
# Question: What is an example of a physical change?
# Choices: ["cutting paper", "burning wood", "rusting iron"]

# Queries:
# - physical change characteristics
# - cutting paper reversible change
# - burning wood chemical reaction
# - rusting iron oxidation process

# ---

# Pattern 3: VISUAL IDENTIFICATION
# Question: Which organelle is labeled X in the diagram?
# Choices: ["mitochondria", "nucleus", "ribosome"]
# Image Captions: {{diagram.png: "cell diagram with labeled organelles, arrow pointing to oval structure with inner folds"}}

# Queries:
# - oval organelle inner membrane folds
# - mitochondria structure cristae folds
# - nucleus structure appearance cell
# - ribosome size shape cell

# ---

# Pattern 4: MECHANISM/CAUSATION
# Question: Why does wax make a snowboard faster?
# Choices: ["reduces friction", "adds weight", "changes shape"]

# Queries:
# - snowboard wax friction reduction
# - wax coating surface properties
# - friction effect speed movement
# - snowboard base wax purpose

# ---

# Now generate queries for this question:
# """

PLANNER_PROMPT_TEMPLATE = """Generate search queries to answer this question.

Question: {question}
Choices: {choices}
{image_context}

Generate exactly 4 search queries. Each query should:
- Be 3-8 words long
- Focus on ONE concept
- Be suitable for a search engine
- NOT repeat the question verbatim

Output format - just the queries, one per line:
[query 1]
[query 2]
[query 3]
[query 4]

Example for "What type of rock isite?" with choices ["ignite", "sedimentary", "metamorphic"]:
ignite rock definition
sedimentary rock characteristics
metamorphic rock formation process
rock type classification geology

Now generate queries for the question above:"""


def extract_key_terms(question: str, choices: List[str]) -> List[str]:
    """
    Extract important terms from question and choices
    These should be included in search queries
    """
    key_terms = set()

    # Extract capitalized terms (likely proper nouns or important concepts)
    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    key_terms.update(capitalized)

    # Extract numbers and units
    numbers_units = re.findall(
        r"\d+(?:\.\d+)?\s*(?:km|m|cm|kg|g|°C|°F|mph|%)?", question
    )
    key_terms.update(numbers_units)

    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', question)
    key_terms.update(quoted)

    # Extract important words from choices (nouns, likely)
    for choice in choices:
        # Get significant words (>3 chars, not common words)
        words = [
            w
            for w in choice.split()
            if len(w) > 3
            and w.lower()
            not in {
                "that",
                "this",
                "with",
                "from",
                "have",
                "been",
                "were",
                "what",
                "when",
            }
        ]
        key_terms.update(words[:2])  # Take first 2 significant words per choice

    return list(key_terms)


def format_previous_queries(state: State) -> str:
    """Format previous queries to show in feedback"""
    if state.query_history:
        # Flatten all previous queries and show last 5 unique ones
        all_queries = [q for attempt in state.query_history for q in attempt]
        unique_queries = []
        for q in reversed(all_queries):
            if q not in unique_queries:
                unique_queries.append(q)
        return "\n".join([f"- {q}" for q in reversed(unique_queries[-5:])])
    return "None"


def planner_step(state: State, model, tokenizer, kwargs) -> State:
    with tracer.start_as_current_span(
        "Planner", openinference_span_kind="chain"
    ) as planner_span:
        # Save current queries to history before generating new ones
        if state.subqueries:
            state.query_history.append(state.subqueries.copy())

        image_context = ""
        if state.img_captions or state.img_ocr:
            image_context_parts = []

            if state.img_captions:
                captions_text = "\n".join(
                    [f"{k}: {v}" for k, v in state.img_captions.items()]
                )
                image_context_parts.append(f"Image Captions:\n{captions_text}")

            if state.img_ocr:
                ocr_text = "\n".join([f"{k}: {v}" for k, v in state.img_ocr.items()])
                image_context_parts.append(f"OCR Text:\n{ocr_text}")

            image_context = (
                "\n\n".join(image_context_parts) if image_context_parts else ""
            )

        base_prompt = PLANNER_PROMPT_TEMPLATE.format(
            question=state.question,
            image_context=image_context,
            choices=state.choices,
        )

        # Add feedback if this is a retry
        if state.verifier_feedback:
            # Extract key terms for emphasis
            key_terms = extract_key_terms(state.question, state.choices)

            prompt = f"""
            {base_prompt}

            PREVIOUS ATTEMPT #{state.retry_count} FAILED

            {state.verifier_feedback}

            REQUIRED ELEMENTS in new queries:
            - Include these key terms: {", ".join(key_terms[:7])}
            - Reference specific answer choices: {", ".join(state.choices[:3])}

            AVOID repeating these previous queries (they didn't work):
            {format_previous_queries(state)}

            Generate 3-4 NEW queries that directly address the gaps above.
            Each query should be SPECIFIC and DIFFERENT from previous attempts.
            """
        else:
            prompt = f"""
            {base_prompt}

            Generate 3-4 diverse, specific search queries.
            Include terms from answer choices and key concepts from the question.
            """

        planner_span.set_attribute(
            "planner.has_feedback", bool(state.verifier_feedback)
        )
        planner_span.set_attribute("planner.retry_attempt", state.retry_count)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Log input attributes
        planner_span.set_attribute("planner.question", state.question)
        planner_span.set_attribute("planner.choices", json.dumps(state.choices))

        with tracer.start_as_current_span(
            "Llama-3", openinference_span_kind="llm"
        ) as llm_span:
            llm_span.set_attribute(
                "llm.model_name", "unsloth/llama-3-8b-Instruct-bnb-4bit"
            )
            llm_span.set_attribute("llm.input_messages", str(messages))

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **kwargs)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            decoded = decoded.split("assistant")[-1].strip()

            llm_span.set_attribute(
                "llm.token_count.prompt", inputs["input_ids"].shape[1]
            )
            llm_span.set_attribute(
                "llm.token_count.completion",
                outputs.shape[1] - inputs["input_ids"].shape[1],
            )
            llm_span.set_attribute("llm.output", decoded)

        # Parse subqueries
        # state.subqueries = [q.strip() for q in decoded.split("\n") if q.strip()]
        raw_lines = decoded.split("\n")
        clean = []
        SKIP_PATTERNS = ['here are', 'following', 'search quer', 'to answer', 'these quer', 'i generate', 'based on']
        for line in raw_lines:
            line = re.sub(r'^[\d]+\.\s*|^[-*]\s*', '', line).strip()  # strip numbering
            line = line.strip('"').strip()
            if len(line) < 6 or len(line) > 200:  # skip empty/truncated
                continue
            if any(p in line.lower() for p in SKIP_PATTERNS):
                continue
            clean.append(line)
        state.subqueries = clean[:4]  # cap at 4

        # Planner Metrics (logged on planner_span)

        # 1. Subquery count
        planner_span.set_attribute("planner.subquery_count", len(state.subqueries))
        planner_span.set_attribute("planner.subqueries", json.dumps(state.subqueries))

        # 2. Coverage score (can compute now, only needs question/choices/subqueries)
        coverage = planner_coverage_score(state)
        planner_span.set_attribute("planner.coverage_score", coverage)

        # 3. Specificity score
        specificity = planner_specificity_score(state)
        planner_span.set_attribute("planner.specificity_score", specificity)

        # 4. Quality score (heuristic combo)
        quality = (coverage + specificity) / 2
        planner_span.set_attribute("planner.quality_score", quality)

        # Note: planner_hit cannot be computed here (needs retrieval results)
        # It will be logged in retriever_step

    return state


print(states[0].choices)
