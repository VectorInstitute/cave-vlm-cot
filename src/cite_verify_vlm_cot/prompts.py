# Prompt template
# The few-shot examples are in JSON format so the model learns the required
# output structure from the examples themselves.

# Design principles:
#   1. Only question + choices + image-context in the live prompt.
#      Hint / lecture / subject / topic live in the KB and are surfaced via
#      well-phrased queries — not injected as privileged context.
#   2. Three query families per example: definitional, choice-specific,
#      comparative — covering the main KB document types
#   3. Prompt ends with "[" to prime JSON array generation.

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

# Prompt for text-only questions (no images)
SOLVER_PROMPT_TEMPLATE = """
You are an expert reasoning assistant. Answer using ONLY the evidence provided below.
Question: {question_text}

Choices:
{answer_choices}

Retrieved Text Evidence:
{text_evidence}

CITATION RULES (STRICT):
- MANDATORY: Every REASONING step that uses information from the retrieved evidence MUST cite [Text Evidence N].
  Format: "According to [Text Evidence N], ..." or "... [Text Evidence N]."
- ONLY cite [Text Evidence N] if that specific chunk contains the fact you are stating.
- If the evidence does NOT contain the information needed, use "From domain knowledge, ..." — do not make the claim with no attribution.
- NEVER paraphrase a general concept and then cite a text chunk that merely discusses the same broad topic.
- If ALL reasoning steps rely on domain knowledge and NO retrieved text applies, that is acceptable — but always check the evidence FIRST before invoking domain knowledge.

Answer using this structured format:

<SUMMARY>
State the core problem in 1-2 sentences. Do not include factual claims here — save those for REASONING.
</SUMMARY>

<REASONING>
Each step must follow one of these two forms:
- "According to [Text Evidence N], ..." — use this when the evidence directly supports the claim.
- "From domain knowledge, ..." — use this only when no retrieved evidence covers the point.
Never fabricate a citation. Never omit a reasoning step you need to reach the answer.
</REASONING>

<CONCLUSION>
Restate the answer from REASONING only. Do NOT introduce new facts.
The answer is [LETTER]: [option text]
</CONCLUSION>
"""

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
CITATION RULES (STRICT):
- MANDATORY: Every REASONING step that uses information from Retrieved Text Evidence MUST cite [Text Evidence N].
  Format: "According to [Text Evidence N], ..." — do this for EVERY step drawing on retrieved text.
- Use [Question Image N] when describing what you actually see in an image.
- ONLY cite [Text Evidence N] if that specific chunk contains the fact you are stating — never for broad-topic match.
- NEVER attach a citation to a claim just because the evidence discusses the same broad topic.
- If retrieved text does NOT cover a point, use "From domain knowledge, ..." — always check evidence FIRST.
- REASONING CITATION RULE: If a REASONING step is based on what you see in an image
  (e.g. reading a label, counting objects, identifying a shape, interpreting a diagram),
  you MUST cite [Question Image N] in that REASONING step — not just in OBSERVATIONS.
  Example: "Looking at [Question Image 1], the bar for 2020 is taller than 2019."
---
EXAMPLE (do NOT copy — answer YOUR question):

<OBSERVATIONS>
[Question Image 1] shows a diagram with two containers. Container A has 3 particles, Container B has 5 particles. Both have 40mL volume labeled.
</OBSERVATIONS>

<REASONING>
- Looking at [Question Image 1], I can count the particles: Container A has 3, Container B has 5.
- According to [Text Evidence 1], concentration = particles / volume.
- Since both containers have equal volume (40mL) [Question Image 1], Container B has higher concentration.
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
Each step must follow one of these forms:
- "According to [Text Evidence N], ..." — when text evidence supports.
- "Looking at [Question Image N], ..." — when your reasoning depends on what you see in the image.
- "From domain knowledge, ..." — only when no evidence covers the point.
Never fabricate a citation. Never omit a reasoning step you need to reach the answer.

</REASONING>

<CONCLUSION>
Restate the answer from REASONING only. Do NOT introduce new facts.
The answer is [LETTER]: [full answer text]
</CONCLUSION>

---
CRITICAL:
- You MUST cite [Question Image N] when describing visual observations
- You MUST cite [Question Image N] in any REASONING step that depends on what you see
- ONLY cite [Text Evidence N] when that specific evidence supports your claim else don't make the claim
- Your CONCLUSION must match your REASONING and add no new information
- MANDATORY: Every observation MUST include [Question Image N] citation.
"""

VERIFIER_PROMPT_TEMPLATE = """
You are a strict fact-checker. Your ONLY job: does the evidence actually support the solver's claims?

QUESTION
{question}

ANSWER CHOICES
{choices}

SOLVER'S ANSWER
{solver_answer}

SOLVER'S REASONING
{solver_reasoning}

RETRIEVED TEXT EVIDENCE
{text_evidence}

QUESTION IMAGES (shown above in order)
{visual_evidence}

---
TASK: Verify whether the solver's reasoning is hallucination-free using the following
structured chain-of-thought. Work through each step in order — do not skip steps.

STEP 1 — IDENTIFY THE KEY CLAIM
State the single most important factual claim the solver makes to reach its answer.
Key claim: [one sentence]

STEP 2 — LOCATE SUPPORTING EVIDENCE
For each citation used anywhere in the solver's reasoning, quote what the evidence actually says.
- [Text Evidence N] says: "[exact relevant quote or 'does not exist']"
- [Question Image N] shows: "[what you actually see, or 'not visible']"

STEP 3 — CHECK EACH CITATION
For every citation in the solver's reasoning (not just the key claim):
- Does [Text Evidence N] contain the stated fact? YES / NO / PARTIALLY
- Does [Question Image N] actually show what is described? YES / NO / PARTIALLY
List any mismatch as a candidate hallucination.

STEP 4 — CLASSIFY HALLUCINATIONS
Based on Step 3, classify:
- NONE DETECTED: all citations check out; any uncited claims are common knowledge or logical inference
- MINOR HALLUCINATIONS: citation slightly misrepresents evidence but conclusion is still plausible
- MAJOR HALLUCINATIONS: citation is fabricated, contradicts evidence, or conclusion depends on a false claim

STEP 5 — RENDER VERDICT
Given the hallucination classification, decide:
- VERIFIED: the solver's answer is well-supported and hallucination-free (or only minor)
- REJECTED: the solver's answer depends on a major hallucination

---
OUTPUT FORMAT (use EXACTLY this format after completing the steps above):

Hallucination Check: [NONE DETECTED] or [MINOR HALLUCINATIONS] or [MAJOR HALLUCINATIONS]

[If hallucinations detected, list each one:]
Claim: "[exact quote from solver]"
Issue: [fake citation / not in evidence / misrepresented / fabricated]
Evidence: [what the evidence actually says, or "doesn't exist"]

[If no hallucinations:]
All claims properly supported by evidence or common knowledge.

Final Verdict: [VERIFIED] or [REJECTED]

Confidence: [HIGH] or [MEDIUM] or [LOW]

Verified Answer: [copy the answer letter here — see rule below]

VERIFIED ANSWER RULE (mandatory):
- If Final Verdict is VERIFIED → copy the letter from SOLVER'S ANSWER (e.g. if solver said "The answer is B", write: Verified Answer: B)
- If Final Verdict is REJECTED and you know the correct answer → write that letter (e.g. Verified Answer: C)
- If Final Verdict is REJECTED and you cannot determine the correct answer → write: Verified Answer: INCONCLUSIVE

EXAMPLE — do not copy, just follow the pattern:
  SOLVER'S ANSWER: The answer is B: Container B has higher concentration
  Final Verdict: [VERIFIED]
  Confidence: [HIGH]
  Verified Answer: B       ← copied from solver since VERIFIED
"""