"""
citation_injector.py — Post-hoc citation injection for CaVe-VLM-CoT pipeline.

Works as a post-processing step between solver and verifier in the LangGraph pipeline.

Integration in verifier.py / build_cave_vlm_cot_graph:
    from citation_injector.citation_injector import inject_citations_step

    # Option A: Add as a separate node
    graph.add_node("inject_citations", inject_citations_step)
    graph.add_edge("solve", "inject_citations")
    graph.add_edge("inject_citations", "verify")

    # Option B: Call directly after solver in a combined node
    state = solver_step(state, model, tokenizer, kwargs)
    state = inject_citations(state, encoder=cross_encoder)
"""

import re
import os
from typing import List, Dict, Tuple, Optional

from utils import State
from retriever.retriever import cross_encoder, web_search
from utils import ChunkInfo

# ms-marco cross-encoder scores range roughly -10 to +10
# 0.5 balances precision and recall for the larger evidence pool 
# (web-first merge, cross-encoder filtered local docs).
# History:
#   0.8 → too conservative, many valid pairs in 0.4–0.8 range went uncited
#   0.4 → good for 10-chunk pool, too many false positives when evidence pool grows
#   0.5 → filters the extra false positives from the
#         wider retrieval while preserving the recall gains
# already tried - 1.5, 0.3, 0.8, 0.4
CITATION_THRESHOLD = 0.5
MIN_CLAIM_LENGTH = 20  # skip very short fragments
# Matches "From domain knowledge, <claim>" sentences produced by the solver
# when retrieved evidence didn't cover a reasoning step. We recover citations
# for these by running a targeted web search at inject time.
_DK_RE = re.compile(
    r'From domain knowledge[,.]?\s+(.{20,}?)(?=[.!?]|\Z)',
    re.IGNORECASE,
)
# Maximum number of domain-knowledge claims to web-search per question
# (caps latency: each web_search call adds ~1–2 s)
_DK_MAX_LOOKUPS = 3

# XML-like tags the solver produces — not real claims
_TAG_PATTERN = re.compile(
    r'</?(?:SUMMARY|CAPTION|REASONING|CONCLUSION|OBSERVATIONS|summary|caption|reasoning|conclusion|observations)>'
)

# Claim splitting
def _get_observations_span(reasoning: str):
    """Return (start, end) char span of <OBSERVATIONS>...</OBSERVATIONS>, or None."""
    m = re.search(r'<OBSERVATIONS>(.*?)</OBSERVATIONS>', reasoning, re.IGNORECASE | re.DOTALL)
    return (m.start(), m.end()) if m else None

def split_reasoning_into_claims(reasoning: str) -> List[Dict]:
    """
    Split solver reasoning into individual citable claims.
    Each claim is tagged with 'in_observations' so the cross-encoder
    matching step can skip visual observation sentences (which should only
    ever receive [Question Image N] citations, not text citations).
    """
    claims = []
    obs_span = _get_observations_span(reasoning)

    # Split on sentence ends, step markers, and bullet points
    segments = re.split(r'(?<=[.!?])\s+|(?=- Step \d)|(?=\n-\s)', reasoning)

    offset = 0
    for seg in segments:
        seg = seg.strip()
        if len(seg) < MIN_CLAIM_LENGTH:
            offset += len(seg) + 1
            continue

        # Skip lines that are just XML tags or tag + whitespace
        stripped = _TAG_PATTERN.sub('', seg).strip()
        if len(stripped) < MIN_CLAIM_LENGTH:
            offset += len(seg) + 1
            continue

        # Check for existing citations (Text Evidence or any image citation)
        existing = re.findall(
            r'\[Text Evidence \d+\]|\[Image ROI \d+\]|\[Question Image \d+\]', seg
        )

        # Clean text for cross-encoder matching (remove all citation labels)
        clean_text = re.sub(
            r'\[Text Evidence \d+\]|\[Image ROI \d+\]|\[Question Image \d+\]', '', seg
        ).strip()

        # Also strip any remaining XML tags
        clean_text = _TAG_PATTERN.sub('', clean_text).strip()

        if len(clean_text) >= MIN_CLAIM_LENGTH:
            # Search from `offset` so repeated phrases resolve to the
            # correct occurrence.  The old code searched from 0, causing all
            # duplicate sentences to record the same (wrong) start position.
            start = reasoning.find(seg, offset)
            claim_start = start if start >= 0 else offset
            # Mark whether this claim falls inside the <OBSERVATIONS> block.
            # Observation sentences are image-derived and should never receive
            # text evidence citations — only [Question Image N] citations.
            in_obs = bool(obs_span and claim_start >= obs_span[0] and claim_start < obs_span[1])
            claims.append({
                'text': clean_text,
                'original': seg,
                'start': claim_start,
                'end': claim_start + len(seg),
                'existing_citations': existing,
                'matched_citations': [],
                'in_observations': in_obs,
            })

        offset += len(seg) + 1

    return claims

# CANONICAL EVIDENCE LIST BUILDER

# Single source of truth for chunk ordering, filtering, and capping.
# Every place that needs to map [Text Evidence N] → chunk must call this so
# the numbering is identical across solver, citation_injector, verifier, and
# all evaluation functions.
#
# Rules (must stay in sync with any prompt changes):
#   1. dk_web_* keys first (DK enrichment), then all other non-_ keys in
#      insertion order — mirrors build_evidence_index priority.
#   2. Skip keys starting with '_'.
#   3. Per-query cap: first 5 chunks per key.
#   4. Length filter: skip chunks with len(stripped) <= 10.
#   5. Global cap: stop after `total_cap` chunks (default 15, set to match
#      build_evidence_index).  Pass total_cap=10 if you need the old solver
#      prompt limit — but prefer raising the solver cap to 15 so they match.
#   6. Truncate each chunk to `truncate` chars (default 500).  Pass
#      truncate=400 when building the solver prompt to preserve old behaviour.

def _is_yesno_dk_key(query: str, retrieved_chunks: dict) -> bool:
    """Return True for dk_web_ keys that were generated from yes/no answer choices.
    For yes/no MCQ questions the planner creates dk_web_<answer_choice> queries
    like 'dk_web_would you find yes' or 'dk_web_run sentence yes'.  These retrieve
    generic conversational-English content (e.g. "ways to say yes") rather than
    domain knowledge, and they occupy the priority dk_web_* slots while crowding
    out the actual domain-knowledge chunks retrieved by content queries.
    Detection heuristic: the query text (after stripping the 'dk_web_' prefix)
    ends with the bare words 'yes' or 'no'.
    """
    if not query.startswith("dk_web_"):
        return False
    suffix = query[len("dk_web_"):].strip().lower()
    return suffix.endswith(" yes") or suffix.endswith(" no") or suffix in ("yes", "no")

def _build_text_chunk_list(
    retrieved_chunks: dict,
    total_cap: int = 15,
    truncate: int = 500,
) -> list:
    """Return an ordered list of text chunks that defines [Text Evidence N] numbering.
    This is the single source of truth used by:
      - citation_injector.build_evidence_index  (to label and match chunks)
      - solver.format_evidence                  (to build the solver prompt)
      - verifier_step                           (to show evidence to verifier)
      - evaluations.get_text_evidence_by_id     (to resolve citation IDs)
      - evaluations.evidence_grounding_check    (to collect cited chunks)
    Args:
        retrieved_chunks: state.retrieved_chunks dict
        total_cap:  maximum number of chunks to return (15 by default, matching
                    build_evidence_index; use 10 only for backward compat)
        truncate:   truncate each chunk to this many characters (500 default;
                    solver prompt uses 400 for context-length safety)
    Returns:
        List of chunk strings, 0-indexed, where index i → [Text Evidence i+1].
    """
    if not retrieved_chunks:
        return []
    # dk_web_* keys first so DK-enrichment evidence isn't evicted by the cap.
    # Exception: skip dk_web_ keys derived from yes/no answer choices — these
    # retrieve generic conversational-English content instead of factual evidence
    # and poison the evidence list for factual yes/no questions.
    dk_keys = [q for q in retrieved_chunks
               if isinstance(q, str) and q.startswith("dk_web_")
               and not _is_yesno_dk_key(q, retrieved_chunks)]
    other_keys = [q for q in retrieved_chunks
                  if q not in dk_keys
                  and not (isinstance(q, str) and q.startswith("_"))
                  and not (isinstance(q, str) and _is_yesno_dk_key(q, retrieved_chunks))]
    ordered_keys = dk_keys + other_keys
    chunks = []
    for query in ordered_keys:
        chunk_info = retrieved_chunks[query]
        raw = (chunk_info.text_chunks
               if hasattr(chunk_info, "text_chunks")
               else chunk_info.get("text_chunks", []))
        for chunk in raw[:5]:                         # per-query cap
            if chunk and len(chunk.strip()) > 10:     # length filter
                text = chunk.strip()
                if truncate:
                    text = text[:truncate]
                chunks.append(text)
        if len(chunks) >= total_cap:
            break

    return chunks[:total_cap]

# Evidence index
def _is_generic_lecture(chunk: str) -> bool:
    """Detect ScienceQA generic lecture prefixes that aren't specific evidence."""
    prefixes = ("natural science", "social science", "language science")
    stripped = chunk.strip().lower()
    return any(stripped.startswith(p) for p in prefixes)

def build_evidence_index(state: State) -> Tuple[List[str], List[str], List[bool]]:
    """
    Extract all evidence items from state.retrieved_chunks with labels.
    Text chunks are ordered and capped by _build_text_chunk_list (the single
    source of truth), so label numbers here match solver and verifier exactly.
    Generic lecture chunks are included (to preserve label alignment) but
    flagged via is_generic so match_claims_to_evidence can skip them.
    Returns:
        evidence_texts: content strings
        evidence_labels: labels like "[Text Evidence 1]" or "[Question Image 1]"
        is_generic: whether each entry is a generic lecture (skip during matching)
    """
    # Delegate to canonical helper — same ordering, filtering, and cap as
    # solver.format_evidence, verifier_step, and get_text_evidence_by_id.
    text_chunks = _build_text_chunk_list(
        state.retrieved_chunks, total_cap=15, truncate=500
    )
    evidence_texts  = list(text_chunks)
    evidence_labels = [f"[Text Evidence {i+1}]" for i in range(len(text_chunks))]
    is_generic      = [_is_generic_lecture(c) for c in text_chunks]
    
    # Add Question Image captions for matching
    qi_idx = 1
    for img_path in (state.image_paths or []):
        if img_path and os.path.exists(img_path):
            img_filename = os.path.basename(img_path)
            caption = (state.img_captions or {}).get(img_filename, "")
            # Create a searchable description for the image
            if caption:
                img_desc = f"Image showing: {caption}"
            else:
                img_desc = f"Visual evidence from question image showing the subject matter"
            evidence_texts.append(img_desc)
            evidence_labels.append(f"[Question Image {qi_idx}]")
            is_generic.append(False)  # images are never generic lectures
            qi_idx += 1
            if qi_idx > 5:  # Max 5 question images
                break
    return evidence_texts, evidence_labels, is_generic

# Cross-encoder matching
def match_claims_to_evidence(
    claims: List[Dict],
    evidence_texts: List[str],
    evidence_labels: List[str],
    cross_encoder,
    is_generic=None,
    threshold: float = CITATION_THRESHOLD,
    max_citations_per_claim: int = 2,
) -> List[Dict]:
    """
    For each uncited claim, find best-matching evidence via cross-encoder.
    Skips generic lecture chunks during matching but preserves label numbering
    so injected [Text Evidence N] labels stay aligned with the solver's prompt.
    """
    if not evidence_texts or not claims:
        return claims

    for claim in claims:
        # Skip claims that already have citations
        if claim['existing_citations']:
            continue

        # Skip claims inside <OBSERVATIONS> — these describe what the model
        # sees in images and should only ever cite [Question Image N], never
        # text evidence. The QI injection step handles them separately.
        if claim.get('in_observations'):
            continue

        # Build pairs, filtering out generic lectures while keeping
        # labels/texts in sync so the zip after scoring is correct.
        pairs = []
        filtered_labels = []
        filtered_texts = []
        for i, ev in enumerate(evidence_texts):
            if is_generic and i < len(is_generic) and is_generic[i]:
                continue  # skip generic lectures
            pairs.append([claim['text'], ev])
            filtered_labels.append(evidence_labels[i])
            filtered_texts.append(ev)
        if not pairs:
            continue
        try:
            scores = cross_encoder.predict(pairs)
        except Exception as e:
            print(f"  [CitationInjector] Cross-encoder error: {e}")
            continue

        scored = sorted(
            zip(filtered_labels, scores, filtered_texts),
            key=lambda x: x[1],
            reverse=True,
        )

        for label, score, ev_text in scored[:max_citations_per_claim]:
            if score >= threshold:
                claim['matched_citations'].append({
                    'label': label,
                    'score': float(score),
                    'evidence_preview': ev_text[:80],
                })

    return claims

# Citation insertion
def inject_citations_into_reasoning(reasoning: str, claims: List[Dict]) -> str:
    """
    Insert matched citation labels into reasoning text.
    Works backwards through the text so character offsets remain valid.
    Appends citations at the end of each matched claim sentence.
    """
    insertions = []  # (position, chars_to_delete, replacement_text)

    for claim in claims:
        if not claim['matched_citations']:
            continue

        labels = [m['label'] for m in claim['matched_citations']]
        original = claim['original']
        pos = reasoning.find(original)
        if pos < 0:
            # Fallback: try from the beginning in case offset drifted slightly
            pos = reasoning.find(original)
        if pos < 0:
            continue

        if original.rstrip().endswith('.'):
            # Replace trailing period: "claim." → "claim [citation]."
            period_pos = pos + len(original.rstrip()) - 1
            citation_str = ' ' + ' '.join(labels) + '.'
            insertions.append((period_pos, 1, citation_str))
        else:
            # Append after claim
            insert_at = pos + len(original)
            citation_str = ' ' + ' '.join(labels)
            insertions.append((insert_at, 0, citation_str))

    # Sort descending by position so earlier insertions don't shift later offsets.
    insertions_with_idx = [(pos, delete_len, text, i) for i, (pos, delete_len, text) in enumerate(insertions)]
    insertions_with_idx.sort(key=lambda x: (x[0], -x[3]), reverse=True)

    result = reasoning
    for pos, delete_len, text, _idx in insertions_with_idx:
        result = result[:pos] + text + result[pos + delete_len:]

    return result

def inject_qi_into_observations(reasoning: str, num_images: int) -> str:
    """
    Inject [Question Image N] citations into every substantive uncited line
    inside the <OBSERVATIONS> block.
    Strategy:
    - If a line already contains [Question Image N]: leave it unchanged.
    - If a line contains an explicit "Image N" reference: inject at that spot
      (preserves alignment when the solver numbered images explicitly).
    - Otherwise: infer the image number from context and append the citation
      at the end of the line. Every line inside <OBSERVATIONS> is by definition
      a visual observation, so all substantive lines should cite an image.
    Image number inference for uncited lines (no "Image N" marker):
    - Single image question → always [Question Image 1].
    - Multi-image: scan backwards for the most recently established image
      number in the same block; default to 1 if none found yet.
      This handles blocks like:
          "The left panel shows a forest ecosystem."   ← no explicit number
          "The right panel shows a bar chart."         ← no explicit number
      where the solver omitted the numbering prefix.
    """
    if num_images == 0:
        return reasoning
    
    obs_match = re.search(
        r'(<OBSERVATIONS>)(.*?)(</OBSERVATIONS>)',
        reasoning,
        re.IGNORECASE | re.DOTALL
    )
    
    if not obs_match:
        return reasoning

    obs_content = obs_match.group(2)
    lines = obs_content.split('\n')
    new_lines = []
    last_img_num = 1  # running tracker for multi-image inference

    for line in lines:
        stripped = line.strip()

        # Skip blank lines and lines that are just dashes/bullets with no content
        if not stripped or stripped in ('-', '*', '•'):
            new_lines.append(line)
            continue

        # Skip if already has a [Question Image N] citation
        if '[Question Image' in line:
            # Update tracker in case we see "Image 2" already cited
            m = re.search(r'\[Question Image (\d+)\]', line)
            if m:
                last_img_num = int(m.group(1))
            new_lines.append(line)
            continue

        # Case 1: explicit "Image N" reference in the line — inject inline
        img_ref = re.search(r'[Ii]mage\s*(\d+)', line)
        if img_ref:
            img_num = int(img_ref.group(1))
            if 1 <= img_num <= num_images:
                last_img_num = img_num
                citation = f'[Question Image {img_num}]'
                line = re.sub(
                    r'([Ii]mage\s*\d+:?)',
                    rf'\1 {citation}',
                    line,
                    count=1
                )
            new_lines.append(line)
            continue

        # Case 2: no explicit image reference — append citation at end of line.
        # Only inject on lines with enough content to be a real observation
        # (guards against injecting on section headers or very short fragments).
        if len(stripped) >= 15:
            img_num = last_img_num if num_images >= last_img_num else 1
            citation = f'[Question Image {img_num}]'
            # Append before trailing period if present, otherwise at end
            if line.rstrip().endswith('.'):
                line = line.rstrip()[:-1] + f' {citation}.'
            else:
                line = line.rstrip() + f' {citation}'
        new_lines.append(line)
    new_obs = '\n'.join(new_lines)
    return reasoning[:obs_match.start(2)] + new_obs + reasoning[obs_match.end(2):]

def _has_sufficient_kb_evidence(state: State, min_chunks: int = 2, min_length: int = 80) -> bool:
    """
    Return True if the KB retrieval already produced enough substantive evidence
    that DK enrichment is unlikely to add value.

    Criteria: at least `min_chunks` text chunks across all non-DK subqueries
    that are (a) longer than `min_length` chars and (b) not generic lecture headers.

    When this returns True, `_enrich_domain_knowledge_evidence` is skipped to
    avoid polluting the evidence index with loosely-matched web snippets, which
    degrades citation precision on well-retrieved questions.

    DK enrichment helped image questions (weak KB retrieval, Δcite_prec=+1.7pp) 
    but hurt text-only questions (good KB retrieval, Δcite_prec=−9.1pp) 
    because topical-but-imprecise web snippets matched at
    the 0.4 cross-encoder threshold and displaced higher-quality KB citations.

    min_length=150 was too strict for language science — grammar
    and vocabulary KB chunks are typically 80–120 chars. Those questions slipped
    through the gate, received DK web snippets, and lost citation precision.
    Lowered to 80 chars so short-but-precise language science chunks are correctly
    counted as sufficient evidence.
    """
    substantive = 0
    for key, ci in (state.retrieved_chunks or {}).items():
        if isinstance(key, str) and (key.startswith("dk_web_") or key.startswith("_")):
            continue
        chunks = ci.text_chunks if hasattr(ci, 'text_chunks') else ci.get('text_chunks', [])
        for chunk in chunks[:5]:
            if (chunk and len(chunk.strip()) > min_length
                    and not chunk.strip().lower().startswith(
                        ("natural science", "social science", "language science"))):
                substantive += 1
                if substantive >= min_chunks:
                    return True
    return False

def _enrich_domain_knowledge_evidence(state: State) -> State:
    """
    Retroactively back "From domain knowledge, <claim>" sentences with web evidence.

    The solver emits these steps when retrieved chunks don't cover a reasoning
    point. They are typically *correct* (the model's parametric knowledge is
    reliable for common science facts) but uncited, which drags citation_recall
    and AIS.

    For each such sentence we:
      1. Extract the factual claim portion after the prefix.
      2. Run a targeted web_search(claim, k=2) to surface supporting snippets.
      3. Store the snippets in state.retrieved_chunks under a 'dk_web_N' key.

    Downstream consumers that already read from retrieved_chunks benefit automatically:
      - build_evidence_index (citation_injector) → can inject [Text Evidence N]
      - _collect_all_evidence (evaluations) → improves AIS / hallucination_rate

    Capped at _DK_MAX_LOOKUPS to limit added latency (~1–2 s per lookup).
    """
    reasoning = state.reasoning_steps[0] if state.reasoning_steps else ""
    if not reasoning:
        return state

    dk_claims = _DK_RE.findall(reasoning)
    if not dk_claims:
        return state

    added = 0
    for i, claim in enumerate(dk_claims[:_DK_MAX_LOOKUPS]):
        claim_clean = claim.strip()[:150]
        key = f"dk_web_{i + 1}"
        if key in state.retrieved_chunks:
            continue  # already enriched (e.g. on retry)
        try:
            snippets = web_search(claim_clean, k=2)
        except Exception as e:
            print(f"  [CitationInjector] DK web_search failed: {e}")
            snippets = []
        if snippets:
            state.retrieved_chunks[key] = ChunkInfo(
                text_chunks=snippets,
                image_rois=[],
            )
            added += 1
            print(
                f"  [CitationInjector] DK web: '{claim_clean[:60]}...' "
                f"→ {len(snippets)} snippets (key={key})"
            )

    if added:
        print(f"  [CitationInjector] DK enrichment: {added} domain-knowledge "
              f"claim(s) backed by web snippets")
    return state

def inject_citations(state: State, encoder=None) -> State:
    """
    Post-hoc citation injection into solver reasoning.
    - Skip injection for image-primary questions with non-substantive text
    - Use single threshold (0.8) for all evidence
    - Simple pattern-based QI injection in OBSERVATIONS only
    """
    _encoder = encoder or cross_encoder

    if not state.reasoning_steps or not state.retrieved_chunks:
        return state

    # Step 0: Retroactively back "From domain knowledge, X" steps with web
    # evidence before building the evidence index. This runs web_search only
    # for claims that have no KB match, and stores the results in
    # state.retrieved_chunks so both AIS and citation injection benefit.

    # GATED: skip when KB retrieval already produced ≥2 substantive chunks —
    # adding web snippets in that case reduces citation precision.
    if not _has_sufficient_kb_evidence(state):
        state = _enrich_domain_knowledge_evidence(state)
    else:
        print("  [CitationInjector] DK enrichment skipped — KB evidence sufficient")

    reasoning = state.reasoning_steps[0] if state.reasoning_steps else ""
    if not reasoning:
        return state

    # Count existing citations BEFORE
    existing_text = len(re.findall(r'\[Text Evidence \d+\]', reasoning))
    existing_roi = len(re.findall(r'\[Image ROI \d+\]', reasoning))
    existing_qi = len(re.findall(r'\[Question Image \d+\]', reasoning))

    # Count images
    num_images = sum(1 for p in (state.image_paths or []) if p and os.path.exists(p))
    # Step 1: Simple QI injection into OBSERVATIONS (pattern-based only)
    if num_images > 0:
        reasoning = inject_qi_into_observations(reasoning, num_images)
        
    # Step 2: Determine whether cross-encoder matching should run.
    # We always proceed to cross-encoder matching, but skip claims that
    # fall inside the <OBSERVATIONS> block (handled above in match_claims_to_evidence).
    # Only skip if there is genuinely no text evidence at all.
    has_observations = bool(re.search(
        r'<(?:OBSERVATIONS|CAPTION)>', reasoning, re.IGNORECASE
    ))
    text_is_substantive = any(
        len(chunk.strip()) > 50
        for ci in (state.retrieved_chunks or {}).values()
        for chunk in (ci.text_chunks if hasattr(ci, 'text_chunks') else ci.get('text_chunks', []))[:3]
        if not chunk.strip().lower().startswith(
            ("natural science", "social science", "language science")
        )
    )

    # Only skip cross-encoder when there's no useful text evidence at all
    # (purely visual question with no substantive retrieval).
    if has_observations and not text_is_substantive:
        # Still update with QI injections from step 1
        state.reasoning_steps = [reasoning]
        final_qi = len(re.findall(r'\[Question Image \d+\]', reasoning))
        print(f"  CitationInjector: skipped cross-encoder (no substantive text evidence), "
              f"QuestionImage {existing_qi}→{final_qi}")
        return state

    # Step 3: Split into claims
    claims = split_reasoning_into_claims(reasoning)
    uncited = [c for c in claims if not c['existing_citations']]

    # Targeted CONCLUSION injection — always attempt to cite the final answer
    # sentence regardless of how many other claims are already cited.

    # grounding_score checks NLI(answer_claim, cited_evidence): if the conclusion
    # is uncited the grounding check has no evidence to work with and fails.
    # We extract the conclusion span and mark uncited conclusion claims so the
    # early-exit below cannot skip them.
    _CONCLUSION_RE = re.compile(
        r'<CONCLUSION>(.*?)</CONCLUSION>', re.IGNORECASE | re.DOTALL
    )

    conclusion_span = None
    _cm = _CONCLUSION_RE.search(reasoning)
    if _cm:
        conclusion_span = (_cm.start(), _cm.end())

    def _in_conclusion(claim):
        if conclusion_span is None:
            return False
        return claim['start'] >= conclusion_span[0] and claim['start'] < conclusion_span[1]
    uncited_conclusion = [c for c in uncited if _in_conclusion(c)]

    # If >50% already cited, skip — UNLESS there are uncited conclusion or
    # reasoning claims that need targeted injection 
    # Threshold kept at >50% (reverted from >80%) to avoid injecting
    # weak citations onto transitional/meta sentences.
    cited_count = len(claims) - len(uncited)
    if len(claims) > 0 and cited_count / len(claims) > 0.5:
        # Build evidence index once — shared by both targeted passes below.
        evidence_texts, evidence_labels, is_generic = build_evidence_index(state)
        
        # Conclusion-targeted injection (all question types).
        # grounding_score checks NLI(answer_claim, cited_evidence) — if the
        # conclusion is uncited the grounding check fails.
        if uncited_conclusion and evidence_texts:
            uncited_conclusion = match_claims_to_evidence(
                uncited_conclusion, evidence_texts, evidence_labels, _encoder,
                is_generic=is_generic,
                threshold=CITATION_THRESHOLD,
                max_citations_per_claim=2,
            )
            new_conc = sum(len(c['matched_citations']) for c in uncited_conclusion)
            if new_conc > 0:
                reasoning = inject_citations_into_reasoning(reasoning, uncited_conclusion)
                print(f"  CitationInjector: conclusion-targeted injection ({new_conc} matches)")
        
        state.reasoning_steps = [reasoning]
        final_qi = len(re.findall(r'\[Question Image \d+\]', reasoning))
        print(f"  CitationInjector: {cited_count}/{len(claims)} already cited (>{50}%), "
              f"QuestionImage {existing_qi}→{final_qi}")
        return state

    # Step 4: Build evidence index
    evidence_texts, evidence_labels, is_generic = build_evidence_index(state)
    if not evidence_texts:
        state.reasoning_steps = [reasoning]
        print("  CitationInjector: no evidence available")
        return state

    # Step 5: Get cross-encoder and match
    claims = match_claims_to_evidence(
        claims, evidence_texts, evidence_labels, _encoder,
        is_generic=is_generic,
        threshold=CITATION_THRESHOLD,
        max_citations_per_claim=2,
    )

    # Step 6: Inject matched citations
    new_citations = sum(len(c['matched_citations']) for c in claims)
    if new_citations > 0:
        reasoning = inject_citations_into_reasoning(reasoning, claims)

    # Update state
    state.reasoning_steps = [reasoning]

    # Log results
    final_text = len(re.findall(r'\[Text Evidence \d+\]', reasoning))
    final_roi = len(re.findall(r'\[Image ROI \d+\]', reasoning))
    final_qi = len(re.findall(r'\[Question Image \d+\]', reasoning))

    print(f"  CitationInjector: text {existing_text}→{final_text}, "
          f"ROI {existing_roi}→{final_roi}, "
          f"QuestionImage {existing_qi}→{final_qi} "
          f"({new_citations} cross-encoder matches)")

    return state


def inject_citations_step(state: State) -> State:
    """
    LangGraph node wrapper. Drop into the graph between solver and verifier:

        graph.add_node("inject_citations", inject_citations_step)
        graph.add_edge("solve", "inject_citations")
        graph.add_edge("inject_citations", "verify")
    """
    from tracer import tracer

    with tracer.start_as_current_span(
        "CitationInjector", openinference_span_kind="chain"
    ) as span:
        state = inject_citations(state)
        reasoning = state.reasoning_steps[0] if state.reasoning_steps else ""
        text_cites = len(re.findall(r'\[Text Evidence \d+\]', reasoning))
        roi_cites = len(re.findall(r'\[Image ROI \d+\]', reasoning))
        qi_cites = len(re.findall(r'\[Question Image \d+\]', reasoning))

        span.set_attribute("citation_injector.text_citations", text_cites)
        span.set_attribute("citation_injector.roi_citations", roi_cites)
        span.set_attribute("citation_injector.question_image_citations", qi_cites)
        span.set_attribute("citation_injector.total", text_cites + roi_cites + qi_cites)
        span.set_attribute("citation_injector.reasoning", reasoning)
    return state