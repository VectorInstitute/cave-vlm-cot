import json
import os
import re
import time as _time
from typing import List, Tuple

import numpy as np

# import phoenix as px
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()  # loads .env into os.environ before any key checks

# DuckDuckGo search - FREE, no API key needed!
# from ddgs import DDGS
import torch
from duckduckgo_search import DDGS


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import nest_asyncio


nest_asyncio.apply()

import functools
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluations import recall_at_k
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tracer import tracer
from utils import ChunkInfo


# Load cross-encoder — force onto GPU if available so batch scoring runs
# ~10× faster than CPU (A100 latency: ~50ms vs ~500ms for 15-pair batches).
# Pin cross-encoder to cuda:0 explicitly.
# cuda:0 hosts the small Qwen3-8B-4bit planner (~5GB) and leaves >70GB free,
# so co-locating the <0.5GB cross-encoder there wastes nothing while giving
# ~10× speedup vs CPU.  Using bare "cuda" (without an index) defaults to
# cuda:0 only when CUDA_VISIBLE_DEVICES is set correctly; being explicit
# avoids surprises in multi-GPU SLURM environments.
_CE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=_CE_DEVICE,
)

text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper functions


# @tracer.chain()
def text_to_embedding(text):
    text = text.replace("\n", " ")
    embedding = text_model.encode(text, batch_size=1, normalize_embeddings=True)
    return embedding


@functools.lru_cache(maxsize=131072)
def _web_search_cached(query: str, k: int) -> tuple:
    """
    Cached inner implementation of web_search.
    ScienceQA has heavy query overlap across 21,000 rows — the same
    "latitude of X", "definition of Y", "Punnett square Z" queries appear
    dozens of times.  An LRU cache keyed on (query, k) eliminates redundant
    DDG round-trips (~2-5s each) and reduces rate-limit pressure.
    maxsize=131072 (128k): Qwen2.5 generates 7-8 subqueries per question;
    across 21,000 questions that is up to ~168,000 unique queries. 128k covers
    the vast majority given the overlap between questions on the same topic.
    (Previous value of 4096 was sized for the 5k pilot run and caused cache
    thrashing on the full 21k dataset, re-issuing DDG calls for queries already
    seen earlier in the same run.)

    Returns a tuple (hashable) so lru_cache can store it.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=k))
            return tuple(r["body"] for r in results if r.get("body"))
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(w in err for w in ("ratelimit", "rate limit", "202", "blocked", "timeout"))
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2**attempt  # 1s, 2s, 4s
                print(f"  [DDG] Rate limited on attempt {attempt + 1}, retrying in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"  DuckDuckGo search failed: {e}")
                return tuple()
    return tuple()


def web_search(query: str, k: int = 2) -> List[str]:
    """
    Search using DuckDuckGo (free, no API key needed).
    Results are LRU-cached by (query, k): identical queries within a run
    return instantly without a network call.
    Includes exponential backoff on rate-limit errors so that parallel bursts
    from _web_search_with_choices don't permanently exhaust the DDG rate-limit window.

    Args:
        query: Search query string
        k: Number of results to return
    Returns:
        List of text snippets from search results
    """
    return list(_web_search_cached(query, k))


# HYBRID RETRIEVER
class BM25Retriever:
    """BM25 sparse retriever for keyword-based search."""

    def __init__(self, corpus_texts: list):
        self.corpus = corpus_texts
        tokenized = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_k_indices]


def dense_retrieval(query: str, text_index, k: int = 5) -> list:
    # IMPORTANT: text_index must have been built from text-only rows so that
    # integer positions returned by FAISS map correctly to the reset-index
    # text_data slice used in hybrid_retrieval.
    q_embed = text_to_embedding(query).reshape(1, -1)
    D, I = text_index.search(q_embed.astype(np.float32), k)
    # Note: FAISS returns distances, lower is better for L2, higher for IP
    # Convert to (index, score) format
    return [(I[0][i], float(D[0][i])) for i in range(k) if I[0][i] >= 0]


def rrf_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    scores = {}

    # Add dense retrieval scores
    for rank, (idx, _) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Add sparse retrieval scores
    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by combined RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieval(
    query: str, text_index, bm25_retriever: BM25Retriever, data, k: int = 5, dense_weight: float = 0.5
) -> list:
    # Reset index so FAISS row numbers (0-based integer positions) and
    # BM25Retriever corpus positions both map into the same DataFrame slice.
    # Without this, a DataFrame that interleaves text and image rows would
    # cause FAISS index position i to refer to a different row than iloc[i].

    # Filter to text-only data
    text_data = data[data["media_type"] == "text"].reset_index(drop=True)

    # Get more candidates than needed for fusion
    num_candidates = k * 3

    # Dense retrieval (uses text_data indices)
    dense_results = dense_retrieval(query, text_index, k=num_candidates)

    # Sparse retrieval (BM25 already built on text corpus)
    sparse_results = bm25_retriever.retrieve(query, k=num_candidates)

    # Fuse results using RRF
    fused_results = rrf_fusion(dense_results, sparse_results)

    # Return top-k document texts
    return [text_data.iloc[idx]["text"].strip() for idx, _ in fused_results[:k]]


# RERANKER

# https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669
# https://medium.com/@aishikbhattacharjee98/reranking-using-cross-encoder-boost-your-rag-pipeline-accuracy-d2da22006dad
# https://medium.com/@abheshith7/mastering-reranking-in-rag-from-basic-retrieval-to-advanced-methods-db297530361a


def rerank_with_cross_encoder(query: str, documents: list, top_k: int = 2) -> list:
    """Rerank documents using cross-encoder."""
    if not documents:
        return []

    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]

    # Get scores from cross-encoder
    scores = cross_encoder.predict(pairs)

    # Sort by score (descending) and return top_k
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]


# Helpers for retriever_step (generic, no question-metadata injection)
def _expand_subquery(query: str) -> List[str]:
    """
    Generate lightweight paraphrases of a planner subquery for query expansion.

    Motivation: planner_hit has been flat at 28% across all versions.
    The primary cause is vocabulary mismatch between planner-generated queries
    and KB / web content. The same fact can be expressed in many ways, and a
    single query formulation reliably misses documents that use different
    terminology (e.g. "photosynthesis light reaction" vs "Calvin cycle input").

    Strategy: rule-based paraphrase generation, zero extra LLM calls, ~0ms overhead.
    Three expansion types, each targeting a different vocabulary gap:
      1. Keyword extraction — strip stop words and emit the core content words
         as a compact query. Catches cases where the full query is too specific
         but the core concept is in the KB.
      2. Concept-first reorder — move the last content word to the front.
         Targets cases where the KB indexes by concept name first (e.g.
         "mitosis definition" → "definition mitosis cell division phase").
      3. "What is X" → "X definition explanation" — rephrase definition-seeking
         queries into noun-phrase form, which BM25 handles better than
         question-form queries.

    Returns: list of unique paraphrases (not including the original query).
    Capped at 2 paraphrases to avoid flooding the evidence pool.
    """
    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "through",
        "during",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "that",
        "this",
        "these",
        "those",
        "and",
        "or",
        "but",
        "if",
        "as",
        "it",
    }

    words = query.strip().split()
    content_words = [w for w in words if w.lower() not in stop_words and len(w) >= 3]
    paraphrases = []

    # 1. Keyword-only compact query
    if len(content_words) >= 2:
        compact = " ".join(content_words[:6])
        if compact.lower() != query.lower():
            paraphrases.append(compact)

    # 2. "What is X" → "X definition explanation"
    wh_match = re.match(r"^(what\s+is|what\s+are|define|definition\s+of)\s+(.+)", query, re.IGNORECASE)
    if wh_match:
        concept = wh_match.group(2).strip().rstrip("?")
        rephrased = f"{concept} definition explanation"
        if rephrased.lower() != query.lower() and rephrased not in paraphrases:
            paraphrases.append(rephrased)
    elif len(content_words) >= 2:
        # 3. Concept-first reorder (move last content word to front)
        reordered = f"{content_words[-1]} {' '.join(content_words[:-1])}"
        if reordered.lower() != query.lower() and reordered not in paraphrases:
            paraphrases.append(reordered)
    return paraphrases[:2]


def _web_search_with_choices(query: str, choices: list, k: int = 2) -> list:
    """
    Run web search with the base subquery PLUS choice-augmented variants,
    all in parallel via ThreadPoolExecutor.

    Sequential DDG calls at ~3-5s each were the single biggest latency
    bottleneck (~35s/question at 8 subqueries × 3 calls each). Parallel
    execution collapses N calls to the time of the slowest single call (~5s).

    Choice-augmented variants produce discriminating snippets for comparative
    questions (e.g. "northernmost state Idaho") that a generic query would miss.
    Results are deduplicated before returning.

    This function uses ONLY query text and choices — no question metadata.
    """
    # Build list of (query_string, k) jobs — base query first
    jobs: List[Tuple[str, int]] = [(query, k)]

    for choice in choices[:4]:
        choice_str = str(choice).strip()
        if 2 <= len(choice_str.split()) <= 5:
            augmented = f"{query} {choice_str}"[:200]
            jobs.append((augmented, 1))

    # Fire all jobs in parallel with a small stagger (0.3s between submissions).
    # Without staggering, all 5 DDG calls land simultaneously, which triggers
    # DDG's burst rate-limit after ~20-25 questions. A 0.3s stagger spreads
    # the burst over ~1.5s while keeping total wall-time far below sequential.
    results_by_job: dict = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for i, (q, n) in enumerate(jobs):
            if i > 0:
                _time.sleep(0.3)  # stagger submissions, not blocking — pool runs in background
            futures[pool.submit(web_search, q, n)] = i
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_job[idx] = future.result()
            except Exception as e:
                print(f"  [WebSearch] Job {idx} failed: {e}")
                results_by_job[idx] = []

    # Merge in job order so base query results come first
    all_results = []
    seen_bodies: set = set()
    for i in range(len(jobs)):
        for r in results_by_job.get(i, []):
            if r not in seen_bodies:
                seen_bodies.add(r)
                all_results.append(r)

    return all_results


def retriever_step(state, text_index, data, k=3, use_hybrid=True, use_cross_encoder=True):
    """
    Simplified retriever: Text + Web Search only.
    Question images are NOT processed here - they are passed directly
    to the solver via state.image_paths.

    Uses DuckDuckGo for web search (free, no API key needed).

    Retrieval pipeline per subquery:
      1. Hybrid local retrieval (dense FAISS + BM25 + RRF fusion) with query expansion
      2. Choice-augmented web search
      3. Cross-encoder reranking and web-first merge
    """
    # chain is just a logic step, it's almost the default in a way.
    # There's no LLM or tool call or it's not an agent it's just a chain.
    with tracer.start_as_current_span("Retriever", openinference_span_kind="retriever") as retriever_span:
        # Initialise BM25 on the text-only slice (reset index so that
        # FAISS row numbers and BM25 corpus positions align).
        text_data = data[data["media_type"] == "text"].reset_index(drop=True)

        # Initialize BM25 retriever once (outside the loop for efficiency)
        if use_hybrid:
            corpus_texts = text_data["text"].tolist()
            bm25_retriever = BM25Retriever(corpus_texts)

        retrieved = {}
        retrieval_k = k

        # Widen the web search budget for questions where the local KB is sparse
        # or the subject is a science domain likely to have thin KB coverage.
        #
        # ScienceQA:  subject == "natural science" fires the old heuristic.
        # MMMU:       subject is e.g. "Physics", "Biology", "Chemistry" — these
        #             never matched "natural science", so MMMU science questions
        #             never got the doubled budget even though they need it more
        #             (MMMU has no lecture/hint KB at all).
        #
        # New rule — double web_k when EITHER:
        #   (a) subject is a recognised science domain (covers both datasets), OR
        #   (b) the KB fields are all empty (catches any future dataset with no KB)
        _SCIENCE_SUBJECTS = {
            "natural science",  # ScienceQA label
            "physics",
            "biology",
            "chemistry",
            "basic_medical_science",
            "clinical_medicine",
            "diagnostics_and_laboratory_medicine",
            "pharmacy",
            "energy_and_power",
            "electronics",
            "materials",
            "mechanical_engineering",
            "architecture_and_engineering",
        }
        _subject = getattr(state, "subject", "").lower()
        _has_sparse_kb = not bool(getattr(state, "lecture", "") or getattr(state, "hint", ""))
        is_science_or_sparse = _subject in _SCIENCE_SUBJECTS or _has_sparse_kb
        web_k = retrieval_k * 2 if is_science_or_sparse else retrieval_k
        if is_science_or_sparse:
            print(f"  [Retriever] Science/sparse-KB question (subject='{_subject}') — web_k={web_k} (doubled)")

        # Per-subquery retrieval with query expansion
        # For each planner subquery, generate up to 2 rule-based paraphrases
        # and retrieve from the local KB using all variants.
        # Web search uses only the original query (choice-augmented) to avoid latency blowup.
        # Deduplication ensures the same chunk is never added twice.
        for subquery in state.subqueries:
            evidence = []
            local_docs = []

            seen_local: set = set()
            # Expand subquery into paraphrases for local KB retrieval only
            paraphrases = _expand_subquery(subquery)
            all_local_queries = [subquery] + paraphrases
            # Local corpus (hybrid or dense) — run on original + paraphrases
            for lq in all_local_queries:
                # Text embedding search
                try:
                    if use_hybrid:
                        # Hybrid: BM25 + Dense with RRF fusion
                        for doc in hybrid_retrieval(
                            query=lq,
                            text_index=text_index,
                            bm25_retriever=bm25_retriever,
                            data=data,
                            k=retrieval_k * 2,  # Get more candidates for reranking
                        ):
                            if doc not in seen_local:
                                seen_local.add(doc)
                                local_docs.append(doc)
                    else:
                        # Dense-only retrieval
                        q_embed = text_to_embedding(lq).reshape(1, -1)
                        D, I = text_index.search(q_embed.astype(np.float32), retrieval_k * 2)
                        for j in range(retrieval_k * 2):
                            idx = I[0][j]
                            if 0 <= idx < len(data):
                                doc = data.iloc[idx]["text"].strip()
                                if doc not in seen_local:
                                    seen_local.add(doc)
                                    local_docs.append(doc)
                except Exception as e:
                    print(f"Text retrieval failed for [{lq}]: {e}")
            evidence.extend(local_docs)

            if paraphrases:
                print(
                    f"  [Retriever] Query expansion: {len(paraphrases)} paraphrase(s) → {len(local_docs)} unique local docs"
                )

            # Web search: choice-augmented
            # Natural science uses web_k (2× retrieval_k) to compensate for
            # sparse KB coverage of specialised science
            try:
                web_docs = _web_search_with_choices(subquery, state.choices, k=web_k)
                evidence.extend(web_docs)
            except Exception as e:
                print(f"Web search failed for [{subquery}]: {e}")

            # Cross-encoder reranking — generalised web-first merge.

            # Design rationale:
            #   The system is designed as a general-purpose VQA reasoner, not a
            #   ScienceQA-specific one. The local KB may or may not contain
            #   content relevant to any given question. Blindly keeping top-N
            #   local docs regardless of relevance fills the evidence cap with
            #   noise and hurts citation precision.

            #   Strategy: web-first, then admit local docs only when the
            #   cross-encoder confirms they are relevant to this specific subquery
            #   (score > 0). This approach is domain-agnostic:
            #     - For questions well-covered by the KB (e.g. ScienceQA lecture
            #       content), the cross-encoder will score those chunks highly and
            #       they will be included.
            #     - For questions not covered by the KB (e.g. generic VQA), local
            #       docs will score <= 0 and be excluded, leaving web results to
            #       fill all evidence slots.

            #   top_k raised from (retrieval_k-1) to retrieval_k for web results
            if use_cross_encoder and evidence:
                local_set = set(local_docs)
                web_evidence = [e for e in evidence if e not in local_set]

                # Rerank web results — always the primary evidence signal
                reranked_web = rerank_with_cross_encoder(subquery, web_evidence, top_k=max(1, retrieval_k))

                # Start with web results; admit local docs that pass relevance filter
                merged = list(reranked_web)
                seen_set = set(merged)
                local_admitted = 0
                if local_docs:
                    # Cap raised from 5 → 15: with query expansion, local_docs can
                    # contain up to retrieval_k*2*(1+num_paraphrases) ≈ 18 unique docs.
                    # Scoring only 5 meant expansion candidates never reached the
                    # cross-encoder filter, negating most of the planner_hit benefit.
                    pairs = [[subquery, doc] for doc in local_docs[:15]]
                    try:
                        local_scores = cross_encoder.predict(pairs)
                        for doc, score in zip(local_docs[:15], local_scores):
                            if score > 0 and doc not in seen_set:
                                seen_set.add(doc)
                                merged.append(doc)
                                local_admitted += 1
                    except Exception as e:
                        print(f"  [Retriever] Local doc scoring failed: {e}")

                print(f"  [Retriever] Merge: {len(reranked_web)} web + {local_admitted} local (cross-encoder filtered)")
                evidence = merged

            retrieved[subquery] = ChunkInfo(text_chunks=evidence, image_rois=[])
            print(f"  Query [{subquery[:40]}...] -> {len(evidence)} chunks")

        state.retrieved_chunks = retrieved

        num_imgs = len([p for p in (state.image_paths or []) if p and os.path.exists(p)])
        print(f"  Question has {num_imgs} images (passed directly to solver)")

        retrieved_dict = {query: v.model_dump() for query, v in retrieved.items()}

        # use OpenInference semantic conventions
        retriever_span.set_attribute("retriever.queries", json.dumps(retrieved_dict))

        # Metrics
        recall_result = recall_at_k(state, k=retrieval_k)

        # Log to span
        retriever_span.set_attribute("retriever.recall", recall_result["recall"])

    return state
