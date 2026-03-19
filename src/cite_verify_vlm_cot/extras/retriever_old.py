import math
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import gc
from PIL import Image
import os
import json
from typing import List
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.experiments import run_experiment, evaluate_experiment
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
# CLIPProcessor handles image pre-processing like resizing and normalization
from planner import State

import phoenix as px
import os
# from phoenix.otel import register
# from openinference.instrumentation.openai import OpenAIInstrumentor
# from openinference.semconv.trace import SpanAttributes
# from opentelemetry.trace import Status, StatusCode
# from openinference.instrumentation import TracerProvider
# from opentelemetry import trace  # Required for span access

from tavily import TavilyClient
search = TavilyClient(api_key="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")

import torch
# Enable only the safe fallback backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# SDPA = Scaled Dot-Product Attention, core math behind attention
# PyTorch provides multiple implementations of this, called backends (FlashAttention, SDPA(default), Math)
# SDPA is fast but allocates large temporary buffers, especially bad for long sequences + big models
# Prefer FlashAttention (best) if your GPU supports it

import nest_asyncio # for running multiple calls asynchronously and simultaneously to speed up some of your evaluation
nest_asyncio.apply()

from evaluations import recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

from tracer import tracer

# # Phoenix is an application that can receive the traces that you're going to send from your agent here and then can visualize those in a UI
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

# Load cross-encoder (do this once, outside the function)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare a dataframe to store file path, media_type, text, and embeddings in
columns = ["media_type", "text", "embeddings", "roi_id", "bbox", "source_image"]
data = pd.DataFrame(columns=columns)

# https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117 (Method 2)
# helper functions
@tracer.chain()
def text_to_embedding(text):
    text = text.replace("\n", " ")
    embedding = text_model.encode(text, batch_size=1, normalize_embeddings=True)
    return embedding

@tracer.chain()
def image_embedding(image_path):
    """Generate CLIP embedding for an image."""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        # Already a PIL Image (e.g., from a patch)
        image = image_path
    
    # Prepare the image for the model
    # return_tensors="pt" specifies PyTorch tensors
    inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)
    
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    
    # Extract the tensor from BaseModelOutputWithPooling
    # get_image_features returns BaseModelOutputWithPooling with .pooler_output attribute
    image_features = outputs.pooler_output

    # Normalize the embeddings (important for accurate similarity comparisons later)
    image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)

    print("Image embeddings generated successfully.")
    print(image_embeddings.cpu().numpy().shape) # Should be (1, 512) for the base model
    
    return image_embeddings.cpu().numpy()

@tracer.chain()
def safe_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()

@tracer.chain()
def tavily_search(query, k=2):
    results = search.search(query=query, search_depth="advanced", max_results=k)
    return [r["content"] for r in results["results"]]

def crop_image_roi(image_path, patch=16):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    patches = []
    for y in range(0, 224, patch):
        for x in range(0, 224, patch):
            roi_np = img_np[y:y+patch, x:x+patch]
            roi = Image.fromarray(roi_np)
            patches.append({
                "roi": roi,
                "bbox": (x, y, x+patch, y+patch)
            })
    return patches

# https://blog.roboflow.com/image-search-engine-gaudi2/
# https://cookbook.openai.com/examples/custom_image_embedding_search
"""
ROI index → built from image patch/object embeddings → use CLIP text encoder
CLIP is explicitly trained for text–image alignment. SentenceTransformers are not.

SentenceTransformer for ROIs is a semantic mismatch
If your ROI index contains embeddings from:
- CLIP vision encoder
- ViT patches
- region crops
then querying it with SentenceTransformer text embeddings puts you in two different embedding spaces.
That leads to:
- weaker retrieval
-brittle similarity scores
- harder-to-debug failures
"""

def get_text_embedding(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(clip_model.device)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    
    # Extract tensor from BaseModelOutputWithPooling
    text_features = outputs.pooler_output
    
    # Normalize
    feats = text_features / text_features.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)  # (1,512)

# HYBRID RETRIEVER
class BM25Retriever:
    """BM25 sparse retriever for keyword-based search."""

    def __init__(self, corpus_texts: list):
        """
        Args:
            corpus_texts: List of document strings
        """
        self.corpus = corpus_texts
        # Tokenize corpus for BM25
        tokenized = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Retrieve top-k documents by BM25 score.

        Returns:
            List of (index, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_k_indices]

def dense_retrieval(query: str, text_index, text_data, k: int = 5) -> list:
    """
    Dense vector retrieval using FAISS index.

    Returns:
        List of (index, score) tuples
    """
    q_embed = text_to_embedding(query).reshape(1, -1)
    D, I = text_index.search(q_embed.astype(np.float32), k)
    # Note: FAISS returns distances, lower is better for L2, higher for IP
    # Convert to (index, score) format
    return [(I[0][i], float(D[0][i])) for i in range(k)]


def rrf_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion to combine dense and sparse results.

    Args:
        dense_results: List of (index, score) from dense retrieval
        sparse_results: List of (index, score) from BM25
        k: RRF constant (default 60)

    Returns:
        List of (index, combined_score) sorted by score descending
    """
    scores = {}

    # Add dense retrieval scores
    for rank, (idx, _) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Add sparse retrieval scores
    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by combined RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieval(query: str, text_index, bm25_retriever: BM25Retriever,
                     data, k: int = 5, dense_weight: float = 0.5) -> list:
    """
    Hybrid retrieval combining dense (FAISS) and sparse (BM25) search.

    Args:
        query: Search query string
        text_index: FAISS index for dense retrieval
        bm25_retriever: BM25Retriever instance
        data: DataFrame with document texts
        k: Number of results to return
        dense_weight: Not used with RRF, kept for API compatibility

    Returns:
        List of document text strings (top-k)
    """

    # Filter to text-only data
    text_data = data[data['media_type'] == 'text'].reset_index(drop=True)

    # Get more candidates than needed for fusion
    num_candidates = k * 3

    # Dense retrieval (uses text_data indices)
    dense_results = dense_retrieval(query, text_index, text_data, k=num_candidates)

    # Sparse retrieval (BM25 already built on text corpus)
    sparse_results = bm25_retriever.retrieve(query, k=num_candidates)

    # Fuse results using RRF
    fused_results = rrf_fusion(dense_results, sparse_results)

    # Return top-k document texts
    return [text_data.iloc[idx]["text"].strip() for idx, _ in fused_results[:k]]

# RERANKER
# Cosine similarity reranker (text-only)
from sentence_transformers.util import cos_sim

def rerank_by_similarity(subquery: str, evidences: List[str], model) -> List[str]:
    q_embed = model.encode(subquery, normalize_embeddings=True)
    e_embeds = model.encode(evidences, normalize_embeddings=True)

    scores = [cos_sim(q_embed, e)[0][0].item() for e in e_embeds]
    ranked = sorted(zip(evidences, scores), key=lambda x: x[1], reverse=True)
    return [e[0] for e in ranked]

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

def retrieve_image_rois(subquery, image_index, roi_metadata, k=2):
    q = get_text_embedding(subquery)          # (1,512)
    D, I = image_index.search(q, k)            # IP: higher is better

    rois = []
    for rank, idx in enumerate(I[0]):
        md = roi_metadata[int(idx)]
        rois.append({
            "roi_id": md["roi_id"],
            "bbox": md["bbox"],
            "source_image": md["source_image"],
            "score": float(D[0][rank])
        })
    return rois

# Perform Dual Retrieval at Inference

# At query time:
# 1. Embed your query in both formats:
# Text embedding → use bge or other text model → 384-dim
# Image embedding (if query includes image) → use CLIP → 512-dim
# 2. Search both indexes independently:
# 3. Normalize scores and merge (e.g., via weighted sum or ranking fusion):
# 4. Return top-N unique results across modalities.

# , image_index
def retriever_step(state: State, text_index, image_index, roi_metadata, data, k: int = 2, use_hybrid: bool = True, use_cross_encoder: bool = True) -> State:
    # chain is just a logic step, it's almost the default in a way. There's no LLM or tool call or it's not an agent it's just a chain.
    with tracer.start_as_current_span("Retriever", openinference_span_kind="retriever") as retriever_span:

        retrieved = {}

        # Initialize BM25 retriever once (outside the loop for efficiency)
        if use_hybrid:
            text_data = data[data['media_type'] == 'text']
            corpus_texts = text_data["text"].tolist()
            bm25_retriever = BM25Retriever(corpus_texts)
            # Store index mapping for BM25 results
            bm25_to_data_idx = text_data.index.tolist()

        for subquery in state.subqueries[1:]:
            evidence = []

            # Text embedding search
            try:
                if use_hybrid:
                    # Hybrid: BM25 + Dense with RRF fusion
                    local_docs = hybrid_retrieval(
                        query=subquery,
                        text_index=text_index,
                        bm25_retriever=bm25_retriever,
                        data=data,
                        k=k * 2  # Get more candidates for reranking
                    )
                    evidence.extend(local_docs)
                else:
                    # Dense-only retrieval
                    q_text_embed = text_to_embedding(subquery).reshape(1, -1)
                    D_text, I_text = text_index.search(q_text_embed, k * 2)
                    for j in range(k * 2):
                        idx = I_text[0][j]
                        evidence.append(data.iloc[idx]["text"].strip())
            except Exception as e:
                print(f"Text embedding failed: {subquery}\n{e}")

            # Tavily web search
            try:
                tavily_docs = tavily_search(subquery, k)
                evidence.extend(tavily_docs)
            except Exception as e:
                print(f"Tavily search failed: {subquery}\n{e}")

            if use_cross_encoder:
              evidence = rerank_with_cross_encoder(subquery, evidence, top_k=k)
            else:
              evidence = rerank_by_similarity(subquery, evidence, text_model)

            retrieved[subquery] = {
                "text_chunks": evidence,
                "image_rois": retrieve_image_rois(
                    subquery, image_index, roi_metadata, k=k
                )
            }


        state.retrieved_chunks = retrieved
        # use OpenInference semantic conventions
        print(retrieved)
        retriever_span.set_attribute("retriever.queries", json.dumps(retrieved))

        # In retriever_step, after state.retrieved_chunks = retrieved
        # planner_hit = any(
        #     state.gold_answer and state.gold_answer.lower() in ev.lower()
        #     for evidence_dict in state.retrieved_chunks.values()
        #     for ev in evidence_dict.get("text_chunks", [])
        # )
        # retriever_span.set_attribute("retriever.planner_hit", planner_hit)

        # Evaluation
        # Compute metrics
        recall_result = recall_at_k(state, k=k)
        p_at_k = precision_at_k(state, k=k)
        mrr_score = mean_reciprocal_rank(state)
        ndcg_score = ndcg_at_k(state, k=k)
        planner_hit = planner_hit_rate(state, k=k)

        # Log to span
        retriever_span.set_attribute("retriever.recall", recall_result["recall"])
        retriever_span.set_attribute("retriever.precision", p_at_k)
        retriever_span.set_attribute("retriever.mrr", mrr_score)
        retriever_span.set_attribute("retriever.ndcg", ndcg_score)
        retriever_span.set_attribute("retriever.planner_hit", planner_hit)
        retriever_span.set_attribute("retriever.recall_hits", json.dumps(recall_result["hits"]))

    return state


# store image embeddings as well - Multimodal Querying : If your future queries (from planner or user) are visual or multimodal (e.g., visual question + region crop), you’ll need an image-to-image or image-to-text match. This requires image embeddings.
df = pd.read_csv("scienceqa_augmented_100.csv")

roi_vectors = []
roi_metadata = []

for idx, row in df.iterrows():
    hint = safe_str(row.get("hint"))
    lecture = safe_str(row.get("lecture"))
    img_caption = safe_str(row.get("img_caption"))
    img_ocr = safe_str(row.get("img_ocr"))

    text = """
        hint: {hint}
        lecture: {lecture}
    """
    text = text.format(
        hint=hint,
        lecture=lecture
    )
    media_type="text"
    embedding=text_to_embedding(text)
    text_row={'media_type':media_type,
                'text' : text,
                'embeddings': embedding.tolist()}
    data = pd.concat([data, pd.DataFrame([text_row])], ignore_index=True)

    torch.cuda.empty_cache()
    gc.collect()

    # Parse image_paths from JSON string
    image_paths_str = safe_str(row.get("image_paths", "[]"))
    try:
        # Try to parse as JSON
        image_paths = json.loads(image_paths_str)
    except json.JSONDecodeError:
        try:
            # Fallback to ast.literal_eval for Python list strings
            image_paths = ast.literal_eval(image_paths_str)
        except:
            print(f"Could not parse image_paths: {image_paths_str}")
            image_paths = []

    # Skip if no images
    if not image_paths or len(image_paths) == 0:
        print(f"Row {idx}: No images to process")
        continue

    print(f"image path: {image_paths}")

    # Process each image in the list
    for image_path in image_paths:
        # Skip if path is missing or image file doesn't exist
        if not image_path or not os.path.exists(image_path):
            print(f"Skipping missing file: {image_path}")
            continue

        print(f"Processing image path: {image_path}")

        patches = crop_image_roi(image_path, 16)
        for patch in patches:
            p_embedding = image_embedding(patch["roi"])
            roi_vectors.append(p_embedding[0])
            roi_metadata.append({
                "roi_id": f"{os.path.basename(image_path)}_{patch['bbox']}",
                "bbox": patch["bbox"],
                "source_image": image_path
            })
            media_type = "image"
            image_row = {
                "media_type": "image",
                "roi_id": f"{os.path.basename(image_path)}_{patch['bbox']}",
                "source_image": image_path,
                "bbox": str(patch["bbox"]),
                "text": img_caption,  # You might want to parse this JSON too
                "embeddings": p_embedding[0].tolist()
            }

            data = pd.concat([data, pd.DataFrame([image_row])], ignore_index=True)

        torch.cuda.empty_cache()
        gc.collect()

data.to_csv("multimodal_embeddings.csv", index=False)

# Maintain separate FAISS indexes for text and image
text_data = data[data['media_type'] == 'text']
image_data = data[data['media_type'] == 'image']

# Check if we have data before creating indexes
if len(text_data) > 0:
    text_vectors = np.vstack(text_data['embeddings'].values)
    text_index = faiss.IndexFlatIP(text_vectors.shape[1])
    text_index.add(text_vectors.astype(np.float32))
    faiss.write_index(text_index, "text_index.faiss")
    print(f"Created text index with {len(text_data)} entries")
else:
    print("Warning: No text data to index")

if len(image_data) > 0:
    image_vectors = np.vstack(image_data['embeddings'].values)
    image_index = faiss.IndexFlatIP(image_vectors.shape[1])
    image_index.add(image_vectors.astype(np.float32))
    faiss.write_index(image_index, "image_index.faiss")
    print(f"Created image index with {len(image_data)} entries")
else:
    print("Warning: No image data to index")