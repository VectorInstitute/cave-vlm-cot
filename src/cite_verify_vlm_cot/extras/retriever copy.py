import math
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import gc
from PIL import Image
import os
import json
import ast
from typing import List, Tuple
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.experiments import run_experiment, evaluate_experiment
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
import hashlib
import base64
from io import BytesIO

import phoenix as px
import os

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ before any key checks

from tavily import TavilyClient
_tavily_key = os.environ.get("TAVILY_API_KEY", "")
if not _tavily_key:
    raise EnvironmentError("TAVILY_API_KEY is not set. Export it before running.")
search = TavilyClient(api_key=_tavily_key)

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import nest_asyncio
nest_asyncio.apply()

from evaluations import recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k, planner_hit_rate, planner_coverage_score
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from tracer import tracer
from utils import State, RoiInfo, ChunkInfo, safe_parse_json, safe_str

# Load cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare a dataframe to store file path, media_type, text, and embeddings in
columns = ["media_type", "text", "embeddings", "roi_id", "bbox", "source_image"]
data = pd.DataFrame(columns=columns)

# Helper functions
@tracer.chain()
def text_to_embedding(text):
    text = text.replace("\n", " ")
    embedding = text_model.encode(text, batch_size=1, normalize_embeddings=True)
    return embedding

@tracer.chain()
def image_embedding(image_path):
    """Generate CLIP embedding for an image in the shared image-text space."""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)
    
    with torch.no_grad():
        # Get image features - handle different return types
        output = clip_model.get_image_features(**inputs)
        
        # Handle case where output is a ModelOutput object vs raw tensor
        if isinstance(output, torch.Tensor):
            image_features = output
        elif hasattr(output, 'pooler_output'):
            image_features = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            # Use CLS token if pooler_output not available
            image_features = output.last_hidden_state[:, 0, :]
        else:
            # Try to convert directly
            image_features = torch.tensor(output)
    
    # Normalize the embeddings
    image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_embeddings.cpu().numpy()

@tracer.chain()
def tavily_search(query, k=2):
    results = search.search(query=query, search_depth="advanced", max_results=k)
    return [r["content"] for r in results["results"]]

# Grounding DINO constants
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_RUN_SIZE = 800          # upsample 224px → 800px for richer feature maps
GDINO_BOX_THRESHOLD  = 0.25
GDINO_TEXT_THRESHOLD = 0.20

SUBJECT_PROMPTS = {
    "natural science":  "organism. cell. plant. animal. organ. tissue. diagram. label. arrow. chart. graph. map. molecule. atom. force. energy. wave. circuit. magnet.",
    "social science":   "map. region. border. country. city. river. population. chart. graph. table. label. flag. legend.",
    "language science": "text. sentence. word. diagram. table. chart. label.",
}
DEFAULT_PROMPT = "object. region. label. diagram. chart. arrow. table. graph."

MAX_ROIS       = 6
WHITE_THRESH   = 0.85
IOU_NMS_THRESH = 0.40
CC_MIN_AREA    = 150
CC_PADDING     = 8
TARGET_SIZE    = 112

# Lazy GDINO loader (only used during indexing)
_gdino_model     = None
_gdino_processor = None

def _get_gdino():
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        print("Loading Grounding DINO (first call)...")
        _gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID)
        _gdino_model.eval()
        if torch.cuda.is_available():
            _gdino_model = _gdino_model.cuda()
        print("Grounding DINO loaded.")
    return _gdino_model, _gdino_processor

# Shared helpers
def _patch_to_b64(roi_img: Image.Image) -> str:
    buf = BytesIO()
    roi_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _is_blank(patch_arr: np.ndarray, thresh: float = WHITE_THRESH) -> bool:
    return (patch_arr > 240).all(axis=2).mean() > thresh

def _iou(a: Tuple, b: Tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def _nms(rois: List[dict], thresh: float = IOU_NMS_THRESH) -> List[dict]:
    keep = []
    for roi in rois:
        box = tuple(roi["bbox"])
        if not any(_iou(box, tuple(k["bbox"])) > thresh for k in keep):
            keep.append(roi)
    return keep

def classify_image_type(arr: np.ndarray) -> str:
    white_pct = (arr > 240).all(axis=2).mean()
    std       = arr.std()
    unique    = len(np.unique(arr.reshape(-1, 3), axis=0))
    if white_pct > 0.45 and unique < 6000:
        return "diagram_text"
    elif white_pct < 0.12 and std > 45:
        return "natural_photo"
    else:
        return "complex_visual"

def extract_cc_rois(img_arr: np.ndarray) -> List[dict]:
    gray   = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    H, W = img_arr.shape[:2]
    rois = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < CC_MIN_AREA:
            continue
        x = max(0, stats[i, cv2.CC_STAT_LEFT]  - CC_PADDING)
        y = max(0, stats[i, cv2.CC_STAT_TOP]   - CC_PADDING)
        w = min(W - x, stats[i, cv2.CC_STAT_WIDTH]  + 2 * CC_PADDING)
        h = min(H - y, stats[i, cv2.CC_STAT_HEIGHT] + 2 * CC_PADDING)
        patch = img_arr[y:y+h, x:x+w]
        if _is_blank(patch):
            continue
        roi_img = Image.fromarray(patch).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        rois.append({"bbox": [x, y, x+w, y+h], "image_patch": _patch_to_b64(roi_img),
                     "area": stats[i, cv2.CC_STAT_AREA]})
    rois.sort(key=lambda r: r["area"], reverse=True)
    return rois

"""
Updated SUBJECT_PROMPTS covering ALL ScienceQA subjects/topics.
Two-tier lookup: topic-level (specific) → subject-level (broad) → DEFAULT_PROMPT.
Topic keys are normalised to lowercase with hyphens (matching ScienceQA's topic field).
"""

# Tier 2: topic-specific prompts (checked first)
TOPIC_PROMPTS = {
    # Natural Science topics
    "biology":
        "organism. cell. plant. animal. organ. tissue. life cycle. habitat. "
        "food chain. species. diagram. label. arrow. ecosystem. DNA. gene. "
        "trait. inherited. offspring. classification. vertebrate. invertebrate.",
    "physics":
        "force. energy. wave. circuit. magnet. diagram. arrow. graph. motion. "
        "velocity. acceleration. lens. mirror. light. sound. heat. temperature. "
        "friction. gravity. pulley. lever. inclined plane.",
    "earth-science":
        "rock. mineral. layer. fossil. volcano. weather. map. diagram. erosion. "
        "plate. earthquake. cloud. precipitation. water cycle. soil. atmosphere. "
        "ocean. continent. landform.",
    "chemistry":
        "molecule. atom. element. reaction. diagram. table. bond. solution. "
        "periodic table. mixture. compound. solid. liquid. gas. phase change. "
        "chemical equation. acid. base. pH.",
    "science-and-engineering-practices":
        "experiment. diagram. chart. graph. table. variable. measurement. data. "
        "hypothesis. control. observation. result. conclusion. bar graph. line graph.",
    "units-and-measurement":
        "ruler. scale. thermometer. graduated cylinder. unit. meter. gram. liter. "
        "measurement. chart. table. conversion. estimate.",

    # Social Science topics
    "geography":
        "map. region. border. country. state. city. river. mountain. compass. "
        "legend. scale. continent. ocean. latitude. longitude. cardinal direction. "
        "hemisphere. island. peninsula. climate zone.",
    "us-history":
        "map. timeline. document. flag. artifact. region. territory. colony. "
        "revolution. constitution. president. war. amendment. landmark.",
    "world-history":
        "map. timeline. artifact. empire. civilization. trade route. monument. "
        "document. war. treaty. dynasty. ancient. medieval.",
    "civics":
        "government. law. constitution. vote. election. court. rights. citizen. "
        "diagram. chart. branch. congress. president. amendment.",
    "economics":
        "graph. chart. table. supply. demand. curve. price. market. money. "
        "trade. goods. services. producer. consumer. scarcity. opportunity cost.",
    "global-studies":
        "map. region. country. culture. flag. population. chart. graph. "
        "continent. trade. resource. climate. religion. language.",

    # Language Science topics
    "figurative-language":
        "text. sentence. word. paragraph. label.",
    "writing-strategies":
        "text. sentence. paragraph. word. label. table.",
    "vocabulary":
        "text. word. sentence. definition. context.",
    "grammar":
        "text. sentence. word. punctuation. diagram.",
    "verbs":
        "text. sentence. word.",
    "capitalization":
        "text. sentence. word. letter.",
    "punctuation":
        "text. sentence. word. punctuation mark.",
    "phonological-awareness":
        "text. word. syllable. sound. letter.",
    "reference-skills":
        "table. chart. dictionary. index. glossary. map. diagram. label. "
        "guide. reference. key. entry.",
}

# Tier 1: broad subject-level prompts (fallback)
SUBJECT_PROMPTS = {
    "natural science":
        "organism. cell. plant. animal. organ. tissue. diagram. label. arrow. "
        "chart. graph. map. molecule. atom. force. energy. wave. circuit. magnet. "
        "rock. mineral. weather. ecosystem. experiment. measurement.",
    "social science":
        "map. region. border. country. city. river. population. chart. graph. "
        "table. label. flag. legend. timeline. artifact. document. government.",
    "language science":
        "text. sentence. word. diagram. table. chart. label. paragraph. "
        "punctuation. dictionary. reference.",
}

# Default
DEFAULT_PROMPT = "object. region. label. diagram. chart. arrow. table. graph. text. map."

def get_gdino_prompt(subject: str = "", topic: str = "") -> str:
    """
    Resolve the best Grounding DINO text prompt for a given subject/topic.

    Priority: topic-specific → subject-level → default.
    Normalises keys to lowercase-hyphenated form to match ScienceQA conventions.
    """
    topic_key = topic.lower().strip().replace(" ", "-").replace("_", "-")
    subject_key = subject.lower().strip()

    prompt = (
        TOPIC_PROMPTS.get(topic_key)
        or SUBJECT_PROMPTS.get(subject_key)
        or DEFAULT_PROMPT
    )
    return prompt

def extract_gdino_rois(img_pil, img_arr, subject: str = "", topic: str = ""):
    """
    Extract ROIs using Grounding DINO with subject/topic-aware prompting.
    """
    from retriever import (          # adjust import path as needed
        _get_gdino,
        _is_blank, _patch_to_b64,
        GDINO_RUN_SIZE, TARGET_SIZE,
    )
    import numpy as np
    from PIL import Image as PILImage

    # two-tier prompt resolution
    prompt = get_gdino_prompt(subject, topic)

    # Optionally append the raw topic string if it's not already covered
    if topic:
        topic_clean = topic.lower().replace("(", "").replace(")", "").strip()
        if topic_clean not in prompt.lower():
            prompt = f"{prompt} {topic_clean}."
    prompt = prompt[:300]

    img_high = img_pil.resize((GDINO_RUN_SIZE, GDINO_RUN_SIZE), PILImage.LANCZOS)
    scale = 224.0 / GDINO_RUN_SIZE

    model, processor = _get_gdino()
    device = next(model.parameters()).device
    inputs = processor(images=img_high, text=prompt, return_tensors="pt").to(device)

    import torch
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        target_sizes=[(GDINO_RUN_SIZE, GDINO_RUN_SIZE)],
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    boxes, scores = boxes[order], scores[order]

    rois = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v * scale) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(224, x2), min(224, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = img_arr[y1:y2, x1:x2]
        if _is_blank(patch):
            continue
        roi_img = PILImage.fromarray(patch).resize(
            (TARGET_SIZE, TARGET_SIZE), PILImage.LANCZOS
        )
        rois.append({
            "bbox": [x1, y1, x2, y2],
            "image_patch": _patch_to_b64(roi_img),
            "gdino_score": float(score),
        })
    return rois

def extract_rois(image_path: str, subject: str = "", topic: str = "") -> List[dict]:
    img_pil  = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)
    img_arr  = np.array(img_pil)
    img_type = classify_image_type(img_arr)
    fname    = os.path.basename(image_path)
    print(f"  [{fname}] type={img_type}")

    raw_rois = []
    if img_type == "diagram_text":
        raw_rois = extract_cc_rois(img_arr)
        if not raw_rois:
            print(f"  [{fname}] CC found nothing → trying GDINO")
            raw_rois = extract_gdino_rois(img_pil, img_arr, subject, topic)
    else:
        raw_rois = extract_gdino_rois(img_pil, img_arr, subject, topic)
        if not raw_rois:
            print(f"  [{fname}] GDINO found nothing → trying CC")
            raw_rois = extract_cc_rois(img_arr)

    if not raw_rois:
        print(f"  [{fname}] both methods found nothing → skipping image")
        return []

    # shared post-processing
    # REPLACE the filtered = [...] block with:
    filtered = []
    for r in raw_rois:
        x1, y1, x2, y2 = int(r["bbox"][0]), int(r["bbox"][1]), int(r["bbox"][2]), int(r["bbox"][3])
        patch = img_arr[y1:y2, x1:x2]
        if patch.size > 0 and not _is_blank(patch):
            filtered.append(r)
    deduped = _nms(filtered)
    final   = deduped[:MAX_ROIS]

    result = []
    for roi in final:
        # In extract_rois(), change the result.append(...) to:
        result.append({
            "roi_id":            f"{fname}_{tuple(roi['bbox'])}",
            "bbox":              roi["bbox"],
            "image_patch":       roi["image_patch"],
            "source_image_path": image_path,
            "subject":           subject,
            "topic":             topic,
        })
    print(f"  [{fname}] → {len(result)} RoIs")
    return result

def get_text_embedding(text: str) -> np.ndarray:
    """Generate CLIP text embedding in the shared image-text space."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(clip_model.device)
    with torch.no_grad():
        # Get text features - handle different return types
        output = clip_model.get_text_features(**inputs)
        
        # Handle case where output is a ModelOutput object vs raw tensor
        if isinstance(output, torch.Tensor):
            text_features = output
        elif hasattr(output, 'pooler_output'):
            text_features = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            text_features = output.last_hidden_state[:, 0, :]
        else:
            text_features = torch.tensor(output)
    
    # Normalize
    feats = text_features / text_features.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32) 

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

def dense_retrieval(query: str, text_index, text_data, k: int = 5) -> list:
    q_embed = text_to_embedding(query).reshape(1, -1)
    D, I = text_index.search(q_embed.astype(np.float32), k)
    return [(I[0][i], float(D[0][i])) for i in range(k)]

def rrf_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    scores = {}

    for rank, (idx, _) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieval(query: str, text_index, bm25_retriever: BM25Retriever,
                     data, k: int = 5, dense_weight: float = 0.5) -> list:
    text_data = data[data['media_type'] == 'text'].reset_index(drop=True)
    num_candidates = k * 3

    dense_results = dense_retrieval(query, text_index, text_data, k=num_candidates)
    sparse_results = bm25_retriever.retrieve(query, k=num_candidates)
    fused_results = rrf_fusion(dense_results, sparse_results)

    return [text_data.iloc[idx]["text"].strip() for idx, _ in fused_results[:k]]

# RERANKER
from sentence_transformers.util import cos_sim

def rerank_by_similarity(subquery: str, evidences: List[str], model) -> List[str]:
    q_embed = model.encode(subquery, normalize_embeddings=True)
    e_embeds = model.encode(evidences, normalize_embeddings=True)

    scores = [cos_sim(q_embed, e)[0][0].item() for e in e_embeds]
    ranked = sorted(zip(evidences, scores), key=lambda x: x[1], reverse=True)
    return [e[0] for e in ranked]

def rerank_with_cross_encoder(query: str, documents: list, top_k: int = 2) -> list:
    """Rerank documents using cross-encoder."""
    if not documents:
        return []

    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]

def retrieve_image_rois(subquery, image_index, roi_metadata, k=2):
    """Retrieve top-k image ROIs for a subquery."""
    
    if not roi_metadata or len(roi_metadata) == 0:
        print(f"Warning: No ROI metadata available, skipping image ROI retrieval")
        return []
    
    q = get_text_embedding(subquery)
    D, I = image_index.search(q, k)

    rois = []
    for rank, idx in enumerate(I[0]):
        if idx >= len(roi_metadata):
            print(f"Warning: Index {idx} out of range for roi_metadata (len={len(roi_metadata)})")
            continue
            
        md = roi_metadata[int(idx)]
        # Create RoiInfo objects instead of dicts
        roi = RoiInfo(
            roi_id=md["roi_id"],
            bbox=list(md["bbox"]) if isinstance(md["bbox"], tuple) else md["bbox"],
            source_image=md["source_image_path"],  # Will be accessed as roi.source_image
            image_patch=md["image_patch"],
            caption=md.get("caption", ""),
            score=float(D[0][rank]),
            subject=md.get("subject", ""),
            topic=md.get("topic", "") 
        )
        rois.append(roi)
    return rois

class HybridROIRetriever:
    """
    Hybrid ROI retrieval combining:
    1. Text query matching (semantic)
    2. Visual similarity to input image
    3. Caption-based matching
    4. Cross-encoder reranking
    """
    
    def __init__(
        self,
        roi_index,        # FAISS index of ROI embeddings
        roi_metadata,     # List of ROI info dicts
        clip_model,
        clip_processor,
        text_model,       # SentenceTransformer
        reranker=None,    # CrossEncoder
    ):
        self.roi_index = roi_index
        self.roi_metadata = roi_metadata
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.text_model = text_model
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        input_image: "Image.Image",
        input_image_caption: str = "",
        k: int = 5,
        method: str = "hybrid",  # "text", "visual", "caption", "hybrid"
        filter_subject: str = None,
        filter_topic: str = None,
    ) -> list:
        """Retrieve relevant ROIs from the global database."""
        
        if method == "text":
            return self._retrieve_by_text(query, k * 2, filter_subject, filter_topic)[:k]
        
        elif method == "visual":
            return self._retrieve_by_visual(input_image, k * 2, filter_subject, filter_topic)[:k]
        
        elif method == "caption":
            return self._retrieve_by_caption(query, input_image_caption, k * 2, filter_subject, filter_topic)[:k]
        
        else:  # hybrid
            # Get candidates from multiple methods
            text_results = self._retrieve_by_text(query, k * 3, filter_subject, filter_topic)
            # Only do visual retrieval when we actually have an input image
            if input_image is not None:
                visual_results = self._retrieve_by_visual(input_image, k * 3, filter_subject, filter_topic)
            else:
                visual_results = []
            
            # Merge and deduplicate
            seen_ids = set()
            candidates = []
            
            for roi in text_results + visual_results:
                roi_id = roi.get('roi_id', str(roi.get('bbox')))
                if roi_id not in seen_ids:
                    seen_ids.add(roi_id)
                    candidates.append(roi)
            
            # Rerank with combined scoring
            candidates = self._rerank_hybrid(candidates, query, input_image)
            
            return candidates[:k]

    @staticmethod
    def _normalize_roi(roi: dict) -> dict:
        """Ensure roi dict has the field names RoiInfo expects.
        roi_metadata stores source_image_path; RoiInfo.source_image is the required field name.
        """
        if 'source_image' not in roi and 'source_image_path' in roi:
            roi['source_image'] = roi['source_image_path']
        return roi
    
    def _retrieve_by_text(self, query: str, k: int, filter_subject: str = None, filter_topic: str = None) -> list:
        """Text-based retrieval using CLIP text encoder."""
        import torch
        
        device = next(self.clip_model.parameters()).device
        
        with torch.no_grad():
            inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
            text_out = self.clip_model.get_text_features(**inputs)
            # get_text_features() may return a tensor directly (CLIPModel) or a
            # BaseModelOutputWithPooling struct (base transformer). Handle both.
            text_emb = text_out if isinstance(text_out, torch.Tensor) else (
                text_out.pooler_output if text_out.pooler_output is not None
                else text_out.last_hidden_state[:, 0]
            )
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        query_np = text_emb.cpu().numpy().astype('float32')
        k_retrieve = k * 3 if filter_subject or filter_topic else k
        scores, indices = self.roi_index.search(query_np, k_retrieve)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.roi_metadata):
                roi = self._normalize_roi(self.roi_metadata[idx].copy())
                
                # Apply subject/topic filtering
                if filter_subject and roi.get('subject', '') != filter_subject:
                    continue
                if filter_topic and roi.get('topic', '') != filter_topic:
                    continue
                
                roi['text_score'] = float(score)
                roi['score'] = float(score)
                results.append(roi)
                
                # Stop when we have enough filtered results
                if len(results) >= k:
                    break
        
        return results
    
    def _retrieve_by_visual(self, input_image: "Image.Image", k: int, filter_subject: str = None,
    filter_topic: str = None) -> list:
        """Visual retrieval - find ROIs similar to input image."""
        import torch
        
        device = next(self.clip_model.parameters()).device
        
        with torch.no_grad():
            inputs = self.clip_processor(images=input_image, return_tensors="pt").to(device)
            img_out = self.clip_model.get_image_features(**inputs)
            # get_image_features() may return a tensor or BaseModelOutputWithPooling
            img_emb = img_out if isinstance(img_out, torch.Tensor) else (
                img_out.pooler_output if img_out.pooler_output is not None
                else img_out.last_hidden_state[:, 0]
            )
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        query_np = img_emb.cpu().numpy().astype('float32')
        k_retrieve = k * 3 if (filter_subject or filter_topic) else k
        scores, indices = self.roi_index.search(query_np, k_retrieve)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.roi_metadata):
                roi = self._normalize_roi(self.roi_metadata[idx].copy())

                # Apply subject/topic filtering (mirrors _retrieve_by_text)
                if filter_subject and roi.get('subject', '') != filter_subject:
                    continue
                if filter_topic and roi.get('topic', '') != filter_topic:
                    continue

                roi['visual_score'] = float(score)
                roi['score'] = float(score)
                results.append(roi)

                if len(results) >= k:
                    break

        return results
    
    def _retrieve_by_caption(self, query: str, input_caption: str, k: int, filter_subject: str = None,
    filter_topic: str = None) -> list:
        """Caption-based retrieval using text embeddings."""
        
        combined = f"{query}. Image shows: {input_caption}" if input_caption else query
        query_emb = self.text_model.encode(combined, normalize_embeddings=True)
        
        # Filter metadata before scoring if filters are set — avoids scoring
        # the entire corpus when we only care about one subject/topic
        candidates = []
        candidate_indices = []
        for i, r in enumerate(self.roi_metadata):
            if filter_subject and r.get('subject', '') != filter_subject:
                continue
            if filter_topic and r.get('topic', '') != filter_topic:
                continue
            candidates.append(r)
            candidate_indices.append(i)

        if not candidates:
            return []

        # when caption is empty (ROIs from image-less corpus entries), fall back
        # to topic then subject as a proxy — avoids encoding hundreds of empty strings
        # which would make all scores identical and ranking meaningless.
        def _effective_caption(r: dict) -> str:
            return r.get('caption') or r.get('topic') or r.get('subject') or ''

        captions = [_effective_caption(r) for r in candidates]
        caption_embs = self.text_model.encode(
            captions, normalize_embeddings=True, show_progress_bar=False
        )

        scores = np.dot(caption_embs, query_emb)
        top_k = np.argsort(scores)[-k:][::-1]

        results = []
        for local_idx in top_k:
            roi = self._normalize_roi(candidates[local_idx].copy())
            roi['caption_score'] = float(scores[local_idx])
            roi['score'] = float(scores[local_idx])
            results.append(roi)

        return results
    
    def _rerank_hybrid(self, candidates: list, query: str, input_image: "Image.Image") -> list:
        """Rerank candidates using combined text + visual scoring."""
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        
        device = next(self.clip_model.parameters()).device
        
        # skip visual embedding if no input image is available
        input_emb = None
        if input_image is not None:
            # Get input image embedding
            with torch.no_grad():
                img_inputs = self.clip_processor(images=input_image, return_tensors="pt").to(device)
                _img_out = self.clip_model.get_image_features(**img_inputs)
                input_emb = _img_out if isinstance(_img_out, torch.Tensor) else (
                    _img_out.pooler_output if _img_out.pooler_output is not None
                    else _img_out.last_hidden_state[:, 0]
                )
                input_emb = input_emb / input_emb.norm(dim=-1, keepdim=True)
        
        for roi in candidates:
            # Compute visual similarity to input image
            if input_emb is not None and roi.get('image_patch'):
                try:
                    img_data = base64.b64decode(roi['image_patch'])
                    roi_img = Image.open(BytesIO(img_data)).convert('RGB')
                    
                    with torch.no_grad():
                        roi_inputs = self.clip_processor(images=roi_img, return_tensors="pt").to(device)
                        _roi_out = self.clip_model.get_image_features(**roi_inputs)
                        roi_emb = _roi_out if isinstance(_roi_out, torch.Tensor) else (
                            _roi_out.pooler_output if _roi_out.pooler_output is not None
                            else _roi_out.last_hidden_state[:, 0]
                        )
                        roi_emb = roi_emb / roi_emb.norm(dim=-1, keepdim=True)
                        
                        visual_sim = torch.matmul(input_emb, roi_emb.T).item()
                        roi['visual_score'] = visual_sim
                except:
                    roi['visual_score'] = roi.get('visual_score', 0)
            
            # Combined score: text + visual + caption
            text_s = roi.get('text_score', 0)
            visual_s = roi.get('visual_score', 0)
            caption_s = roi.get('caption_score', 0)
            
            # Weighted combination
            roi['score'] = 0.15 * text_s + 0.65 * visual_s + 0.2 * caption_s
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates

def dict_to_roi_info(roi_dict: dict) -> RoiInfo:
    """Convert roi dict from HybridROIRetriever to RoiInfo object."""
    return RoiInfo(
        roi_id=roi_dict.get('roi_id', ''),
        bbox=roi_dict.get('bbox', []),
        source_image=roi_dict.get('source_image', ''),
        image_patch=roi_dict.get('image_patch', ''),
        caption=roi_dict.get('caption', ''),
        score=roi_dict.get('score', 0.0),
        subject=roi_dict.get('subject', ''),
        topic=roi_dict.get('topic', '')
    )

def retriever_step(state: State, text_index, image_index, roi_metadata, data, k: int = 2, use_hybrid: bool = True, use_cross_encoder: bool = True) -> State:
    with tracer.start_as_current_span("Retriever", openinference_span_kind="retriever") as retriever_span:

        retrieved = {}
        # Increase k for low-coverage queries
        # If planner coverage is low, retrieve more docs to compensate
        coverage = planner_coverage_score(state) if hasattr(state, 'subqueries') and state.subqueries else 1.0
        
        if coverage < 0.4:
            # Low coverage → retrieve more docs
            retrieval_k = k + 2  # e.g., 5 → 7
            print(f"Low coverage ({coverage:.1%}), increasing k to {retrieval_k}")
        else:
            retrieval_k = k

        # Initialize BM25 retriever once
        if use_hybrid:
            text_data = data[data['media_type'] == 'text']
            corpus_texts = text_data["text"].tolist()
            bm25_retriever = BM25Retriever(corpus_texts)

        # Initialize hybrid retriever
        roi_retriever = HybridROIRetriever(
            roi_index=image_index,
            roi_metadata=roi_metadata,
            clip_model=clip_model,
            clip_processor=clip_processor,
            text_model=text_model,
            reranker=cross_encoder,
        )

        # resolve input_image and input_caption from state (were previously undefined variables)
        input_image = None
        input_caption = ""
        for img_path in (state.image_paths or []):
            if img_path and os.path.exists(img_path):
                try:
                    input_image = Image.open(img_path).convert("RGB")
                    img_filename = os.path.basename(img_path)
                    input_caption = (state.img_captions or {}).get(img_filename, "")
                    break
                except Exception as e:
                    print(f"Warning: Could not load question image {img_path}: {e}")

        for subquery in state.subqueries:
            evidence = []

            # Text embedding search
            local_docs = []  # track local corpus docs separately for min_local guarantee
            try:
                if use_hybrid:
                    local_docs = hybrid_retrieval(
                        query=subquery,
                        text_index=text_index,
                        bm25_retriever=bm25_retriever,
                        data=data,
                        k=retrieval_k * 2
                    )
                    evidence.extend(local_docs)
                else:
                    q_text_embed = text_to_embedding(subquery).reshape(1, -1)
                    D_text, I_text = text_index.search(q_text_embed, retrieval_k * 2)
                    for j in range(retrieval_k * 2):
                        idx = I_text[0][j]
                        doc = data.iloc[idx]["text"].strip()
                        local_docs.append(doc)
                        evidence.append(doc)
            except Exception as e:
                print(f"Text embedding failed: {subquery}\n{e}")

            # Tavily web search
            try:
                tavily_docs = cached_tavily_search(subquery, retrieval_k)
                evidence.extend(tavily_docs)
            except Exception as e:
                print(f"Tavily search failed: {subquery}\n{e}")

            if use_cross_encoder:
                # guarantee at least one local corpus result survives reranking.
                # Tavily web results often score higher than lecture corpus chunks, pushing
                # gold-answer text out of the top-k entirely. We reserve one slot for the
                # best local result before the cross-encoder runs on the rest.
                local_set = set(local_docs)
                web_evidence = [e for e in evidence if e not in local_set]
                # Rerank web evidence normally
                reranked_web = rerank_with_cross_encoder(subquery, web_evidence, top_k=max(1, retrieval_k - 1))
                # Keep best local result (already ranked by hybrid/BM25 — take first)
                best_local = local_docs[:1]
                # Merge: local first, then reranked web (dedup)
                seen = set(best_local)
                merged = list(best_local)
                for e in reranked_web:
                    if e not in seen:
                        seen.add(e)
                        merged.append(e)
                evidence = merged
            else:
                evidence = rerank_by_similarity(subquery, evidence, text_model)

            # Image ROI retrieval - use hybrid when image is available, caption+text otherwise
            # initialize before conditional so ChunkInfo() never hits a NameError
            image_rois = []
            if roi_metadata and len(roi_metadata) > 0:
                filter_subject = state.subject
                if input_image is not None:
                    # Full hybrid: text query + visual similarity
                    roi_dicts = roi_retriever.retrieve(
                        query=subquery,
                        input_image=input_image,
                        input_image_caption=input_caption,
                        k=5,
                        method="hybrid",
                        filter_subject=filter_subject, 
                    )
                else:
                    # No question image — use caption-based retrieval (still leverages
                    # both the subquery text and any available captions, better than
                    # pure text-only CLIP which ignores caption semantics)
                    roi_dicts = roi_retriever.retrieve(
                        query=subquery,
                        input_image=None,
                        input_image_caption=input_caption,
                        k=5,
                        method="caption" if input_caption else "text",
                        filter_subject=filter_subject, 
                    )
                image_rois = [dict_to_roi_info(roi) for roi in roi_dicts]

            # Create ChunkInfo objects instead of dicts
            retrieved[subquery] = ChunkInfo(
                text_chunks=evidence,
                image_rois=image_rois
            )

        state.retrieved_chunks = retrieved
        print(retrieved)

        # Convert retrieved to a JSON-serializable dict - # Convert retrieved to a JSON-serializable dict
        retrieved_dict = {key: v.model_dump() for key, v in retrieved.items()}
        retriever_span.set_attribute("retriever.queries", json.dumps(retrieved_dict))

        # Evaluation metrics
        recall_result = recall_at_k(state, k=retrieval_k)
        p_at_k = precision_at_k(state, k=retrieval_k)
        mrr_score = mean_reciprocal_rank(state)
        ndcg_score = ndcg_at_k(state, k=retrieval_k)
        planner_hit = planner_hit_rate(state, k=retrieval_k)

        # Log to span
        retriever_span.set_attribute("retriever.recall", recall_result["recall"])
        retriever_span.set_attribute("retriever.precision", p_at_k)
        retriever_span.set_attribute("retriever.mrr", mrr_score)
        retriever_span.set_attribute("retriever.ndcg", ndcg_score)
        retriever_span.set_attribute("retriever.planner_hit", planner_hit)
        retriever_span.set_attribute("retriever.recall_hits", json.dumps(recall_result["hits"]))

    return state


# Build embeddings from CSV
df = pd.read_csv("scienceqa_augmented_100.csv")

roi_vectors = []
roi_metadata = []

for idx, row in df.iterrows():
    hint = safe_str(row.get("hint"))
    lecture = safe_str(row.get("lecture"))
    solution = safe_str(row.get("solution"))
    subject = safe_str(row.get("subject"))
    topic = safe_str(row.get("topic"))
    category = safe_str(row.get("category"))
    skill = safe_str(row.get("skill"))
    
    # Parse img_captions and img_ocr as dicts
    img_captions = safe_parse_json(row.get("img_captions"), default={})
    img_ocr_data = safe_parse_json(row.get("img_ocr"), default={})

    # Remove formatting, concatenate naturally
    # This improves both dense embedding quality and BM25 matching
    text_parts = []

    if subject:
        text_parts.append(f"{subject}")
    if topic:
        text_parts.append(f"{topic}")
    if category:
        text_parts.append(f"{category}")
    if skill:
        text_parts.append(f"{skill}")
    if lecture:
        text_parts.append(f"{lecture}")
    if hint:
        text_parts.append(f"{hint}")
    if solution:
        text_parts.append(f"{solution}")

    # Natural language format improves embedding quality
    text = ". ".join(text_parts) + "."
    
    media_type = "text"
    embedding = text_to_embedding(text)
    text_row = {
        'media_type': media_type,
        'text': text,
        'embeddings': embedding.tolist(),
        'subject': subject,  # Add as separate column for filtering
        'topic': topic,
        'category': category,
    }
    data = pd.concat([data, pd.DataFrame([text_row])], ignore_index=True)

    torch.cuda.empty_cache()
    gc.collect()

    # Parse image_paths
    image_paths = safe_parse_json(row.get("image_paths"), default=[])
    if not isinstance(image_paths, list):
        image_paths = []

    if not image_paths or len(image_paths) == 0:
        print(f"Row {idx}: No images to process")
        continue

    print(f"image paths: {image_paths}")

    # Process each image
    for image_path in image_paths:
        if not image_path or not os.path.exists(image_path):
            print(f"Skipping missing file: {image_path}")
            continue

        print(f"Processing image path: {image_path}")
        subject = safe_str(row.get("subject"))
        topic = safe_str(row.get("topic"))

        img_filename = os.path.basename(image_path)
        img_caption  = img_captions.get(img_filename, "")

        rois = extract_rois(image_path, subject=subject, topic=topic)
        for roi in rois:
            patch_bytes = base64.b64decode(roi["image_patch"])
            patch_pil   = Image.open(BytesIO(patch_bytes)).convert("RGB")
            p_embedding = image_embedding(patch_pil)
            roi_vectors.append(p_embedding[0])

            roi_metadata.append({
                "roi_id":            roi["roi_id"],
                "bbox":              roi["bbox"],
                "source_image_path": image_path,
                "image_patch":       roi["image_patch"],
                "caption":           img_caption,
                "subject":           subject,
                "topic":             topic,
            })

            data = pd.concat([data, pd.DataFrame([{
                "media_type":   "image",
                "roi_id":       roi["roi_id"],
                "source_image": image_path,
                "image_patch":  roi["image_patch"],
                "bbox":         str(roi["bbox"]),
                "text":         img_caption,
                "embeddings":   p_embedding[0].tolist(),
            }])], ignore_index=True)

        torch.cuda.empty_cache()
        gc.collect()

# Save data and metadata
data.to_csv("multimodal_embeddings.csv", index=False)

# Save ROI metadata with verification
import pickle
roi_metadata_path = "roi_metadata.pkl"
with open(roi_metadata_path, "wb") as f:
    pickle.dump(roi_metadata, f)
print(f"Saved {len(roi_metadata)} ROI metadata entries to {roi_metadata_path}")

# Verify the save
with open(roi_metadata_path, "rb") as f:
    verify_metadata = pickle.load(f)
print(f"Verified: {len(verify_metadata)} ROI metadata entries can be loaded")

# Create FAISS indexes
text_data = data[data['media_type'] == 'text']
image_data = data[data['media_type'] == 'image']

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
    
    # Verify consistency between image index and ROI metadata
    if len(roi_metadata) != image_index.ntotal:
        print(f"ERROR: Mismatch! roi_metadata has {len(roi_metadata)} entries but image_index has {image_index.ntotal}")
        print("This will cause image retrieval to fail!")
    else:
        print(f"Image index and ROI metadata are consistent ({len(roi_metadata)} entries)")
else:
    print("Warning: No image data to index")

# Final summary
print("INDEX CREATION SUMMARY")
print(f"Text entries:  {len(text_data) if len(text_data) > 0 else 0}")
print(f"Image entries: {len(image_data) if len(image_data) > 0 else 0}")
print(f"ROI metadata:  {len(roi_metadata)}")

# Quick sanity check
if __name__ == "__main__":
    test_cases = [
        ("natural science", "biology"),
        ("natural science", "physics"),
        ("natural science", "earth-science"),
        ("natural science", "chemistry"),
        ("natural science", "units-and-measurement"),
        ("natural science", "science-and-engineering-practices"),
        ("social science", "geography"),
        ("social science", "us-history"),
        ("social science", "world-history"),
        ("social science", "civics"),
        ("social science", "economics"),
        ("social science", "global-studies"),
        ("language science", "figurative-language"),
        ("language science", "writing-strategies"),
        ("language science", "vocabulary"),
        ("language science", "grammar"),
        ("language science", "verbs"),
        ("language science", "capitalization"),
        ("language science", "punctuation"),
        ("language science", "phonological-awareness"),
        ("language science", "reference-skills"),
        # Edge cases
        ("natural science", "unknown-topic"),
        ("", ""),
        ("weird subject", "weird topic"),
    ]

    print(f"{'Subject':<22} {'Topic':<35} {'Prompt (first 80 chars)'}")
    for subj, top in test_cases:
        p = get_gdino_prompt(subj, top)
        resolved = "TOPIC" if top.lower().replace(" ", "-") in TOPIC_PROMPTS else (
            "SUBJECT" if subj.lower() in SUBJECT_PROMPTS else "DEFAULT"
        )
        print(f"{subj:<22} {top:<35} [{resolved:>7}] {p[:80]}")