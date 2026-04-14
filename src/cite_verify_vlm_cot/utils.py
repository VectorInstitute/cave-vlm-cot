"""
utils.py
-------------------
Pydantic state models, shared utilities, and FAISS index construction
for the CaVe-VLM-CoT retrieval pipeline.
"""
import ast
import gc
import json
import math
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel

# Ordered list of text fields used to build a document's text representation.
TEXT_FIELD_ORDER = ["subject", "topic", "category", "skill", "lecture", "hint", "solution"]

# Pydantic models
# NOTE: Pydantic v2 models do NOT support dict-style access:
#   model['field']  → AttributeError
#   model.get(...)  → AttributeError
#   model.field     → Correct

# Define RoiInfo with all fields used in retriever
class RoiInfo(BaseModel):
    roi_id: str
    bbox: List[int]  
    source_image: str  
    image_patch: str = "" # empty when patch is stored in State.image_patch_cache
    caption: str
    score: float
    subject: str = "" 
    topic: str = ""

class ChunkInfo(BaseModel):
    text_chunks: List[str] = []
    image_rois: List[RoiInfo] = []

class State(BaseModel):
    pid: str
    question: str
    hint: Optional[str] = ""
    image_paths: Optional[List[str]] = []
    lecture: Optional[str] = ""
    choices: List[str]
    answer: int
    gold_answer: str = ""
    subject: str = ""
    topic: str = ""
    skill: str = ""
    category: str = ""

    img_captions: Optional[Dict[str, str]] = {}
    img_ocr: Optional[Dict[str, str]] = {}

    # Planner output
    subqueries: Optional[List[str]] = []
    query_history: List[List[str]] = []  # Track all query attempts

    # Retriever output
    retrieved_chunks: Dict[str, ChunkInfo] = {}
    # Deduplicated image patch store: roi_id → base64 patch.
    # RoiInfo.image_patch is left empty; consumers must resolve via this cache.
    image_patch_cache: Dict[str, str] = {}

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

# Helpers
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

def build_text_index(csv_path: str = None, index_dir: str = None):
    """
    Build the text FAISS index from the ScienceQA KB.
    Question images are passed directly to the solver at inference time
    and are not indexed here.
    """
    from retriever.retriever import text_to_embedding

    _project_dir = os.environ.get("CAVE_PROJECT_DIR", "/projects/cave-vlm-cot")
    _out_dir = index_dir or os.path.join(_project_dir, "indexes")  # ← use passed dir
    os.makedirs(_out_dir, exist_ok=True)

    if csv_path is None:
        csv_path = os.path.join(
            os.path.expanduser("~"),
            "cave-vlm-cot/src/cite_verify_vlm_cot/outputs/scienceqa_augmented.csv"
        )

    df = pd.read_csv(csv_path)
    # df = df.iloc[:10]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    data = pd.DataFrame()

    for idx, row in df.iterrows():
        # TEXT INDEXING
        hint = safe_str(row.get("hint"))
        lecture = safe_str(row.get("lecture"))
        solution = safe_str(row.get("solution"))
        subject = safe_str(row.get("subject"))
        topic = safe_str(row.get("topic"))
        category = safe_str(row.get("category"))
        skill = safe_str(row.get("skill"))

        text_parts = [p for p in [subject, topic, category, skill, lecture, hint, solution] if p]
        text = ". ".join(text_parts) + "."
        embedding = text_to_embedding(text)

        text_row = {
            "media_type": "text",
            "text": text,
            "embeddings": embedding.tolist(),
            "subject": subject,
            "topic": topic,
            "category": category,
        }
        data = pd.concat([data, pd.DataFrame([text_row])], ignore_index=True)

        torch.cuda.empty_cache()
        gc.collect()

    # Save the embeddings DataFrame so experiments.py can load it as `data`.
    # Without this, experiments.py crashes with FileNotFoundError on every run
    # because it expects "multimodal_embeddings.csv" to exist alongside the FAISS indexes.
    # Embeddings are stored as JSON strings since raw float lists aren't CSV-native.
    data_to_save = data.copy()
    data_to_save["embeddings"] = data_to_save["embeddings"].apply(
        lambda e: json.dumps(e) if isinstance(e, list) else e
    )
    data_to_save.to_csv(os.path.join(_out_dir, "multimodal_embeddings.csv"), index=False)
    print(f"Saved multimodal_embeddings.csv ({len(data_to_save)} rows)")

    # Build TEXT index — filter first, then reset index so FAISS row numbers
    # correspond directly to text_data.iloc[i], matching hybrid_retrieval's
    # expectation.
    text_data = data[data["media_type"] == "text"].reset_index(drop=True)
    if len(text_data):
        text_vectors = np.vstack(text_data["embeddings"].values).astype(np.float32)
        text_index = faiss.IndexFlatIP(text_vectors.shape[1])
        text_index.add(text_vectors)
        faiss.write_index(text_index, os.path.join(_out_dir, "text_index.faiss"))
        print(f"Text index: {len(text_data)} entries")

    print(f"\nINDEX SUMMARY:")
    print(f"  Text entries: {len(text_data)}")

    return data