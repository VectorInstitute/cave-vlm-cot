import ast
import base64
import gc
import json
import math
import os
import pickle
from io import BytesIO
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel

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

# def extract_rois(image_path: str, subject: str = "", topic: str = "") -> List[dict]:
#     from retriever import classify_image_type, extract_cc_rois, extract_gdino_rois, _is_blank, _nms, MAX_ROIS

#     img_pil  = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)
#     img_arr  = np.array(img_pil)
#     img_type = classify_image_type(img_arr)
#     fname    = os.path.basename(image_path)
#     print(f"  [{fname}] type={img_type}")

#     raw_rois = []
#     if img_type == "diagram_text":
#         raw_rois = extract_cc_rois(img_arr)
#         if not raw_rois:
#             print(f"  [{fname}] CC found nothing → trying GDINO")
#             raw_rois = extract_gdino_rois(img_pil, img_arr, subject, topic)
#     else:
#         raw_rois = extract_gdino_rois(img_pil, img_arr, subject, topic)
#         if not raw_rois:
#             print(f"  [{fname}] GDINO found nothing → trying CC")
#             raw_rois = extract_cc_rois(img_arr)

#     if not raw_rois:
#         print(f"  [{fname}] both methods found nothing → skipping image")
#         return []

#     # shared post-processing
#     filtered = []
#     for r in raw_rois:
#         x1, y1, x2, y2 = int(r["bbox"][0]), int(r["bbox"][1]), int(r["bbox"][2]), int(r["bbox"][3])
#         patch = img_arr[y1:y2, x1:x2]
#         if patch.size > 0 and not _is_blank(patch):
#             filtered.append(r)
#     deduped = _nms(filtered)
#     final   = deduped[:MAX_ROIS]

#     result = []
#     for roi in final:
#         result.append({
#             "roi_id":            f"{fname}_{tuple(roi['bbox'])}",
#             "bbox":              roi["bbox"],
#             "image_patch":       roi["image_patch"],
#             "source_image_path": image_path,
#             "subject":           subject,
#             "topic":             topic,
#         })
#     print(f"  [{fname}] → {len(result)} RoIs")
#     return result

# Index builder
# def build_indexes(csv_path: str = "scienceqa_augmented.csv"):
#     """Build text + image FAISS indexes and ROI metadata from the augmented CSV."""
#     from retriever import image_embedding, text_to_embedding

#     df = pd.read_csv(csv_path)
#     df = df.iloc[:1000]

#     # initialise data as an empty DataFrame so pd.concat works from the first row
#     data = pd.DataFrame()

#     roi_meta: list = []

#     for idx, row in df.iterrows():
#         hint = safe_str(row.get("hint"))
#         lecture = safe_str(row.get("lecture"))
#         solution = safe_str(row.get("solution"))
#         subject = safe_str(row.get("subject"))
#         topic = safe_str(row.get("topic"))
#         category = safe_str(row.get("category"))
#         skill = safe_str(row.get("skill"))

#         img_captions = safe_parse_json(row.get("img_captions"), default={})

#         text_parts = [p for p in [subject, topic, category, skill, lecture, hint, solution] if p]
#         text = ". ".join(text_parts) + "."
#         embedding = text_to_embedding(text)

#         text_row = {
#             "media_type": "text",
#             "text": text,
#             "embeddings": embedding.tolist(),
#             "subject": subject,
#             "topic": topic,
#             "category": category,
#         }
#         data = pd.concat([data, pd.DataFrame([text_row])], ignore_index=True)

#         torch.cuda.empty_cache()
#         gc.collect()

#         image_paths = safe_parse_json(row.get("image_paths"), default=[])
#         if not isinstance(image_paths, list):
#             image_paths = []

#         if not image_paths:
#             print(f"Row {idx}: no images to process")
#             continue

#         print(f"Row {idx} image paths: {image_paths}")

#         for image_path in image_paths:
#             if not image_path or not os.path.exists(image_path):
#                 print(f"  Skipping missing file: {image_path}")
#                 continue

#             img_filename = os.path.basename(image_path)
#             img_caption = img_captions.get(img_filename, "")

#             rois = extract_rois(image_path, subject=subject, topic=topic)
#             for roi in rois:
#                 patch_bytes = base64.b64decode(roi["image_patch"])
#                 patch_pil = Image.open(BytesIO(patch_bytes)).convert("RGB")
#                 p_embedding = image_embedding(patch_pil)

#                 roi_meta.append(
#                     {
#                         "roi_id": roi["roi_id"],
#                         "bbox": roi["bbox"],
#                         "source_image_path": image_path,
#                         "image_patch": roi["image_patch"],
#                         "caption": img_caption,
#                         "subject": subject,
#                         "topic": topic,
#                     }
#                 )

#                 data = pd.concat(
#                     [
#                         data,
#                         pd.DataFrame(
#                             [
#                                 {
#                                     "media_type": "image",
#                                     "roi_id": roi["roi_id"],
#                                     "source_image": image_path,
#                                     "image_patch": roi["image_patch"],
#                                     "bbox": str(roi["bbox"]),
#                                     "text": img_caption,
#                                     "embeddings": p_embedding[0].tolist(),
#                                 }
#                             ]
#                         ),
#                     ],
#                     ignore_index=True,
#                 )

#             torch.cuda.empty_cache()
#             gc.collect()

#     # Persist embeddings DataFrame
#     data.to_csv("multimodal_embeddings.csv", index=False)

#     # Persist ROI metadata
#     roi_metadata_path = "roi_metadata.pkl"
#     with open(roi_metadata_path, "wb") as f:
#         pickle.dump(roi_meta, f)
#     print(f"Saved {len(roi_meta)} ROI metadata entries to {roi_metadata_path}")

#     # Verify the pickle round-trips correctly
#     with open(roi_metadata_path, "rb") as f:
#         verify_metadata = pickle.load(f)
#     print(f"Verified: {len(verify_metadata)} ROI metadata entries loaded back successfully")

#     # Build and save FAISS indexes
#     text_data = data[data["media_type"] == "text"]
#     image_data = data[data["media_type"] == "image"]

#     if len(text_data):
#         text_vectors = np.vstack(text_data["embeddings"].values).astype(np.float32)
#         text_index = faiss.IndexFlatIP(text_vectors.shape[1])
#         text_index.add(text_vectors)
#         faiss.write_index(text_index, "text_index.faiss")
#         print(f"Text index: {len(text_data)} entries")

#     if len(image_data):
#         image_vectors = np.vstack(image_data["embeddings"].values).astype(np.float32)
#         image_index = faiss.IndexFlatIP(image_vectors.shape[1])
#         image_index.add(image_vectors)
#         faiss.write_index(image_index, "image_index.faiss")
#         print(f"Image index: {len(image_data)} entries")

#         if len(roi_meta) != image_index.ntotal:
#             print(
#                 f"ERROR: Mismatch — roi_metadata={len(roi_meta)}, "
#                 f"image_index={image_index.ntotal}"
#             )
#         else:
#             print(f"Image index and ROI metadata consistent ({len(roi_meta)} entries)")

#     print("\nINDEX CREATION SUMMARY")
#     print(f"  Text entries:  {len(text_data)}")
#     print(f"  Image entries: {len(image_data)}")
#     print(f"  ROI metadata:  {len(roi_meta)}")

#     # Return data so callers don't have to reload the CSV
#     return data

def build_full_image_indexes(csv_path: str = "scienceqa_augmented.csv"):
    """Build indexes storing FULL IMAGES instead of ROI patches."""
    from retriever import image_embedding, text_to_embedding

    df = pd.read_csv(csv_path)
    df = df.iloc[:1000]
    
    data = pd.DataFrame()
    image_metadata = []

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

        # IMAGE INDEXING - Store full image
        image_paths = safe_parse_json(row.get("image_paths"), default=[])
        img_captions = safe_parse_json(row.get("img_captions"), default={})

        for image_path in image_paths:
            if not image_path or not os.path.exists(image_path):
                continue

            img_filename = os.path.basename(image_path)
            img_caption = img_captions.get(img_filename, "")

            img_pil = Image.open(image_path).convert("RGB")
            img_embedding = image_embedding(img_pil)
            
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            image_metadata.append({
                "image_id": f"{row.get('pid')}_{img_filename}",
                "source_path": image_path,
                "image_base64": img_base64,
                "caption": img_caption,
                "subject": subject,
                "topic": topic,
                "pid": str(row.get("pid", "")),
            })

            data = pd.concat([data, pd.DataFrame([{
                "media_type": "image",
                "image_id": f"{row.get('pid')}_{img_filename}",
                "source_image": image_path,
                "text": img_caption,
                "embeddings": img_embedding[0].tolist(),
                "subject": subject,
                "topic": topic,
            }])], ignore_index=True)

        torch.cuda.empty_cache()
        gc.collect()

    # Save metadata
    with open("image_metadata.pkl", "wb") as f:
        pickle.dump(image_metadata, f)
    print(f"Saved {len(image_metadata)} full image metadata entries")

    # Save the embeddings DataFrame so experiments.py can load it as `data`.
    # Without this, experiments.py crashes with FileNotFoundError on every run
    # because it expects "multimodal_embeddings.csv" to exist alongside the FAISS indexes.
    # Embeddings are stored as JSON strings since raw float lists aren't CSV-native.
    data_to_save = data.copy()
    data_to_save["embeddings"] = data_to_save["embeddings"].apply(
        lambda e: json.dumps(e) if isinstance(e, list) else e
    )
    data_to_save.to_csv("multimodal_embeddings.csv", index=False)
    print(f"Saved multimodal_embeddings.csv ({len(data_to_save)} rows)")

    # Build TEXT index — filter first, then reset index so FAISS row numbers
    # correspond directly to text_data.iloc[i], matching hybrid_retrieval's
    # expectation.  Building from the full interleaved `data` would misalign
    # FAISS positions with DataFrame rows the moment image rows are present.
    text_data = data[data["media_type"] == "text"].reset_index(drop=True)
    if len(text_data):
        text_vectors = np.vstack(text_data["embeddings"].values).astype(np.float32)
        text_index = faiss.IndexFlatIP(text_vectors.shape[1])
        text_index.add(text_vectors)
        faiss.write_index(text_index, "text_index.faiss")
        print(f"Text index: {len(text_data)} entries")

    # Build IMAGE index (full images)
    image_data = data[data["media_type"] == "image"]
    if len(image_data):
        image_vectors = np.vstack(image_data["embeddings"].values).astype(np.float32)
        image_index = faiss.IndexFlatIP(image_vectors.shape[1])
        image_index.add(image_vectors)
        faiss.write_index(image_index, "full_image_index.faiss")
        print(f"Full image index: {len(image_data)} entries")

    print(f"\nINDEX SUMMARY:")
    print(f"  Text entries: {len(text_data)}")
    print(f"  Full images:  {len(image_metadata)}")

    return data, image_metadata