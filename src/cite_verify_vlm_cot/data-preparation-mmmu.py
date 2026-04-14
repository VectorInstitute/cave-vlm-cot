"""
data-preparation-mmmu.py
------

Prepares the MMMU dataset (dev + validation splits) for the CaVe-VLM-CoT pipeline.

Key differences from ScienceQA data-preparation.py:
  1. Loads from HuggingFace Hub via `datasets`, not from a local problems.json.
  2. Covers dev + validation splits only (test withholds gold answers).
  3. Answer is a letter ("A"–"E") — converted to an integer index.
  4. Images are PIL objects embedded in HF rows — materialised to disk here.
  5. No hint / lecture / solution fields.
     - dev split has an `explanation` field → mapped to `lecture` so it is
       picked up by build_text_index() in utils.py without any changes there.
     - validation split has no explanation → `lecture` is left as "".
  6. MMMU taxonomy: `subject` (broad) + `subfield` (narrow) → mapped to
     `subject` and `topic` respectively, matching the State schema.

Output
------
  mmmu_augmented.csv   — one row per question, same column schema as
                         scienceqa_augmented.csv so experiments.py can load
                         it without modification (beyond the --dataset flag).
  mmmu_augmented.json  — same data, JSON format.

Usage
-----
  # All 30 MMMU subjects, both splits:
  python data-preparation-mmmu.py

  # Subset of subjects (faster for testing):
  MMMU_SUBJECTS="Math,Physics" python data-preparation-mmmu.py
"""

import json
import os
import ast

import pandas as pd
import pytesseract
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

TARGET_SIZE  = (224, 224)
OUTPUT_ROOT = os.path.join(os.environ.get("CAVE_PROJECT_DIR", "/projects/cave-vlm-cot"), "processed_images_mmmu")
BATCH_SIZE   = 8

# dev has explanations; validation does not.
# test is excluded: gold answers are withheld.
SPLITS = ["dev", "validation"]

# All 30 MMMU subjects. Override via env var for quick tests:
#   MMMU_SUBJECTS="Math,Physics" python data-preparation-mmmu.py
_subjects_env = os.environ.get("MMMU_SUBJECTS", "")
ALL_SUBJECTS = [s.strip() for s in _subjects_env.split(",") if s.strip()] or [
    "Accounting", "Agriculture", "Architecture_and_Engineering",
    "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
    "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
    "Electronics", "Energy_and_Power", "Finance", "Geography",
    "History", "Literature", "Manage", "Marketing",
    "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology",
    "Public_Health", "Sociology",
]

# MMMU answers are letters; map to 0-based integer index.
ANSWER_LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Helpers
def letter_to_index(letter: str) -> int:
    return ANSWER_LETTER_TO_INDEX.get(str(letter).strip().upper(), 0)

def collect_mmmu_images(row: dict, pid: str) -> list[str]:
    """
    Materialise PIL images embedded in the HF row to disk.

    MMMU rows carry up to 7 PIL images in fields image_1 … image_7.
    Each is resized to TARGET_SIZE and saved as PNG so the rest of the
    pipeline (BLIP captioning, OCR, solver) can treat them identically
    to ScienceQA on-disk images.

    Returns a list of saved file paths (empty slots are skipped).
    """
    paths = []
    for i in range(1, 8):
        img_obj = row.get(f"image_{i}")
        if img_obj is None:
            continue
        out_dir = os.path.join(OUTPUT_ROOT, pid)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"image_{i}.png")
        img_obj.convert("RGB").resize(TARGET_SIZE).save(out_path, format="PNG")
        paths.append(out_path)
    return paths


def generate_captions_batch(image_paths: list[str], batch_size: int = BATCH_SIZE) -> dict:
    """Batch BLIP captioning — identical logic to ScienceQA data-preparation.py"""
    captions = {}
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = blip_processor(images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = blip_model.generate(**inputs)
        for path, out in zip(batch, outputs):
            captions[path] = blip_processor.decode(out, skip_special_tokens=True)
    return captions


def run_ocr(image_path: str) -> str:
    try:
        return pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception as e:
        print(f"OCR error on {image_path}: {e}")
        return ""


# Data collection
examples = []
for subject in ALL_SUBJECTS:
    for split in SPLITS:
        print(f"\nLoading MMMU/{subject} — {split} split")
        try:
            ds = load_dataset("MMMU/MMMU", subject, split=split)
        except Exception as e:
            print(f"  Skipping {subject}/{split}: {e}")
            continue

        for row in tqdm(ds, desc=f"{subject}/{split}"):
            pid = row["id"]  # e.g. "validation_Accounting_1" — unique across splits

            # Choices
            raw_options = row.get("options", "[]")
            if isinstance(raw_options, list):
                choices = raw_options
            elif isinstance(raw_options, str) and raw_options.strip():
                try:
                    choices = json.loads(raw_options)
                except json.JSONDecodeError:
                    try:
                        choices = ast.literal_eval(raw_options)
                    except (ValueError, SyntaxError):
                        print(f"[WARN] Could not parse options for {row.get('id', '?')}: {raw_options!r}")
                        choices = []
            else:
                choices = []

            # Answer (letter → index)
            answer_letter = row.get("answer", "A") or "A"
            answer_idx    = letter_to_index(answer_letter)
            gold_answer   = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""

            # Images
            image_paths = collect_mmmu_images(row, pid)

            # Explanation → lecture
            # Only the dev split carries an `explanation` field.
            # Mapping it to `lecture` means build_text_index() in utils.py
            # indexes it automatically (TEXT_FIELD_ORDER includes "lecture")
            # without any changes to utils.py or experiments.py
            # validation rows get an empty string — safe_str("") → "" → dropped
            # by the `if p` filter in build_text_index, so no noise is added.
            explanation = (row.get("explanation") or "").strip()

            # Taxonomy
            # MMMU field  →  State / CSV field
            # subject     →  subject   (broad, e.g. "Math")
            # subfield    →  topic     (narrow, e.g. "Calculus")
            # subject     →  category  (no finer grouping available)
            # (none)      →  skill     (leave empty)
            mmmu_subject  = row.get("subject",  subject)
            mmmu_subfield = row.get("subfield", "")

            examples.append(
                {
                    "pid":          pid,
                    "split":        split,
                    "question":     row.get("question", ""),
                    "choices":      choices,        # list — serialised below
                    "answer":       answer_idx,     # int, same as ScienceQA
                    "gold_answer":  gold_answer,
                    "image_paths":  image_paths,    # list — serialised below
                    "hint":         "",             # MMMU has no hint field
                    # explanation (dev only) fills the lecture slot so it is
                    # automatically picked up by the FAISS index builder.
                    "lecture":      explanation,
                    "solution":     "",             # MMMU has no solution field
                    "img_captions": {},             # filled in captioning pass
                    "img_ocr":      {},             # filled in OCR pass
                    "subject":      mmmu_subject,
                    "topic":        mmmu_subfield,
                    "category":     mmmu_subject,   # best available proxy
                    "skill":        "",
                }
            )

print(f"\nCollected {len(examples)} examples across {SPLITS} splits.")

df = pd.DataFrame(examples)

# Image captioning + OCR
# Collect every unique on-disk path across all rows (images are already saved).
all_image_paths = list(
    dict.fromkeys(p for paths in df["image_paths"] for p in paths)
)
print(f"Running BLIP captioning on {len(all_image_paths)} images")
captions_batch = generate_captions_batch(all_image_paths, batch_size=BATCH_SIZE)

print(f"Running OCR on {len(all_image_paths)} images")
ocr_batch = {p: run_ocr(p) for p in tqdm(all_image_paths, desc="OCR")}

# Write captions + OCR back into the dataframe, serialise list columns to JSON.
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Finalising rows"):
    captions    = {}
    ocr_outputs = {}

    for img_path in row["image_paths"]:
        fname = os.path.basename(img_path)
        captions[fname]    = captions_batch.get(img_path, "")
        ocr_outputs[fname] = ocr_batch.get(img_path, "")

    df.at[idx, "image_paths"]  = json.dumps(row["image_paths"])
    df.at[idx, "img_captions"] = json.dumps(captions)
    df.at[idx, "img_ocr"]      = json.dumps(ocr_outputs)
    df.at[idx, "choices"]      = json.dumps(row["choices"])
    # gold_answer is a plain string — no serialisation needed.

# Save
_output_dir = os.path.join(os.environ.get("CAVE_PROJECT_DIR", "/projects/cave-vlm-cot"), "outputs")
os.makedirs(_output_dir, exist_ok=True)
df.to_csv(os.path.join(_output_dir, "mmmu_augmented.csv"),  index=False)
df.to_json(os.path.join(_output_dir, "mmmu_augmented.json"), orient="records", indent=2)

print(f"\nSaved mmmu_augmented.csv  ({len(df)} rows)")
print(f"Saved mmmu_augmented.json ({len(df)} rows)")
print("\nSplit breakdown:")
print(df["split"].value_counts().to_string())
print("\nSubject breakdown:")
print(df["subject"].value_counts().to_string())