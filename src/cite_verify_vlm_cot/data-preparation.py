import json
import os

import pandas as pd

# OCR with tesseract/ huggingface OCR
import pytesseract
import torch

# Image Preprocessing : resize image to 224x224
from PIL import Image
from tqdm import tqdm

# Caption and OCR generation
from transformers import BlipForConditionalGeneration, BlipProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Constants
TARGET_SIZE = (224, 224)
OUTPUT_ROOT = "processed_images"
DATASET_ROOT = os.environ.get("SCIENCEQA_ROOT", "/projects/cave-vlm-cot/scienceqa")


def collect_images(split, pid, primary_image_name=None):
    """
    Collect all images for a problem:
    - image.png
    - choice_0.png, choice_1.png, ...
    """
    pid_dir = os.path.join(DATASET_ROOT, split, str(pid))
    if not os.path.isdir(pid_dir):
        return []

    images = []

    # Primary image if specified
    if primary_image_name and primary_image_name.lower() != "none":
        p = os.path.join(pid_dir, primary_image_name)
        if os.path.exists(p):
            images.append(p)

    # Collect choice images
    for fname in sorted(os.listdir(pid_dir)):
        if fname.startswith("choice_") and fname.endswith(".png"):
            images.append(os.path.join(pid_dir, fname))

    return images


# Function to preprocess and save a single image
def preprocess_and_save_image(input_path, output_path):
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(TARGET_SIZE)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


# Batch BLIP captioning
def generate_captions_batch(image_paths, batch_size=8):
    captions = {}

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs)

        for path, out in zip(batch, outputs):
            captions[path] = processor.decode(out, skip_special_tokens=True)

    return captions


def run_ocr(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""


# Load problem annotations
data = json.load(open(os.path.join(DATASET_ROOT, "problems.json")))

examples = []

for pid, ex in tqdm(data.items()):
    question = ex["question"]
    choices = ex["choices"]
    answer = ex["answer"]
    hint = ex.get("hint", "")
    subject = ex.get("subject", "")
    topic = ex.get("topic", "")
    lecture = ex.get("lecture", "")
    category = ex.get("category", "")
    skill = ex.get("skill", "")
    solution = ex.get("solution", "")
    split = ex.get("split", "train")
    img_name = ex.get("image", None)
    image_paths = collect_images(split, pid, img_name)
    print(f"image paths: {image_paths}")

    item = {
        "pid": pid,
        "split": split,
        "question": question,
        "choices": choices,
        "answer": answer,
        "image_paths": image_paths,  # LIST
        "hint": hint,
        "lecture": lecture,
        "solution": solution,
        # Placeholder for image caption & OCR output
        # Later you can update this with actual BLIP/CLIP output
        "img_captions": {},  # BLIP-2 or other model caption
        "img_ocr": {},  # Tesseract or HuggingFace OCR text
        "subject": subject,
        "topic": topic,
        "category": category,
        "skill": skill,
    }

    examples.append(item)

# Convert to DataFrame
df = pd.DataFrame(examples)

all_images = []
processed_paths = []

for idx, row in df.iterrows():
    for img_path in row["image_paths"]:
        if os.path.exists(img_path):
            rel_path = os.path.relpath(img_path, DATASET_ROOT)
            out_path = os.path.join(OUTPUT_ROOT, rel_path)
            all_images.append((img_path, out_path))

# Deduplicate
all_images = list(dict.fromkeys(all_images))
print(f"all images: {all_images}")

for in_path, out_path in tqdm(all_images, desc="Preprocessing images"):
    preprocess_and_save_image(in_path, out_path)
    processed_paths.append(out_path)

captions_batch = generate_captions_batch(processed_paths, batch_size=8)
ocr_batch = {p: run_ocr(p) for p in tqdm(processed_paths, desc="Running OCR")}

for idx, row in tqdm(df.iterrows(), total=len(df)):
    captions = {}
    ocr_outputs = {}
    new_paths = []

    for img_path in row["image_paths"]:
        rel_path = os.path.relpath(img_path, DATASET_ROOT)
        print(f"relative path: {rel_path}")
        out_path = os.path.join(OUTPUT_ROOT, rel_path)
        print(f"output path: {out_path}")

        if out_path not in captions_batch:
            continue

        fname = os.path.basename(out_path)
        captions[fname] = captions_batch[out_path]
        ocr_outputs[fname] = ocr_batch.get(out_path, "")
        new_paths.append(out_path)

    df.at[idx, "image_paths"] = json.dumps(new_paths)
    df.at[idx, "img_captions"] = json.dumps(captions)
    df.at[idx, "img_ocr"] = json.dumps(ocr_outputs)

_output_dir = os.path.join(os.environ.get("CAVE_PROJECT_DIR", "/projects/cave-vlm-cot"), "outputs")
os.makedirs(_output_dir, exist_ok=True)
df.to_csv(os.path.join(_output_dir, "scienceqa_augmented.csv"), index=False)
df.to_json(os.path.join(_output_dir, "scienceqa_augmented.json"), orient="records", indent=2)
