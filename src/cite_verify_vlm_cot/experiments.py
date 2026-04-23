"""
Experiments.py
1. Loads a dataset of science questions (with images)
2. Runs each question through a planner → retriever → solver → verifier pipeline with retries
3. Evaluates the results using multiple metrics
4. Logs everything to Phoenix (an ML observability platform)
"""
import argparse
import os
import ast
import gc
import json
import math
import httpx

import time
import traceback

# Image processing
# Progress bars
import pickle

# Vector search
import faiss

# Data processing
import pandas as pd

import traceback
from opentelemetry import trace as otel_trace

# Set environment variables before any model/library imports
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
HOME_DIR = os.path.expanduser("~")

# Project storage — 250 GB, used for model caches, logs, outputs, and indexes.
# Falls back to home directory if the project path doesn't exist.
PROJECT_DIR = os.environ.get("CAVE_PROJECT_DIR", "/projects/cave-vlm-cot")
if not os.path.isdir(PROJECT_DIR):
    print(f"[Storage] Project dir {PROJECT_DIR} not found, falling back to ~/cave-vlm-cot")
    PROJECT_DIR = os.path.join(HOME_DIR, "cave-vlm-cot")
# Explicit cache path — bypasses HF_HOME/HF_HUB_CACHE env var confusion
# Models are stored at PROJECT_DIR/hf_cache/ (not .../hf_cache/hub/)
VERIFIER_CACHE = os.path.join(PROJECT_DIR, "hf_cache")
WORKDIR = os.path.join(HOME_DIR, "cave-vlm-cot/src/cite_verify_vlm_cot")

# Shard configuration
# Reads from CLI args, falling back to SLURM_ARRAY_TASK_ID when running as
# a job array.  A single-node run (no args) behaves identically to before.
#
# Usage examples:
#   python experiments.py                         # full dataset, one process
#   python experiments.py --shard 0 --num-shards 3  # first third
#   python experiments.py --shard 1 --num-shards 3  # second third
#   python experiments.py --shard 2 --num-shards 3  # final third
#
# SLURM array job (see cave_array.slurm):
#   SLURM_ARRAY_TASK_ID is used as --shard automatically.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--shard",      type=int, default=None,
                     help="0-based shard index to process")
_parser.add_argument("--num-shards", type=int, default=1,
                     help="Total number of shards (default: 1 = no sharding)")
_parser.add_argument(
    "--max-samples",
    type=int,
    default=int(os.environ.get("CAVE_MAX_SAMPLES", 0)),
    help="Optional deterministic random sample size before sharding. 0 means use all rows.",
)
_parser.add_argument(
    "--sample-seed",
    type=int,
    default=int(os.environ.get("CAVE_SAMPLE_SEED", 42)),
    help="Random seed used when --max-samples is set.",
)
# Ablation / experiment selection
_parser.add_argument(
    "--pipeline",
    type=str,
    default="full",
    choices=["full", "retrieval-solver", "solver-only", "no-citation-injector"],
    help=(
        "Pipeline variant to run.\n"
        "  full                   — complete pipeline (default)\n"
        "  retrieval-solver       — Ablation 1: Planner→Retriever→Solver only\n"
        "  solver-only            — Ablation 2: Solver only (no retrieval/verifier)\n"
        "  no-citation-injector   — Ablation 3: full pipeline minus inject_citations"
    ),
)
_parser.add_argument(
    "--verifier-model",
    type=str,
    default=os.environ.get("CAVE_VERIFIER_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
    help=(
        "Verifier VLM model id or local path. For A40 runs, the recommended default "
        "is Qwen/Qwen2.5-VL-32B-Instruct with --verifier-quantization 4bit."
    ),
)
_parser.add_argument(
    "--verifier-quantization",
    type=str,
    default=os.environ.get("CAVE_VERIFIER_QUANTIZATION", "4bit"),
    choices=["4bit", "8bit", "bf16"],
    help="Verifier loading precision. 4bit is recommended for 32B on a single A40.",
)
_parser.add_argument(
    "--verifier-device",
    type=str,
    default=os.environ.get("CAVE_VERIFIER_DEVICE", "cuda:2"),
    help="Device for the verifier when using a single-device map.",
)
_parser.add_argument(
    "--verifier-device-map",
    type=str,
    default=os.environ.get("CAVE_VERIFIER_DEVICE_MAP", "single"),
    choices=["single", "auto"],
    help=(
        "Use 'single' to pin verifier to --verifier-device, or 'auto' to let "
        "Transformers shard the verifier. Keep 'single' for 32B 4bit on A40."
    ),
)
_parser.add_argument(
    "--verifier-max-pixels",
    type=int,
    default=int(os.environ.get("CAVE_VERIFIER_MAX_PIXELS", 768)),
    help=(
        "Verifier image token budget multiplier in units of 28*28 pixels. "
        "Lower is faster; 768 is a good 5k-sample A40 throughput default."
    ),
)
_parser.add_argument(
    "--no-traces",
    action="store_true",
    default=False,
    help=(
        "Disable OpenTelemetry trace export to Phoenix. Experiment results\n"
        "(datasets, evaluators) are still saved — only per-span trace data\n"
        "is suppressed. Saves significant storage on large runs."
    ),
)
_parser.add_argument(
    "--local-results",
    action="store_true",
    default=os.environ.get("CAVE_LOCAL_RESULTS", "0").lower() in {"1", "true", "yes"},
    help="Bypass Phoenix dataset upload/run_experiment and write per-shard CSV results locally.",
)
_parser.add_argument(
    "--dataset",
    type=str,
    default="scienceqa",
    choices=["scienceqa", "mmmu"],
    help="Dataset to evaluate on (default: scienceqa)",
)

_args, _ = _parser.parse_known_args()
# When --shard is passed explicitly (by cave_array.sh), use it directly.
# Only fall back to SLURM_ARRAY_TASK_ID for legacy single-experiment runs.
_slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
SHARD_INDEX = _args.shard if _args.shard is not None else (
    _slurm_task_id if _slurm_task_id >= 0 else 0
)
NUM_SHARDS = _args.num_shards
PIPELINE_MODE  = _args.pipeline        # "full" | "retrieval-solver" | "solver-only" | "no-citation-injector"
MODEL_VARIANT  = "qwen25"             # qwen25 = Qwen2.5 extractor (reverted from Qwen3)
NO_TRACES      = _args.no_traces       # suppress span export to save Phoenix storage
LOCAL_RESULTS  = _args.local_results   # bypass Phoenix completely and write local CSV
DATASET_NAME = _args.dataset
MAX_SAMPLES = _args.max_samples
SAMPLE_SEED = _args.sample_seed
VERIFIER_MODEL_ID = _args.verifier_model
VERIFIER_QUANTIZATION = _args.verifier_quantization
VERIFIER_DEVICE = _args.verifier_device
VERIFIER_DEVICE_MAP = _args.verifier_device_map
VERIFIER_MAX_PIXELS = _args.verifier_max_pixels

# Validate
if not (0 <= SHARD_INDEX < NUM_SHARDS):
    raise ValueError(f"--shard {SHARD_INDEX} out of range for --num-shards {NUM_SHARDS}")
print(f"[Shard]    Running shard {SHARD_INDEX + 1} / {NUM_SHARDS}")
print(f"[Pipeline] {PIPELINE_MODE}")
print(f"[Models]   extractor/verifier variant: {MODEL_VARIANT}")
print(f"[Verifier] model={VERIFIER_MODEL_ID} quantization={VERIFIER_QUANTIZATION} device_map={VERIFIER_DEVICE_MAP}")
print(f"[Traces]   {'DISABLED (--no-traces)' if NO_TRACES else 'enabled'}")
print(f"[Results]  {'LOCAL CSV (--local-results)' if LOCAL_RESULTS else 'Phoenix'}")

CACHE_DIR = os.path.join(PROJECT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ before any key checks

# Deep learning
import torch
from evaluations import (
    mean_reciprocal_rank,
    ndcg_at_k,
    planner_coverage_score,
    planner_hit_rate,
    planner_specificity_score,
    precision_at_k,
    recall_at_k,
    compute_cave_score,
    evaluate_verifier_quality
)

# Phoenix for experiments/tracing
from phoenix.client import Client

# Your local modules
from utils import State, safe_str, safe_parse_json
from transformers import AutoProcessor, MllamaForConditionalGeneration
from unsloth import FastLanguageModel
from verifier.verifier import (
    build_cave_vlm_cot_graph,
    build_retrieval_solver_graph,
    build_solver_only_graph,
    build_pipeline_without_citation_injector,
)

# Load your data
# CSVs live on project storage (/projects/cave-vlm-cot/outputs/) — the same
# location both data-preparation scripts write to via $CAVE_PROJECT_DIR.
# WORKDIR (home dir) holds source code only, not large data files.
_CSV_MAP = {
    "scienceqa": os.path.join(PROJECT_DIR, "outputs/scienceqa_augmented.csv"),
    "mmmu":      os.path.join(PROJECT_DIR, "outputs/mmmu_augmented.csv"),
}
df = pd.read_csv(_CSV_MAP[DATASET_NAME])
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# df = df.iloc[:5]
if MAX_SAMPLES > 0:
    sample_n = min(MAX_SAMPLES, len(df))
    df = df.sample(n=sample_n, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(
        f"[Data]     Using deterministic random sample of {len(df)} rows "
        f"(--max-samples {MAX_SAMPLES}, --sample-seed {SAMPLE_SEED})"
    )
# Shard the dataframe.
# Each shard gets a contiguous, non-overlapping slice.
# If the CSV is not pre-shuffled by subject, add:
#   df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# before this block so each shard gets a balanced cross-subject sample.
_total_rows = len(df)
_rows_per_shard = math.ceil(_total_rows / NUM_SHARDS)
_shard_start   = SHARD_INDEX * _rows_per_shard
_shard_end     = min(_shard_start + _rows_per_shard, _total_rows)
df = df.iloc[_shard_start:_shard_end].reset_index(drop=True)
print(f"[Shard] Rows {_shard_start}-{_shard_end - 1} ({len(df)} questions)")

# Prepare dataset for experiments - Modified to handle JSON fields properly
experiment_data = []

for idx, row in df.iterrows():
    # Parse choices from string to list
    choices_raw = row.get("choices")
    choices = safe_parse_json(choices_raw, default=[]) if isinstance(choices_raw, str) else (choices_raw or [])

    # Parse answer
    answer = int(row["answer"]) if pd.notna(row.get("answer")) else 0

    # Get gold answer
    def parse_choices(x):
        if isinstance(x, float):  # NaN
            return []
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    choices = parse_choices(choices)
    gold_answer = choices[answer] if isinstance(choices, list) and choices and 0 <= int(answer) < len(choices) else ""

    # Parse image_paths
    image_paths = safe_parse_json(row.get("image_paths"), default=[])
    if not isinstance(image_paths, list):
        image_paths = []

    # Parse img_captions and img_ocr
    img_captions = safe_parse_json(row.get("img_captions"), default={})
    img_ocr_data = safe_parse_json(row.get("img_ocr"), default={})

    experiment_data.append(
        {
            "pid": str(row.get("pid", "")),
            "question": safe_str(row.get("question")),
            "hint": safe_str(row.get("hint")),
            "choices": json.dumps(choices),  # Store as JSON string
            "lecture": safe_str(row.get("lecture")),
            "answer": answer,
            "gold_answer": gold_answer,
            "image_paths": json.dumps(image_paths),  # Store as JSON string
            "img_captions": json.dumps(img_captions) if img_captions else "",
            "img_ocr": json.dumps(img_ocr_data) if img_ocr_data else "",
            "subject": safe_str(row.get("subject")),
            "topic": safe_str(row.get("topic")),
            "skill": safe_str(row.get("skill")),
            "category": safe_str(row.get("category")),
            "dataset": DATASET_NAME,   # add this line for traceability in Phoenix
        }
    )

experiment_df = pd.DataFrame(experiment_data)

# Upload (or reuse) dataset in Phoenix
# Initialize the experiments client
experiments_client = Client(
    base_url="https://app.phoenix.arize.com/s/CaVe-VLM-CoT",
    headers={"api_key": os.environ["PHOENIX_API_KEY"]},
)

# Upload dataset to Phoenix
def create_or_get_dataset(client, name, df, input_keys, output_keys, max_retries=3):
    for attempt in range(max_retries):
        try:
            dataset = client.datasets.create_dataset(
                name=name,
                dataframe=df,
                input_keys=input_keys,
                output_keys=output_keys,
                timeout=300,
            )
            print(f"Created dataset: {dataset.name}")
            return dataset
        except Exception as e:
            print(f"Upload attempt {attempt+1} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)   # 30s, 60s, 90s backoff
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    # Fallback: try exact name AND name with ±1 row count (handles dropped bad rows)
    row_count = int(name.split("_")[-1])
    candidate_names = [name, name.replace(f"_{row_count}", f"_{row_count - 1}"),
                             name.replace(f"_{row_count}", f"_{row_count + 1}")]
    for candidate in candidate_names:
        try:
            dataset = client.datasets.get_dataset(dataset=candidate, timeout=10000)
            print(f"Using existing dataset: {dataset.name}")
            return dataset
        except Exception:
            continue

    raise RuntimeError(f"Could not create or fetch dataset '{name}'")

cave_dataset = create_or_get_dataset(
    client=experiments_client,
    name=f"{DATASET_NAME}-cave-vlm-cot-shard{SHARD_INDEX}-of-{NUM_SHARDS}_{len(df)}",
    df=experiment_df,
    input_keys=[
        "pid", "question", "hint", "choices", "lecture", "answer",
        "image_paths", "img_captions", "img_ocr",
        "subject", "topic", "category", "skill", "dataset",
    ],
    output_keys=["gold_answer"],
)


# Define Evaluators
def eval_planner_hit(output: dict, expected: dict) -> bool:
    """Did subqueries lead to retrieving the gold answer?"""
    if output is None:
        return False
    return output.get("planner_hit", 0.0)

def eval_planner_coverage(output: dict) -> float:
    """Extract pre-computed planner coverage score."""
    if output is None:
        return 0.0
    return output.get("planner_coverage", 0.0)

def eval_subquery_count(output: dict) -> int:
    """Count subqueries generated."""
    if output is None:
        return 0
    return len(output.get("subqueries", []))

def eval_recall(output: dict) -> float:
    """Extract pre-computed Recall@2."""
    if output is None:
        return 0.0
    return output.get("recall_at_2", 0.0)

def eval_precision(output: dict) -> float:
    """Extract pre-computed Precision@2."""
    if output is None:
        return 0.0
    return output.get("precision_at_2", 0.0)

def eval_mrr(output: dict) -> float:
    """Extract pre-computed MRR."""
    if output is None:
        return 0.0
    return output.get("mrr", 0.0)

def eval_ndcg(output: dict) -> float:
    """Extract pre-computed NDCG@2."""
    if output is None:
        return 0.0
    return output.get("ndcg_at_2", 0.0)

def eval_recall_pass(output: dict) -> bool:
    """Is recall above 50%?"""
    if output is None:
        return False
    return output.get("recall_at_2", 0.0) > 0.5

def eval_coverage_pass(output: dict) -> bool:
    """Is planner coverage above 50%?"""
    if output is None:
        return False
    return output.get("planner_coverage", 0.0) > 0.5

def eval_planner_specificity(output: dict) -> float:
    """Extract pre-computed planner specificity score."""
    if output is None:
        return 0.0
    return output.get("planner_specificity", 0.0)

# Question Image Citation Evaluators (replacing legacy ROI evaluators)
def eval_qi_citation_coverage(output: dict) -> float:
    """Coverage: Did samples with images cite those images?
    Returns 1.0 if:
    - Sample has Question Image citations, OR
    - Sample has no images (not applicable)
    Returns 0.0 if sample has images but no QI citations.
    """
    if output is None:
        return 0.0
    return output.get("qi_citation_coverage", 0.0)

def eval_qi_citation_count(output: dict) -> int:
    """Count of [Question Image N] citations in reasoning."""
    if output is None:
        return 0
    return output.get("qi_citation_count", 0)

def eval_qi_citation_precision(output: dict) -> float:
    """Precision: Are Question Image citation IDs valid (within range)?"""
    if output is None:
        return 0.0
    return output.get("qi_citation_precision", 0.0)

def eval_num_question_images(output: dict) -> int:
    """Number of question images available for this sample."""
    if output is None:
        return 0
    return output.get("num_question_images", 0)

# Solver evaluators (extract pre-computed values from output dict)
def eval_cave_score(output: dict) -> float:
    """Extract pre-computed CaVeScore."""
    if output is None:
        return 0.0
    return output.get("cave_score", 0.0)

def eval_accuracy(output: dict) -> float:
    """Extract pre-computed accuracy."""
    if output is None:
        return 0.0
    return output.get("accuracy", 0.0)

def eval_text_citation_precision(output: dict) -> float:
    """Extract pre-computed text citation precision."""
    if output is None:
        return 0.0
    return output.get("text_citation_precision", 0.0)

def eval_ais(output: dict) -> float:
    """Extract pre-computed AIS (Attribution Score)."""
    if output is None:
        return 0.0
    return output.get("ais", 0.0)

def eval_hallucination_rate(output: dict) -> float:
    """Extract pre-computed hallucination rate."""
    if output is None:
        return 0.0
    return output.get("hallucination_rate", 0.0)

def eval_citation_precision(output: dict) -> float:
    """Extract pre-computed combined citation precision."""
    if output is None:
        return 0.0
    return output.get("citation_precision", 0.0)

def eval_citation_recall(output: dict) -> float:
    """Extract pre-computed citation recall."""
    if output is None:
        return 0.0
    return output.get("citation_recall", 0.0)

def eval_grounding_score(output: dict) -> float:
    """Extract pre-computed evidence grounding score."""
    if output is None:
        return 0.0
    return output.get("grounding_score", 0.0)

def eval_is_grounded(output: dict) -> bool:
    """Is the final answer grounded in cited evidence?"""
    if output is None:
        return False
    return bool(output.get("is_grounded", False))

def eval_decision_correct(output: dict) -> float:
    """Was verifier's decision correct?"""
    if output is None:
        return 0.0
    return output.get("decision_correct", 0.0)

def eval_hallucination_detection_correct(output: dict) -> float:
    """Did verifier correctly identify hallucinations?"""
    if output is None:
        return 0.0
    return output.get("hallucination_detection_correct", 0.0)

def eval_confidence_appropriate(output: dict) -> float:
    """Was verifier's confidence level appropriate?"""
    if output is None:
        return 0.0
    return output.get("confidence_appropriate", 0.0)

def eval_feedback_quality(output: dict) -> float:
    """Quality of verifier's feedback for retries."""
    if output is None:
        return 0.0
    return output.get("feedback_quality", 0.0)

ALL_EVALUATORS = [
    eval_planner_hit,
    eval_planner_coverage,
    eval_subquery_count,
    eval_recall,
    eval_precision,
    eval_mrr,
    eval_ndcg,
    eval_recall_pass,
    eval_coverage_pass,
    eval_planner_specificity,
    # Question Image citation evaluators (NEW - replacing ROI evaluators)
    eval_qi_citation_coverage,
    eval_qi_citation_count,
    eval_qi_citation_precision,
    eval_num_question_images,
    # Solver
    eval_cave_score,
    eval_accuracy,
    eval_text_citation_precision,
    # eval_roi_citation_precision,
    eval_citation_precision,
    eval_citation_recall,
    eval_ais,
    eval_hallucination_rate,
    eval_grounding_score,
    eval_is_grounded,
    # Verifier
    eval_decision_correct,
    eval_hallucination_detection_correct,
    eval_confidence_appropriate,
    eval_feedback_quality,
]

print(f"Defined {len(ALL_EVALUATORS)} evaluators")

torch.cuda.empty_cache()
gc.collect()

_index_dir = os.path.join(PROJECT_DIR, "indexes", DATASET_NAME)
os.makedirs(_index_dir, exist_ok=True)

# Build indexes if they don't exist yet OR if the dataset has grown since
# the index was last built.  Without the size check, adding more rows to the
# CSV would silently use a stale index that covers fewer documents.
def _index_needs_rebuild(expected_rows: int) -> bool:
    index_path = os.path.join(_index_dir, "text_index.faiss") 
    if not os.path.exists(index_path):
        return True
    try:
        idx = faiss.read_index(index_path)                    
        if idx.ntotal < expected_rows:
            print(f"Index has {idx.ntotal} vectors but dataset has {expected_rows} "
                  f"text rows — rebuilding.")
            return True
    except Exception:
        return True
    return False

num_text_rows = len(df)  # rough lower bound; actual text rows = 1 per CSV row
if _index_needs_rebuild(num_text_rows):
    from utils import build_text_index
    print("Building indexes (first run or stale index)...")
    build_text_index(_CSV_MAP[DATASET_NAME], index_dir=_index_dir)
    print("Indexes built!")

# Load pre-built search indexes for fast retrieval
print("Loading indexes...")
text_index = faiss.read_index(os.path.join(_index_dir, "text_index.faiss"))
data = pd.read_csv(os.path.join(_index_dir, "multimodal_embeddings.csv"))

# Model loading — conditional on pipeline mode and model variant
#  PIPELINE           planner    solver    verifier
#  full               ✓          ✓         ✓
#  retrieval-solver   ✓          ✓         ✗
#  solver-only        ✗          ✓         ✗
#  no-citation-inj.   ✓          ✓         ✓
_needs_planner  = PIPELINE_MODE != "solver-only"
_needs_verifier = PIPELINE_MODE in ("full", "no-citation-injector")

print("\nLoading models for the pipeline...")

# 1. Planner
# GPU0 also hosts CrossEncoder and SentenceTransformer (< 2GB combined)
# Qwen3-8B (planner) — pinned to GPU 0
# Qwen3   variant : unsloth/Qwen3-8B-bnb-4bit              (8 B, GPU 0)
#   • Qwen3 uses 8 B as its base size; confirm the exact Unsloth hub ID at
#     https://huggingface.co/unsloth before running.
#   • Qwen3 supports a "thinking" mode; for deterministic ablation output keep
#     do_sample=False and ensure the tokenizer chat template does NOT inject
#     <think> tokens (pass enable_thinking=False if the template supports it).
if _needs_planner:
    # Reverted from Qwen3-8B back to Qwen2.5-7B-Instruct.
    # Reason: Qwen3-8B generated only 2-3 subqueries per question due to two bugs:
    #   1. Thinking-mode leakage: Qwen3 emits internal chain-of-thought into output
    #      which the parser mistook for query strings.
    #   2. Prompt mismatch: Qwen3 collapsed the structured-list prompt into a minimal
    #      template, producing only "X definition properties characteristics" queries.
    # Result: Planner Hit Rate dropped from 63% → 16%, AIS 62% → 20%,
    #         Hallucination Rate 38% → 80% across 21k ScienceQA examples.
    # Qwen2.5-7B reliably generates 7-8 diverse, targeted subqueries under this prompt.
    # _planner_model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    _planner_model_id = "/projects/cave-vlm-cot/hf_cache/hub/models--unsloth--Qwen2.5-7B-Instruct-bnb-4bit/snapshots/bdd404162d94997f390efbfa660eb3f21cbbc81d"
    print(f"1. Loading Qwen2.5-7B-Instruct for Planner (reverted from Qwen3-8B)...")

    planner_model, planner_tokenizer = FastLanguageModel.from_pretrained(
        model_name=_planner_model_id,
        max_seq_length=4096,
        load_in_4bit=True,
        device_map={"": "cuda:0"}, 
    )
    FastLanguageModel.for_inference(planner_model)
    # max_new_tokens raised from 128→512: Qwen2.5 needs more tokens to generate
    # 7-8 diverse subqueries; 128 was causing truncation with Qwen3 and is too
    # tight for the longer JSON arrays Qwen2.5 produces.
    planner_kwargs = dict(do_sample=False, max_new_tokens=512)

else:
    print("1. Skipping Planner model (not needed for solver-only pipeline)")
    planner_model = planner_tokenizer = None
    planner_kwargs = {}

# 2. Solver — Llama-3.2V-11B — pinned to GPU1 (~22GB bfloat16, leaves 58GB headroom)
print("2. Loading Llama-3.2V-11B for Solver...")
solver_model_id = "zhangsongbo365/Llama-3.2V-11B-cot-nf4"
# Pin solver to GPU1 when other models use GPU0, otherwise GPU0
_solver_gpu = "cuda:1" 
# if PIPELINE_MODE == "solver-only" else "cuda:1"
solver_model = MllamaForConditionalGeneration.from_pretrained(
    solver_model_id,
    use_safetensors=True,
    device_map={"": _solver_gpu},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
solver_processor = AutoProcessor.from_pretrained(solver_model_id, cache_dir=CACHE_DIR)
# temperature=0.3 caused 26 new INCONCLUSIVEs: solver conclusions varied
# enough that the verifier emitted INCONCLUSIVE on previously-stable questions.
# temperature=0.1 keeps light stochasticity
# while dramatically reducing format variance in CONCLUSION output.
# Greedy decoding (do_sample=False) ensures deterministic outputs across ablation runs.
solver_kwargs = dict(do_sample=False, max_new_tokens=1024)
# , temperature=0.1, top_p=0.95

# 3. Verifier — Qwen2.5-VL judge.
# A40 recommendation for 5k-sample runs:
#   Qwen/Qwen2.5-VL-32B-Instruct, 4-bit, pinned to cuda:2.
# This is a large enough jump from 7B to reduce the observed false-accept
# behavior while still fitting on one 48GB A40 with conservative image tokens.
if _needs_verifier:
    _verifier_model_id = VERIFIER_MODEL_ID
    print(f"3. Loading verifier: {_verifier_model_id} ({VERIFIER_QUANTIZATION}, {VERIFIER_DEVICE_MAP})...")
    # _verifier_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    # _verifier_model_id = "/projects/cave-vlm-cot/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
    # _verifier_model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
    # _verifier_model_id = "/projects/cave-vlm-cot/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5"

    verifier_processor = AutoProcessor.from_pretrained(
        _verifier_model_id,
        cache_dir=VERIFIER_CACHE,
        min_pixels=256 * 28 * 28,
        max_pixels=VERIFIER_MAX_PIXELS * 28 * 28,
        trust_remote_code=True,
    )

    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    VerifierModelClass = Qwen2_5_VLForConditionalGeneration

    verifier_load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if VERIFIER_DEVICE_MAP == "auto":
        verifier_load_kwargs["device_map"] = "auto"
    else:
        verifier_load_kwargs["device_map"] = {"": VERIFIER_DEVICE}

    if VERIFIER_QUANTIZATION == "4bit":
        verifier_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif VERIFIER_QUANTIZATION == "8bit":
        verifier_load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    verifier_model = VerifierModelClass.from_pretrained(
        _verifier_model_id,
        cache_dir=VERIFIER_CACHE,
        **verifier_load_kwargs,
    )
    verifier_model.eval()
else:
    print("3. Skipping Verifier model (not needed for this pipeline)")
    verifier_model = verifier_processor = None

print("\n All models loaded successfully!")

# Build the graph matching the requested pipeline mode
print(f"\nBuilding graph: pipeline={PIPELINE_MODE}, model_variant={MODEL_VARIANT} ...")

if PIPELINE_MODE == "full":
    cave_vlm_cot_app = build_cave_vlm_cot_graph(
        planner_model=planner_model,
        planner_tokenizer=planner_tokenizer,
        planner_kwargs=planner_kwargs,
        text_index=text_index,
        data=data,
        solver_model=solver_model,
        solver_processor=solver_processor,
        solver_kwargs=solver_kwargs,
        verifier_model=verifier_model,
        verifier_processor=verifier_processor,
        retrieval_k=5,
    )

elif PIPELINE_MODE == "retrieval-solver":
    cave_vlm_cot_app = build_retrieval_solver_graph(
        planner_model=planner_model,
        planner_tokenizer=planner_tokenizer,
        planner_kwargs=planner_kwargs,
        text_index=text_index,
        data=data,
        solver_model=solver_model,
        solver_processor=solver_processor,
        solver_kwargs=solver_kwargs,
        retrieval_k=5,
    )

elif PIPELINE_MODE == "solver-only":
    cave_vlm_cot_app = build_solver_only_graph(
        solver_model=solver_model,
        solver_processor=solver_processor,
        solver_kwargs=solver_kwargs,
    )

elif PIPELINE_MODE == "no-citation-injector":
    cave_vlm_cot_app = build_pipeline_without_citation_injector(
        planner_model=planner_model,
        planner_tokenizer=planner_tokenizer,
        planner_kwargs=planner_kwargs,
        text_index=text_index,
        data=data,
        solver_model=solver_model,
        solver_processor=solver_processor,
        solver_kwargs=solver_kwargs,
        verifier_model=verifier_model,
        verifier_processor=verifier_processor,
        retrieval_k=5,
    )

else:
    raise ValueError(f"Unknown --pipeline value: {PIPELINE_MODE!r}")
    
print(" Graph compiled and ready!")

# Default output dict used on error — keeps Phoenix from seeing missing keys
_DEFAULT_OUTPUT = {
    "pid": "",
    "choices": [],
    "subqueries": [],
    "retrieved_chunks": {},
    "verified_answer": "",
    "planner_hit": False,
    "planner_coverage": 0.0,
    "planner_specificity": 0.0,
    "recall_at_2": 0.0,
    "precision_at_2": 0.0,
    "mrr": 0.0,
    "ndcg_at_2": 0.0,
    # Question Image metrics (replacing ROI metrics)
    "qi_citation_coverage": 0.0,
    "qi_citation_count": 0,
    "qi_citation_precision": 0.0,
    "num_question_images": 0,
    "cave_score": 0.0,
    "text_citation_precision": 0.0,
    "accuracy": 0.0,
    "ais": 0.0,
    "hallucination_rate": 0.0,
    "citation_precision": 0.0,
    "citation_recall": 0.0,
    "grounding_score": 0.0,
    "is_grounded": False,
    "verdict": "",
    "confidence": "",
    "decision_correct": 0.0,
    "hallucination_detection_correct": 0.0,
    "confidence_appropriate": 0.0,
    "feedback_quality": 0.0,
}
DEBUG_LOG_PATH = os.path.join(PROJECT_DIR, "logs/experiments_debug.log")

def cave_vlm_cot_with_verifier_task(input: dict, expected: dict) -> dict:
    """
    Run the CaVe-VLM-CoT pipeline with verification and retry loop.

    The graph (app) handles: Extractor → Retriever → Solver → Verifier
    - If VERIFIED: Done
    - If REJECTED: Retry extractor with feedback (up to 3 attempts)
    """
    pid = input.get("pid", "unknown")
    
    def _run(state):
        torch.cuda.empty_cache()
        gc.collect()
        result = cave_vlm_cot_app.invoke(state)
        torch.cuda.empty_cache()
        gc.collect()
        return result

    def _build_state():
        choices = safe_parse_json(input.get("choices"), default=[])
        image_paths = safe_parse_json(input.get("image_paths"), default=[])
        img_captions = safe_parse_json(input.get("img_captions"), default={})
        img_ocr_data = safe_parse_json(input.get("img_ocr"), default={})
        answer_index = int(input.get("answer", 0))
        return State(
            pid=input["pid"],
            question=input["question"],
            hint=input.get("hint", ""),
            image_paths=image_paths,
            choices=choices,
            lecture=input.get("lecture", ""),
            img_captions=img_captions,
            img_ocr=img_ocr_data,
            answer=answer_index,
            gold_answer=expected.get("gold_answer", ""),
            subject=input.get("subject", ""),
            topic=input.get("topic", ""),
            skill=input.get("skill", ""),
            category=input.get("category", ""),
        ), choices

    try:
        print(f"Processing PID {pid}")
        state, choices = _build_state()
        print(f"Question: {state.question[:80]}...")

        try:
            result_state = _run(state)
        except RuntimeError as oom:
            if "CUDA out of memory" not in str(oom):
                raise
            # OOM on attempt 1 — clear aggressively and retry once
            print(f"  [OOM attempt 1] PID {pid} — clearing memory, retrying...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(15)
            # Retry — same graph, just cleaner memory state
            result_state = _run(state)
            print(f"  [OOM retry succeeded] PID {pid}")

        if isinstance(result_state, dict):
            result_state = State(**result_state)

        recall_result = recall_at_k(result_state, k=2)
        print(f"Final recall: {recall_result['recall']:.2%}")

        cave_score = compute_cave_score(result_state)
        verifier_quality = evaluate_verifier_quality(result_state, cave_result=cave_score)

        print(
            f"  Verdict: {result_state.verdict} ({result_state.confidence} confidence) "
            f"| Retries: {result_state.retry_count}"
        )

        return {
            "pid": result_state.pid,
            "choices": choices,
            "subqueries": result_state.subqueries,
            "retrieved_chunks": result_state.retrieved_chunks,
            "verified_answer": result_state.verifier_answer,
            "planner_hit": planner_hit_rate(result_state, k=2),
            "planner_coverage": planner_coverage_score(result_state),
            "planner_specificity": planner_specificity_score(result_state),
            "recall_at_2": recall_result["recall"],
            "precision_at_2": precision_at_k(result_state, k=2),
            "mrr": mean_reciprocal_rank(result_state),
            "ndcg_at_2": ndcg_at_k(result_state, k=2),
            "qi_citation_coverage": cave_score["qi_citation_coverage"],
            "qi_citation_count": cave_score["qi_citation_count"],
            "qi_citation_precision": cave_score["qi_citation_precision"],
            "num_question_images": cave_score["num_question_images"],
            "cave_score": cave_score["cave_score"],
            "text_citation_precision": cave_score["text_citation_precision"],
            "accuracy": cave_score["accuracy"],
            "ais": cave_score["ais"],
            "hallucination_rate": cave_score["hallucination_rate"],
            "citation_precision": cave_score["citation_precision"],
            "citation_recall": cave_score["citation_recall"],
            "grounding_score": cave_score["grounding_score"],
            "is_grounded": cave_score["is_grounded"],
            "verdict": verifier_quality["verdict"],
            "confidence": verifier_quality["confidence"],
            "decision_correct": verifier_quality["decision_correct"],
            "hallucination_detection_correct": verifier_quality["hallucination_detection_correct"],
            "confidence_appropriate": verifier_quality["confidence_appropriate"],
            "feedback_quality": verifier_quality["feedback_quality"],
        }

    except Exception as e:
        # Reaches here only if: non-OOM error, OR OOM retry also failed
        is_oom = "CUDA out of memory" in str(e)
        if is_oom:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"  [OOM both attempts failed] PID {pid} — returning default")
        else:
            print(f"Error processing example {pid}: {e}")

        tb_str = traceback.format_exc()
        traceback.print_exc()

        try:
            os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "experiments.py:cave_vlm_cot_with_verifier_task",
                    "pid": pid,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": tb_str,
                }) + "\n")
        except Exception:
            pass

        try:
            span = otel_trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("task.error_type", type(e).__name__)
                span.set_attribute("task.error_message", str(e))
                span.set_attribute("task.traceback", tb_str[:2000])
        except Exception:
            pass

        return {**_DEFAULT_OUTPUT, "pid": pid}

print("Task function defined: cave_vlm_cot_with_verifier_task")

# Wrap task in suppress_tracing if --no-traces is set.
# This stops span export to Phoenix (saving storage) while experiment
# results (datasets, evaluator scores) are still recorded via the Client API.
if NO_TRACES:
    try:
        from phoenix.trace import suppress_tracing
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def suppress_tracing():
            """Fallback: disable tracing via OpenTelemetry context."""
            from opentelemetry.context import attach, detach, set_value
            token = attach(set_value("suppress_instrumentation", True))
            try:
                yield
            finally:
                detach(token)
    _unwrapped_task = cave_vlm_cot_with_verifier_task
    def cave_vlm_cot_with_verifier_task(input: dict, expected: dict) -> dict:
        with suppress_tracing():
            return _unwrapped_task(input, expected)
    print("[Traces]   Task wrapped with suppress_tracing — no spans will be exported")

# Run experiment
print("\nRunning experiment...")
print("Experiment: Cite-Verify VLM CoT (with feedback loop)")

# Dry run first
print("\n1. Running dry run...")
dry_run = experiments_client.experiments.run_experiment(
    dataset=cave_dataset,
    task=cave_vlm_cot_with_verifier_task,
    evaluators=ALL_EVALUATORS,
    dry_run=1,
)
print("Dry run successful")

# Full experiment
print("\n2. Running full experiment...")
experiment_with_verifier = experiments_client.experiments.run_experiment(
    dataset=cave_dataset,
    task=cave_vlm_cot_with_verifier_task,
    evaluators=ALL_EVALUATORS,
    experiment_name=(
        f"CaVe-VLM-CoT-{DATASET_NAME}-{PIPELINE_MODE}-{MODEL_VARIANT}"
        f"-n{MAX_SAMPLES or 'all'}-seed{SAMPLE_SEED}"
        f"-shard{SHARD_INDEX}-of-{NUM_SHARDS}"
    ),
    experiment_description=(
        f"Pipeline: {PIPELINE_MODE} | Models: {MODEL_VARIANT} | "
        f"Dataset cap: {MAX_SAMPLES or 'all'} | Sample seed: {SAMPLE_SEED} | "
        f"Verifier: {VERIFIER_MODEL_ID} ({VERIFIER_QUANTIZATION}) | "
        f"Shard {SHARD_INDEX+1}/{NUM_SHARDS}"
    ),
)

print("\nExperiment complete — results saved to Phoenix.")
print("Key metrics to check:")
print("  eval_planner_hit          — % of questions where retrieval hit gold answer")
print("  eval_accuracy             — % correct final answers")
print("  eval_cave_score           — CaVeScore (citation + accuracy)")
print("  eval_qi_citation_coverage — % of image samples with Question Image citations")
print("  eval_qi_citation_count    — Average number of QI citations per sample")
print("  eval_hallucination_rate   — % of hallucinated claims")
print("  eval_decision_correct     — % of correct verifier decisions")