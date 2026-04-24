# CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework

A five-stage agentic-RAG pipeline for grounded multimodal reasoning on science QA. Every answer is backed by cited evidence and verified by a second VLM before being returned — with an automatic feedback loop that retries retrieval when hallucinations are detected.

---

## Architecture
 
![CaVe-VLM-CoT Architecture](CAVE.png)

Each stage is a node in a **LangGraph** state machine. The shared `State` Pydantic object carries inputs, intermediate outputs, citations, verdicts, and retry history through the entire graph.

---

## Key Features

- **Grounded reasoning** — the solver must cite `[Text Evidence N]` for every text-based claim and `[Question Image N]` for every visual observation. Uncited steps are flagged as hallucinations.
- **Post-hoc citation injection** — a cross-encoder aligns solver claims to retrieved chunks after generation, recovering citations the VLM missed.
- **Two-model verification** — a separate VLM (Qwen2.5-VL-32B-Instruct) re-reads the evidence and checks each citation before accepting the answer.
- **Feedback loop** — when the verifier rejects an answer it returns structured feedback (hallucination type, missing evidence, failed queries) that the extractor uses to generate better queries on the next attempt.
- **Multimodal retrieval** — question images are passed directly to both solver and verifier as `[Question Image N]` references; the knowledge base is indexed with `all-MiniLM-L6-v2` (text).
- **Composite scoring** — CaVeScore = `0.4 × accuracy + 0.2 × citation_precision + 0.2 × citation_recall + 0.1 × AIS + 0.1 × grounding_score`.
- **Weight sensitivity analysis** — 7 alternative CaVeScore weight configurations validate that ablation rankings are robust to weight perturbation.


---

## Installation

Requires Python 3.11 (via module system) and **three A40 48 GB GPUs** for the full pipeline, or one A40 for solver-only.

```bash
git clone https://github.com/VectorInstitute/cave-vlm-cot.git
cd cave-vlm-cot/src/cite_verify_vlm_cot

# Load required modules (Vector Institute HPC)
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11 scipy-stack faiss/1.12.0

# Activate virtual environment
source /fs02/home/sneharao/cave-vlm-cot/env/bin/activate
export PYTHONPATH=/projects/cave-vlm-cot/env/lib/python3.11/site-packages:$PYTHONPATH

```

### Environment variables

Create a `.env` file in the working directory (`src/cite_verify_vlm_cot/`). The SLURM scripts source this automatically.

```bash
PHOENIX_API_KEY=<your-arize-phoenix-api-key>
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
HF_HOME=/projects/cave-vlm-cot/hf_cache
```

The following are set automatically by the SLURM scripts and do not need to appear in `.env`:

```bash
CAVE_PROJECT_DIR=/projects/cave-vlm-cot
TRANSFORMERS_OFFLINE=1        # compute nodes are offline
HF_HUB_OFFLINE=1
UNSLOTH_DISABLE_STATISTICS=1
TQDM_DISABLE=1
PYTORCH_ALLOC_CONF=expandable_segments:True
MMMU_IMAGE_ROOT=/projects/cave-vlm-cot/processed_images_mmmu

```

---

## Data preparation

### ScienceQA

```bash
export SCIENCEQA_ROOT=/projects/cave-vlm-cot/scienceqa
python data-preparation.py
# Outputs: scienceqa_augmented.csv, processed_images/
```

The FAISS text index is built automatically on the first run if not present.

### MMMU

```bash
python data-preparation-mmmu.py
# Outputs: mmmu_augmented.csv, /projects/cave-vlm-cot/processed_images_mmmu/

# Subset of subjects (for faster testing):
MMMU_SUBJECTS="Math,Physics" python data-preparation-mmmu.py
```

> **Note:** Data preparation downloads from HuggingFace Hub and must be run from a login node (`blogin02`). Compute nodes have `HF_HUB_OFFLINE=1` set by the SLURM scripts.

---

## Running experiments

### Ablation configurations

| Task IDs  | Pipeline           | Dataset    | Notes                        |
|-----------|--------------------|------------|------------------------------|
| 0 – 39    | `solver-only`      | scienceqa  | Baseline — no retrieval or verification |
| 40 – 79   | `retrieval-solver` | scienceqa  | Extractor + Retriever + Solver only |
| 80 – 119  | `full`             | scienceqa  | Complete 5-stage CaVe-VLM-CoT pipeline |

### Single run (sequential, no SLURM)

```bash
# Full pipeline, 5 questions, shard 0 of 40, no Phoenix tracing
python experiments.py \
    --pipeline full \
    --dataset scienceqa \
    --shard 0 --num-shards 40 \
    --max-samples 5 \
    --no-traces
```

### Parallel sharding via SLURM (recommended)

```bash
cd batch_files
# Submit all three ScienceQA ablations (120 tasks: 3 experiments × 40 shards)
sbatch cave_array.sh

# Or submit one experiment group at a time:
sbatch --array=0-39   cave_array.sh   # solver-only
sbatch --array=40-79  cave_array.sh   # retrieval-solver
sbatch --array=80-119 cave_array.sh   # full CaVe-VLM-CoT

# Resume a single failed task (e.g. task 47):
sbatch --array=47 cave_array.sh

# MMMU full-pipeline run (20 shards × 25 questions = 500 questions)
sbatch cave_mmmu_array.sh

```

### CLI flags

| Flag | Description |
|------|-------------|
| `--pipeline` | `full`, `retrieval-solver`, `solver-only` |
| `--dataset` | `scienceqa`, `mmmu` |
| `--shard N` | 0-based shard index |
| `--num-shards N` | Total shards (40 for ScienceQA, 20 for MMMU) |
| `--max-samples N` | Cap on questions per shard (default: 1000; env: `CAVE_MAX_SAMPLES`) |
| `--sample-seed N` | RNG seed for sample selection (default: 42) |
| `--verifier-model` | Path or HF model ID for the verifier |
| `--verifier-quantization` | `4bit`, `8bit`, or `none` (default: `4bit`) |
| `--verifier-device` | Which GPU to pin the verifier to (default: `cuda:2`) |
| `--verifier-device-map` | `single` (pin to device) or `auto` (distribute) |
| `--verifier-max-pixels` | Max image resolution for verifier (default: `768`) |
| `--no-traces` | Suppress OpenTelemetry export (recommended for large runs) |

### Overriding the Verifier model

The scripts default to Qwen2.5-VL-32B-Instruct (4-bit). To switch verifier without editing the script:

```bash
export CAVE_VERIFIER_MODEL=/projects/cave-vlm-cot/hf_cache/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da
export CAVE_VERIFIER_QUANTIZATION=4bit
sbatch cave_array.sh
```

---

## Monitoring and verifying jobs

```bash
# Live queue
squeue -u $USER

# Accounting for a completed array
sacct -j <JOBID> --format=JobID,State,ExitCode,Start,End,Elapsed
# Healthy: State=COMPLETED, ExitCode=0:0

# Scan logs for errors
grep -l "Error\|Traceback\|CUDA out of memory" /projects/cave-vlm-cot/logs/cave_<JOBID>_*.out

# Check for truncated output files (each shard should have 26 lines: header + 25 rows)
for f in /projects/cave-vlm-cot/outputs/full-shard*.csv; do
    lines=$(wc -l < "$f")
    [ "$lines" -lt 26 ] && echo "SHORT: $f ($lines lines)"
done
```

---

## Aggregating results

After all tasks for a configuration are complete, merge shards with `aggregate_shards.py`. Always quote the glob to prevent shell expansion ordering issues.

```bash
# ScienceQA ablations
python aggregate_shards.py 'outputs/full-shard*.csv'             --save-csv outputs/full-results.csv
python aggregate_shards.py 'outputs/solver-only-shard*.csv'      --save-csv outputs/solver-only-results.csv
python aggregate_shards.py 'outputs/retrieval-solver-shard*.csv' --save-csv outputs/retrieval-solver-results.csv

# MMMU
python aggregate_shards.py 'outputs/mmmu-shard*.csv'             --save-csv outputs/mmmu-results.csv
```

---

## Results

### ScienceQA ablation (n = 1,000)

| Configuration      | Accuracy | AIS   | Cit. Precision | CaVeScore |
|--------------------|----------|-------|----------------|-----------|
| Solver-Only        | 0.730    | 0.125 | 0.000          | 0.314     |
| Retriever-Solver   | 0.704    | 0.318 | 0.249          | 0.404     |
| CaVe-VLM-CoT (full)| **0.871**| **0.408** | **0.543** | **0.566** |

### Cross-dataset generalisation

| Dataset            | Accuracy | AIS   | Cit. Precision | CaVeScore |
|--------------------|----------|-------|----------------|-----------|
| ScienceQA (n=1,000)| 0.871    | 0.408 | 0.543          | 0.566     |
| MMMU (n=500)       | 0.552    | 0.153 | 0.325          | 0.357     |

### Verifier calibration (Qwen2.5-VL-32B vs prior Qwen3-VL-8B)

| Metric                    | Qwen3-VL-8B | Qwen2.5-VL-32B |
|---------------------------|-------------|----------------|
| Decision Correctness      | 0.099       | **0.525**      |
| Feedback Quality          | 0.000       | **0.274**      |
| Confidence Appropriateness| 0.131       | **0.494**      |
| Accuracy gain over RS     | +4.3 pp     | **+19.2 pp**   |

---

## Pipeline components

### 1. Extractor (`extractor/planner.py`)

Converts a raw MCQ into up to 8 search queries using **Qwen2.5-7B-Instruct** (4-bit via Unsloth). The few-shot prompt ends with `[` to prime the model into producing a structured list. Query quality is enforced by structural validation (`_is_valid_query`) and three-tier parse fallback. On retry, verifier feedback and a list of failed queries are injected into the prompt.

**Guardrails:** structural query validation · 3-way parse fallback · deterministic choice-discriminating queries · keyword-only fallback when LLM output is unparseable · max 8 queries

### 2. Retriever (`retriever/retriever.py`)

Per subquery: hybrid local retrieval (dense FAISS + BM25 + RRF fusion), followed by parallel DuckDuckGo web search with choice-augmented variants, then cross-encoder reranking.

**Guardrails:** LRU cache (4,096 entries) on web search · exponential DDG backoff (1 s → 2 s → 4 s) · 0.3 s stagger between parallel DDG submissions · doubled web budget for natural science questions · local docs admitted only when cross-encoder score > 0

### 3. Solver (`solver/solver.py`)

**Llama-3.2V-11B-CoT** (NF4 quantised) generates a structured chain-of-thought with mandatory citation anchors. Two prompt templates: text-only (`SUMMARY → REASONING → CONCLUSION`) and image-present (`OBSERVATIONS → REASONING → CONCLUSION`). A cross-encoder consistency check corrects the stated answer letter if reasoning supports a different choice by a margin > 0.5.

**Guardrails:** structured XML tags enforce output format · mandatory `[Text Evidence N]` and `[Question Image N]` citations · cross-encoder consistency check on conclusion · multi-pattern answer extraction with tail fallback · max 5 question images

### 4. Citation Injector (`solver/citation_injector.py`)

Post-hoc grounding step between solver and verifier. Splits reasoning into claims and uses the cross-encoder (`ms-marco-MiniLM-L-6-v2`) to match each uncited claim to the best evidence chunk (threshold 0.4). Handles domain-knowledge steps by running targeted web search for `"From domain knowledge, <claim>"` sentences.

**Guardrails:** DK enrichment gated on KB coverage (< 2 substantive chunks) · DK lookups capped at 3 per question · min claim length 20 chars · skips if > 50% already cited (unless conclusion is uncited) · observation-block claims skip text matching (image-only evidence)

### 5. Verifier (`verifier/verifier.py`)

**Qwen2.5-VL-32B-Instruct** (4-bit NF4) runs a structured hallucination check: for each cited claim it verifies whether the referenced evidence actually supports the stated fact, checks citation indices are in range, and flags image description contradictions. Shares the same evidence list as the solver (via `_build_text_chunk_list`) so citation numbers are never misaligned.

**Guardrails:** INCONCLUSIVE-REJECT downgraded to VERIFIED/LOW · VERIFIED-INCONCLUSIVE recovers solver letter · UNKNOWN verdict fallback · max 4 images · max 3 retry attempts

---

## Evaluation metrics

All 26 evaluators are defined in `evaluations.py` and logged to Arize Phoenix per question.

| Metric | Description |
|--------|-------------|
| `accuracy` | Final answer matches gold label |
| `cave_score` | Composite: `0.4×acc + 0.2×cite_prec + 0.2×cite_rec + 0.1×AIS + 0.1×grounding` |
| `citation_precision` | NLI check: does cited evidence entail the claim? |
| `citation_recall` | Fraction of factual sentences that carry a citation |
| `ais` | Attribution score: fraction of reasoning steps attributable to retrieved evidence |
| `hallucination_rate` | `1 − AIS` |
| `grounding_score` | NLI: does retrieved evidence entail the final answer? |
| `planner_hit` | Retrieved chunks include gold-answer content |
| `planner_coverage` | Fraction of answer choices covered by subqueries |
| `recall@2 / precision@2 / MRR / NDCG@2` | Standard retrieval metrics |
| `qi_citation_coverage` | Fraction of image questions with ≥ 1 `[Question Image N]` citation |
| `qi_citation_precision` | `[Question Image N]` IDs are in-range |
| `decision_correct` | Verifier verdict matches ground-truth hallucination label |
| `feedback_quality` | Verifier feedback correctly identifies issues and is actionable |

NLI is computed by `cross-encoder/nli-deberta-v3-base`.

---

## GPU allocation

| GPU | Model | VRAM |
|-----|-------|------|
| `cuda:0` | Qwen2.5-7B-Instruct extractor (4-bit via Unsloth) + cross-encoder + SentenceTransformer | ~8 GB |
| `cuda:1` | Llama-3.2V-11B-CoT solver (NF4) | ~22 GB |
| `cuda:2` | Qwen2.5-VL-32B-Instruct verifier (4-bit) | ~48 GB |

For solver-only runs, the solver loads on `cuda:0` and only 1 GPU is needed.

---

## Weight sensitivity analysis

CaVeScore rankings are validated under 7 alternative weight configurations (n = 1,000 ScienceQA):

| Config | w_acc | w_cprec | w_crec | w_ais | w_gnd | Solver-Only | Retriever-Solver | CaVe-VLM-CoT | Gap |
|--------|-------|---------|--------|-------|-------|-------------|------------------|--------------|-----|
| default | 0.40 | 0.20 | 0.20 | 0.10 | 0.10 | 0.314 | 0.404 | 0.566 | +0.251 |
| accuracy_heavy | 0.60 | 0.10 | 0.10 | 0.10 | 0.10 | 0.460 | 0.508 | 0.661 | +0.201 |
| citation_heavy | 0.20 | 0.30 | 0.30 | 0.10 | 0.10 | 0.168 | 0.300 | 0.470 | +0.302 |
| uniform | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.191 | 0.311 | 0.452 | +0.261 |
| ais_heavy | 0.30 | 0.15 | 0.15 | 0.25 | 0.15 | 0.265 | 0.371 | 0.510 | +0.245 |
| grounding_heavy | 0.30 | 0.15 | 0.15 | 0.15 | 0.25 | 0.263 | 0.356 | 0.489 | +0.227 |
| recall_skewed | 0.35 | 0.10 | 0.35 | 0.10 | 0.10 | 0.278 | 0.362 | 0.504 | +0.226 |

No re-inference required — pure arithmetic re-weighting of pre-computed components. The ordering Solver-Only < Retriever-Solver < CaVe-VLM-CoT is preserved under every configuration.

---

## Observability

All pipeline stages emit OpenTelemetry spans to Arize Phoenix via `BatchSpanProcessor`. Traced attributes include `planner.coverage_score`, `retriever.recall`, `verifier.verdict`, `verifier.hallucination_count`, and token counts for every LLM call. Use `--no-traces` to suppress span export on large runs while still recording experiment results.

---
## Project structure

```
cave-vlm-cot/
└── src/cite_verify_vlm_cot/
    ├── experiments.py              # Main experiment runner (entry point)
    ├── evaluations.py              # 26 evaluation metrics
    ├── aggregate_shards.py         # Merge per-shard CSVs into summary table
    ├── utils.py                    # State schema, FAISS indexing, helpers
    ├── prompts.py                  # All prompt templates
    ├── tracer.py                   # OpenTelemetry / Arize Phoenix integration
    ├── data-preparation.py         # ScienceQA augmentation (captions, OCR, resize)
    ├── data-preparation-mmmu.py    # MMMU augmentation
    ├── extractor/
    │   ├── planner.py              # Qwen2.5-7B query decomposition
    │   └── planner_evals.py        # Extractor metrics (coverage, hit rate, specificity)
    ├── retriever/
    │   ├── retriever.py            # Hybrid BM25 + FAISS + web search + reranking
    │   └── retriever_evals.py      # Retriever metrics (recall, precision, MRR, NDCG)
    ├── solver/
    │   ├── solver.py               # Llama-3.2V-11B-CoT reasoning
    │   ├── solver_evals.py         # CaVeScore, WEIGHT_CONFIGS
    │   └── citation_injector.py    # Post-hoc citation grounding
    ├── verifier/
    │   ├── verifier.py             # Qwen2.5-VL-32B verification + LangGraph builders
    │   └── verifier_evals.py       # Verifier quality metrics
    └── batch_files/
        ├── cave_array.sh           # SLURM array job — ScienceQA ablations (120 tasks)
        └── cave_mmmu_array.sh      # SLURM array job — MMMU full pipeline (20 tasks)
        └── preparation.sh          # SLURM job — ScienceQA dataset preparation
```
---
## Citation

```bibtex
@article{rao2026cave,
  title={CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework},
  author={Rao, Sneha and Raza, Shaina and Ramachandram, Dhanesh},
  year={2026}
}
```
---
## Useful resources

https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117 (Method 2)

https://blog.roboflow.com/image-search-engine-gaudi2/

https://cookbook.openai.com/examples/custom_image_embedding_search

https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669

https://medium.com/@aishikbhattacharjee98/reranking-using-cross-encoder-boost-your-rag-pipeline-accuracy-d2da22006dad

https://medium.com/@abheshith7/mastering-reranking-in-rag-from-basic-retrieval-to-advanced-methods-db297530361a

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct 
