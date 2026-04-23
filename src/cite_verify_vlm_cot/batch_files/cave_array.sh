#!/bin/bash
# cave_array.slurm — CaVe-VLM-CoT: 5k ScienceQA ablation suite
#
# Launches 3 experiments × 40 shards = 120
# Submit the 1k ScienceQA ablation suite
# each shard is 1/40
#
#   Task ID  │ EXPERIMENT_ID │ SHARD  │ --pipeline        │ --dataset  │ Notes
#   ─────────┼───────────────┼────────┼───────────────────┼────────────┼────────────────────
#     0– 39  │      0        │  0–39  │ solver-only       │ scienceqa  │ baseline
#    40– 79  │      1        │  0–39  │ retrieval-solver  │ scienceqa  │ retrieval ablation
#    80– 119 │      2        │  0–39  │ full              │ scienceqa  │ CaVe-VLM-CoT
#
# Submit the 1k ScienceQA ablation suite:
#   sbatch cave_array.sh
#
# Submit one experiment group:
#   sbatch --array=0-19  cave_array.sh   # solver-only
#   sbatch --array=20-39 cave_array.sh   # retrieval-solver
#   sbatch --array=40-59 cave_array.sh   # full
#
# Resume a single failed task (e.g. task 47):
#   sbatch --array=47 cave_array.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /projects/cave-vlm-cot/logs/cave_<jobid>_<taskid>.out

#SBATCH --job-name=cave-vlm-cot
#SBATCH --array=0-119                 # 3 experiments × 20 shards over 5k ScienceQA
#SBATCH --nodes=1                     # each task runs on its own node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12            # DDG web search uses ThreadPoolExecutor
#SBATCH --gres=gpu:a40:3          # 3 GPUs per task (planner/solver/verifier)
#SBATCH --mem=120G
#SBATCH --time=0-12:00:00             # each shard is 1/20 of the capped 5k dataset
#SBATCH --chdir=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
#SBATCH --output=/projects/cave-vlm-cot/logs/cave_%j_%a.out
#SBATCH --error=/projects/cave-vlm-cot/logs/cave_%j_%a.err

set -euo pipefail
cleanup() {
    echo "Compressing logs..."
    gzip -f "${LOGDIR}/cave_${SLURM_JOB_ID}_${TASK_ID}.out"
    gzip -f "${LOGDIR}/cave_${SLURM_JOB_ID}_${TASK_ID}.err"
}
trap cleanup EXIT
mkdir -p /projects/cave-vlm-cot/logs

# Paths — source code lives on home, large files on project storage
PROJECT_DIR=/projects/cave-vlm-cot

# Paths
WORKDIR=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
LOGDIR=${PROJECT_DIR}/logs
OUTDIR=${PROJECT_DIR}/outputs
mkdir -p "${LOGDIR}" "${OUTDIR}" "${PROJECT_DIR}/hf_cache" "${PROJECT_DIR}/indexes"

# Derive experiment config and shard from flat task ID
#
#   SLURM_ARRAY_TASK_ID: 0–59
#   EXPERIMENT_ID:       0–2
#   SHARD_ID:            0–19  (which twentieth of the capped 5k dataset)
#
TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=40
EXPERIMENT_ID=$(( TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
LAST_SHARD=$(( NUM_SHARDS - 1 ))   # = 19

# Experiment tables (indexed by EXPERIMENT_ID)
#
# PIPELINES and DATASETS must stay in sync — both indexed by EXPERIMENT_ID.
# The --dataset flag selects which augmented CSV (and which FAISS index
# sub-directory) experiments.py uses; it is not an env-var override any more.
#
PIPELINES=(
    "solver-only"             # 0  Solver only, no retrieval/verifier
    "retrieval-solver"        # 1  Planner -> Retriever -> Solver
    "full"                    # 2  Full CaVe-VLM-CoT
)

DATASETS=(
    "scienceqa"               # 0
    "scienceqa"               # 1
    "scienceqa"               # 2
)

EXPERIMENT_LABELS=(
    "ablation-solver-only-scienceqa-1k"
    "ablation-retrieval-solver-scienceqa-1k"
    "full-cave-vlm-cot-scienceqa-1k"
)

PIPELINE="${PIPELINES[$EXPERIMENT_ID]}"
DATASET="${DATASETS[$EXPERIMENT_ID]}"
EXP_LABEL="${EXPERIMENT_LABELS[$EXPERIMENT_ID]}"

# Modules
module purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6  # CUDA 12.2 matches the toolkit
# module load slurm/bonecho/25.05.2
module load arrow
module load python/3.11       # activates EBPYTHONPREFIXES
module load scipy-stack
module load faiss/1.12.0
export PYTHONPATH="${PROJECT_DIR}/env/lib/python3.11/site-packages:$PYTHONPATH"
# Python comes from the virtualenv below — no separate python module needed

source /fs02/home/sneharao/cave-vlm-cot/env/bin/activate

# Huggingface / PyTorch environment — all caches on project storage
export CAVE_PROJECT_DIR=/projects/cave-vlm-cot
export HF_HOME=${CAVE_PROJECT_DIR}/hf_cache
export HF_HUB_CACHE=${CAVE_PROJECT_DIR}/hf_cache 
export TRANSFORMERS_CACHE=${CAVE_PROJECT_DIR}/hf_cache
export UNSLOTH_COMPILED_CACHE=/projects/cave-vlm-cot/unsloth_compiled_cache
# export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1     # prevent HF network calls on compute node
export UNSLOTH_DISABLE_STATISTICS=1
export TQDM_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MMMU_IMAGE_ROOT=${CAVE_PROJECT_DIR}/processed_images_mmmu

# Verifier defaults for A40 runs. Override these in the sbatch environment
# if you want to compare another judge without editing the script.
export CAVE_VERIFIER_MODEL=${CAVE_VERIFIER_MODEL:-/projects/cave-vlm-cot/hf_cache/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da}
export CAVE_VERIFIER_QUANTIZATION=${CAVE_VERIFIER_QUANTIZATION:-4bit}
export CAVE_VERIFIER_DEVICE=${CAVE_VERIFIER_DEVICE:-cuda:2}
export CAVE_VERIFIER_DEVICE_MAP=${CAVE_VERIFIER_DEVICE_MAP:-single}
export CAVE_VERIFIER_MAX_PIXELS=${CAVE_VERIFIER_MAX_PIXELS:-768}
export CAVE_MAX_SAMPLES=${CAVE_MAX_SAMPLES:-1000}
export CAVE_SAMPLE_SEED=${CAVE_SAMPLE_SEED:-42}

# API keys
# set -a / source correctly handles values with special characters
# (dashes, equals signs, long tokens). The xargs approach breaks on those.
if [ -f "${WORKDIR}/.env" ]; then
    set -a
    source "${WORKDIR}/.env"
    set +a
fi

# Banner
echo "  CaVe-VLM-CoT experiment run"
echo "  Task ID      : ${TASK_ID}  (experiment ${EXPERIMENT_ID}, shard ${SHARD_ID})"
echo "  Experiment   : ${EXP_LABEL}"
echo "  Pipeline     : ${PIPELINE}"
echo "  Dataset      : ${DATASET}"
echo "  Shard        : ${SHARD_ID} / ${LAST_SHARD}"
echo "  Max samples  : ${CAVE_MAX_SAMPLES}"
echo "  Sample seed  : ${CAVE_SAMPLE_SEED}"
echo "  Verifier     : ${CAVE_VERIFIER_MODEL} (${CAVE_VERIFIER_QUANTIZATION}, ${CAVE_VERIFIER_DEVICE_MAP}, max_pixels=${CAVE_VERIFIER_MAX_PIXELS})"
echo "  Host         : $(hostname)"
echo "  Started      : $(date)"

# Disk quota pre-flight
HOME_USED=$(du -sb ~ 2>/dev/null | awk '{print $1}')
HOME_LIMIT=$((70 * 1024 * 1024 * 1024))  # adjust to your actual quota
if [ "${HOME_USED}" -gt "${HOME_LIMIT}" ]; then
    echo "ERROR: Home quota nearly full (${HOME_USED} bytes used). Aborting."
    exit 1
fi

# Python / GPU sanity check
echo ""

echo " PYTHON DIAGNOSTIC "
which python && python --version

python - <<'PYEOF' || true
import sys, torch
print("Executable:", sys.executable)

for name, mod in [
    ("transformers", "transformers"),
    ("torch",        "torch"),
    ("unsloth",      "unsloth"),
]:
    try:
        __import__(mod)
        print(f"{name} OK")
    except Exception as e:
        print(f"{name} FAIL: {e}")

print(f"\nGPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i}  {props.name}  {total/1e9:.0f} GB total  {free/1e9:.1f} GB free")
PYEOF

# Keep only the 2 most recent log sets per experiment label
find "${LOGDIR}" -name "cave_*_${EXP_LABEL}*" -type f \
    | sort -t_ -k3 -rn \
    | tail -n +11 \
    | xargs rm -f 2>/dev/null || true

# Main experiment
#
# --shard is passed explicitly because SLURM_ARRAY_TASK_ID is now 0–59
# (the flat task index), not 0–19 (the shard index). Without this flag,
# experiments.py would read SLURM_ARRAY_TASK_ID=47 and try shard 47 of 20
# and crash.
#
# --dataset is now driven by the DATASETS array above, not an env-var override.
# This ensures each experiment group uses the correct dataset automatically
# without requiring separate sbatch submissions.
#
echo ""
echo "Starting: ${EXP_LABEL}, shard ${SHARD_ID} of ${LAST_SHARD} ..."

# Wait for GPUs to become available after SLURM allocation
sleep 10
python -c "import torch; torch.cuda.init(); print(f'GPUs ready: {torch.cuda.device_count()}')" || sleep 30

sleep $(( (SLURM_ARRAY_TASK_ID % 5) * 120 ))
python -u experiments.py              \
    --num-shards    "${NUM_SHARDS}"   \
    --shard         "${SHARD_ID}"     \
    --pipeline      "${PIPELINE}"     \
    --dataset       "${DATASET}"      \
    --max-samples   "${CAVE_MAX_SAMPLES}" \
    --sample-seed   "${CAVE_SAMPLE_SEED}" \
    --verifier-model "${CAVE_VERIFIER_MODEL}" \
    --verifier-quantization "${CAVE_VERIFIER_QUANTIZATION}" \
    --verifier-device "${CAVE_VERIFIER_DEVICE}" \
    --verifier-device-map "${CAVE_VERIFIER_DEVICE_MAP}" \
    --verifier-max-pixels "${CAVE_VERIFIER_MAX_PIXELS}" \
    --no-traces                       \
    2>"${LOGDIR}/cave_${SLURM_JOB_ID}_${TASK_ID}.err" \
    >  "${LOGDIR}/cave_${SLURM_JOB_ID}_${TASK_ID}.out"

EXIT_CODE=$?
echo ""
echo "Shard ${SHARD_ID} (task ${TASK_ID}) finished at $(date) with exit code ${EXIT_CODE}"
echo "  Task ${TASK_ID} (${EXP_LABEL}, shard ${SHARD_ID}) done — exit ${EXIT_CODE}"
exit "${EXIT_CODE}"

# 1k ScienceQA solver-only
# sbatch --array=0-39 --gres=gpu:a40:3 -p a40_b3 cave_array.sh

# 1k ScienceQA retrieval-solver
# sbatch --array=40-79 --gres=gpu:a40:3 -p a40_b3 cave_array.sh

# 1k ScienceQA full CaVe-VLM-CoT
# sbatch --array=80-119 --gres=gpu:a40:3 -p a40_b3 cave_array.sh