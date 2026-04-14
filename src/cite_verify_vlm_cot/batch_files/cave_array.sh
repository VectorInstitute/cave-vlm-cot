#!/bin/bash
# cave_array.slurm — CaVe-VLM-CoT: all ablations, all shards, one submission
#
# Launches 3 experiments × 50 shards = 150 SLURM array tasks.
#
#   Task ID  │ EXPERIMENT_ID │ SHARD  │ --pipeline             │ --dataset  │ Notes
#   ─────────┼───────────────┼────────┼────────────────────────┼────────────┼──────────────────
#     0– 49  │      0        │  0–49  │ full                   │ scienceqa  │ baseline
#    50– 99  │      1        │  0–49  │ no-citation-injector   │ scienceqa  │ ablation 3
#    100–149 │      2        │  0–49  │ full                   │ mmmu       │ generalization
#
# Experiment 4 (weight sensitivity) runs as a separate downstream job
# via cave_postprocess.sh after all array tasks complete:
#   ARRAY_JOB_ID=$(sbatch --parsable cave_array.sh)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} cave_postprocess.sh
#
# Submit everything:
#   sbatch cave_array.sh
#
# Submit a single experiment group (e.g. baseline only):
#   sbatch --array=0-49   cave_array.sh
#
# Resume a single failed task (e.g. task 47):
#   sbatch --array=47 cave_array.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /projects/cave-vlm-cot/logs/cave_<jobid>_<taskid>.out

#SBATCH --job-name=cave-vlm-cot
#SBATCH --array=0-149                  # 150 tasks: 3 experiments × 50 shards
#SBATCH --nodes=1                     # each task runs on its own node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12            # DDG web search uses ThreadPoolExecutor
#SBATCH --gres=gpu:a40:3          # 3 GPUs per task (planner/solver/verifier)
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00             # each shard is 1/50 of dataset, comfortably < 1 day
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
#   SLURM_ARRAY_TASK_ID: 0–149 (flat index across all 150 tasks)
#   EXPERIMENT_ID:       0–2   (which experiment group)
#   SHARD_ID:            0–49  (which fiftieth of the dataset)
#
TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=50
EXPERIMENT_ID=$(( TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
LAST_SHARD=$(( NUM_SHARDS - 1 ))   # = 49

# Experiment tables (indexed by EXPERIMENT_ID)
#
# PIPELINES and DATASETS must stay in sync — both indexed by EXPERIMENT_ID.
# The --dataset flag selects which augmented CSV (and which FAISS index
# sub-directory) experiments.py uses; it is not an env-var override any more.
#
PIPELINES=(
    "full"                    # 0  full pipeline on ScienceQA (baseline)
    "no-citation-injector"    # 1  ablation: full minus citation injector, ScienceQA
    "full"                    # 2  full pipeline on MMMU (generalization)
)

DATASETS=(
    "scienceqa"               # 0
    "scienceqa"               # 1
    "mmmu"                    # 2
)

EXPERIMENT_LABELS=(
    "baseline-full-scienceqa"           # 0
    "ablation-no-citation-injector"     # 1
    "generalization-full-mmmu"          # 2
)

PIPELINE="${PIPELINES[$EXPERIMENT_ID]}"
DATASET="${DATASETS[$EXPERIMENT_ID]}"
EXP_LABEL="${EXPERIMENT_LABELS[$EXPERIMENT_ID]}"

# Shard calculation — MMMU is small, use fewer shards
if [ "${DATASET}" = "mmmu" ]; then
    NUM_SHARDS=5
    SHARD_ID=$(( TASK_ID - 100 ))
else
    NUM_SHARDS=50
    SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
fi
LAST_SHARD=$(( NUM_SHARDS - 1 ))

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
export PYTHONPATH=/cvmfs/.../site-packages:$PYTHONPATH   # path from above command
# Python comes from the virtualenv below — no separate python module needed

source /fs02/home/sneharao/cave-vlm-cot/env/bin/activate

# Huggingface / PyTorch environment — all caches on project storage
export CAVE_PROJECT_DIR=/projects/cave-vlm-cot
export HF_HOME=${CAVE_PROJECT_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAVE_PROJECT_DIR}/hf_cache
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MMMU_IMAGE_ROOT=${CAVE_PROJECT_DIR}/processed_images_mmmu

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
# --shard is passed explicitly because SLURM_ARRAY_TASK_ID is now 0–149
# (the flat task index), not 0–49 (the shard index). Without this flag,
# experiments.py would read SLURM_ARRAY_TASK_ID=57 and try shard 57 of 50
# and crash
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

python -u experiments.py              \
    --num-shards    "${NUM_SHARDS}"   \
    --shard         "${SHARD_ID}"     \
    --pipeline      "${PIPELINE}"     \
    --dataset       "${DATASET}"      \
    --no-traces                       \
    2>"${LOGDIR}/cave_${SLURM_JOB_ID}_${TASK_ID}.err" \
    >  "${LOGDIR}/cave_${SLURM_JOB_ID}_${TASK_ID}.out"

EXIT_CODE=$?
echo ""
echo "Shard ${SHARD_ID} (task ${TASK_ID}) finished at $(date) with exit code ${EXIT_CODE}"
echo "  Task ${TASK_ID} (${EXP_LABEL}, shard ${SHARD_ID}) done — exit ${EXIT_CODE}"
exit "${EXIT_CODE}"

# ScienceQA full pipeline
# sbatch --array=0-49 --gres=gpu:a40:3 -p a40_b3 --exclude=bn062,bn063,bn064,bn065,bn067,bn068,bn070,bn072,bn073 cave_array.sh

# ScienceQA no-citation-injector
# sbatch --array=50-99 --gres=gpu:a40:3 -p a40_b3 --exclude=bn062,bn063,bn064,bn065,bn067,bn068,bn070,bn072,bn073 cave_array.sh

# MMMU generalization
# sbatch --array=100-104 --gres=gpu:a40:3 -p a40_b3 --exclude=bn062,bn063,bn064,bn065,bn067,bn068,bn070,bn072,bn073 cave_array.sh