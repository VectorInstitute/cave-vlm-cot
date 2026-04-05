#!/bin/bash
# cave_array.slurm — CaVe-VLM-CoT: all ablations, all shards, one submission
#
# Launches 5 experiments × 5 shards = 25 SLURM array tasks.
# Each task owns 4 RTX-6000 GPUs and processes 1/5 of scienceqa_augmented.csv.
#
#   Task ID │ EXPERIMENT_ID │ SHARD │ --pipeline                │ --model-variant
#   ────────┼───────────────┼───────┼───────────────────────────┼────────────────
#     0– 4  │      0        │  0–4  │ full                      │ qwen25  (baseline)
#     5– 9  │      1        │  0–4  │ retrieval-solver          │ qwen25  (ablation 1)
#    10–14  │      2        │  0–4  │ solver-only               │ qwen25  (ablation 2)
#    15–19  │      3        │  0–4  │ no-citation-injector      │ qwen25  (ablation 3)
#    20–24  │      4        │  0–4  │ full                      │ qwen3   (ablation 5)
#
# Experiment 4 (weight sensitivity) is a CPU post-processing step.
# It runs automatically on the last shard of every experiment group
# (task IDs 4, 9, 14, 19, 24) once the pipeline job exits cleanly.
#
# Submit everything:
#   sbatch cave_array.slurm
#
# Submit only specific tasks (e.g. baseline + qwen3):
#   sbatch --array=0-4,20-24 cave_array.slurm
#
# Resume a failed shard (e.g. task 7):
#   sbatch --array=7 cave_array.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/cave_<jobid>_<taskid>.out

#SBATCH --job-name=cave-vlm-cot
#SBATCH --array=0-24                  # 25 tasks: 5 experiments × 5 shards
#SBATCH --nodes=1                     # each task runs on its own node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12            # DDG web search uses ThreadPoolExecutor
#SBATCH --gres=gpu:rtx6000:4          # 4 GPUs per task (planner/solver/verifier + spare)
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00             # 3 days — comfortably covers 1 shard
#SBATCH --chdir=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
#SBATCH --output=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/cave_%j_%a.out
#SBATCH --error=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/cave_%j_%a.err

set -euo pipefail
cleanup() {
    echo "Compressing logs..."
    gzip -f "${LOGDIR}/cave_${SLURM_JOB_ID}_${SHARD_ID}.out" 2>/dev/null || true
    gzip -f "${LOGDIR}/cave_${SLURM_JOB_ID}_${SHARD_ID}.err" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p /fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs

# Paths
WORKDIR=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
LOGDIR=${WORKDIR}/logs
OUTDIR=${WORKDIR}/outputs
mkdir -p "${LOGDIR}" "${OUTDIR}"

# Derive experiment config and shard from flat task ID
#   SLURM_ARRAY_TASK_ID: 0–24 (flat)
#   EXPERIMENT_ID:       0–4  (which experiment group)
#   SHARD_ID:            0–4  (which fifth of the dataset)
TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=5
EXPERIMENT_ID=$(( TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
LAST_SHARD=$(( NUM_SHARDS - 1 ))   # = 4

# Experiment table (indexed by EXPERIMENT_ID)
PIPELINES=(
    "full"                    # 0  baseline
    "retrieval-solver"        # 1  ablation 1: no citation injector / verifier
    "solver-only"             # 2  ablation 2: no retrieval at all
    "no-citation-injector"    # 3  ablation 3: full pipeline minus citation_injector
    "full"                    # 4  ablation 4: Qwen3 extractor + verifier
)

MODEL_VARIANTS=(
    "qwen25"   # 0
    "qwen25"   # 1
    "qwen25"   # 2
    "qwen25"   # 3
    "qwen3"    # 4
)

EXPERIMENT_LABELS=(
    "baseline-full-qwen25"           # 0
    "ablation1-retrieval-solver"     # 1
    "ablation2-solver-only"          # 2
    "ablation3-no-citation-injector" # 3
    "ablation5-full-qwen3"           # 4
)

PIPELINE="${PIPELINES[$EXPERIMENT_ID]}"
MODEL_VARIANT="${MODEL_VARIANTS[$EXPERIMENT_ID]}"
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
export PYTHONPATH=/cvmfs/.../site-packages:$PYTHONPATH   # ← path from above command
# Python comes from the virtualenv below — no separate python module needed

source /fs02/home/sneharao/cave-vlm-cot/env/bin/activate

# Huggingface / PyTorch environment
export HF_HOME=/fs02/home/sneharao/hf_cache       
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1
export TRANSFORMERS_CACHE=/fs02/home/sneharao/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "  Model variant: ${MODEL_VARIANT}"
echo "  Shard        : ${SHARD_ID} / $((NUM_SHARDS - 1))"
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

python - <<'PYEOF'
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
# --shard is passed explicitly because SLURM_ARRAY_TASK_ID is now 0–24
# (the flat task index), not 0–4 (the shard index).  Without this flag,
# experiments.py would read SLURM_ARRAY_TASK_ID=7 and try shard 7 of 5
# and crash.
echo ""
echo "Starting: ${EXP_LABEL}, shard ${SHARD_ID} of $((NUM_SHARDS - 1)) ..."

python -u experiments.py          \
    --num-shards    "${NUM_SHARDS}"      \
    --shard         "${SHARD_ID}"        \
    --pipeline      "${PIPELINE}"        \
    --model-variant "${MODEL_VARIANT}"   \
    2>"${LOGDIR}/cave_${SLURM_JOB_ID}_${SHARD_ID}.err" \
    | grep -Ev "it/s|B/s|%\||\[.*\].*ETA|Downloading|Loading" \
    > "${LOGDIR}/cave_${SLURM_JOB_ID}_${SHARD_ID}.out"

EXIT_CODE=$?
echo ""
echo "Shard ${SHARD_ID} (task ${TASK_ID}) finished at $(date) with exit code ${EXIT_CODE}"

# Weight sensitivity analysis (Experiment 4)
# Runs only on the LAST shard of each experiment group (SHARD_ID == 4) so it
# executes exactly once per experiment, with no separate sbatch dependency.
# Expects aggregate_shards.py to have merged the 5 per-shard outputs into:
#   outputs/cave_results_<EXP_LABEL>.json
# If you run aggregate_shards.py in a separate downstream job, remove this
# block and call sensitivity_analysis.py manually after aggregation.
if [ "${SHARD_ID}" -eq "${LAST_SHARD}" ] && [ "${EXIT_CODE}" -eq 0 ]; then
    RESULTS_FILE="${OUTDIR}/cave_results_${EXP_LABEL}.json"

    echo ""
    echo "  Experiment 4: weight sensitivity analysis"
    echo "  Experiment   : ${EXP_LABEL}"
    echo "  Looking for  : ${RESULTS_FILE}"

    if [ -f "${RESULTS_FILE}" ]; then
        python sensitivity_analysis.py     \
            --results-file "${RESULTS_FILE}"   \
            --out-dir      "${OUTDIR}"

        # Namespace outputs so parallel experiments do not overwrite each other
        mv "${OUTDIR}/weight_sensitivity.json" \
           "${OUTDIR}/weight_sensitivity_${EXP_LABEL}.json"
        mv "${OUTDIR}/weight_sensitivity.csv"  \
           "${OUTDIR}/weight_sensitivity_${EXP_LABEL}.csv"

        echo "  Sensitivity analysis complete."
        echo "  → ${OUTDIR}/weight_sensitivity_${EXP_LABEL}.json"
        echo "  → ${OUTDIR}/weight_sensitivity_${EXP_LABEL}.csv"
    else
        echo "  NOTE: ${RESULTS_FILE} not found — skipping sensitivity analysis."
        echo "  Run aggregate_shards.py first, then:"
        echo "    python sensitivity_analysis.py --results-file ${RESULTS_FILE}"
    fi
fi

echo ""
echo "  Task ${TASK_ID} (${EXP_LABEL}, shard ${SHARD_ID}) done — exit ${EXIT_CODE}"

exit "${EXIT_CODE}"
