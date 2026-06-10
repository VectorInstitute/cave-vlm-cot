#!/bin/bash
# cave_mmmu_array.slurm — CaVe-VLM-CoT: MMMU full-pipeline run
#
# Launches 5 shards over the full MMMU augmented dataset by default.
#
#   Task ID  │ SHARD  │ --pipeline  │ --dataset
#   ─────────┼────────┼─────────────┼──────────
#     0– 4   │  0–4   │ full        │ mmmu
#
# Submit:
#   sbatch cave_mmmu_array.sh
#
# Smoke test:
#   sbatch --array=0 cave_mmmu_array.sh

#SBATCH --job-name=cave-mmmu
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:3
#SBATCH --mem=120G
#SBATCH --time=0-12:00:00
#SBATCH --chdir=${CAVE_WORKDIR}/src/cite_verify_vlm_cot
#SBATCH --output=/projects/cave-vlm-cot/logs/cave_mmmu_%j_%a.out
#SBATCH --error=/projects/cave-vlm-cot/logs/cave_mmmu_%j_%a.err

set -euo pipefail

PROJECT_DIR=/projects/cave-vlm-cot
WORKDIR=${CAVE_WORKDIR}/src/cite_verify_vlm_cot
LOGDIR=${PROJECT_DIR}/logs
mkdir -p "${LOGDIR}" "${PROJECT_DIR}/hf_cache" "${PROJECT_DIR}/indexes"

TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=${CAVE_MMMU_NUM_SHARDS:-20}
SHARD_ID=${TASK_ID}
LAST_SHARD=$(( NUM_SHARDS - 1 ))

module purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load arrow
module load python/3.11
module load scipy-stack
module load faiss/1.12.0
export PYTHONPATH="${PROJECT_DIR}/env/lib/python3.11/site-packages:$PYTHONPATH"

source "${CAVE_WORKDIR}/env/bin/activate"

export CAVE_PROJECT_DIR=/projects/cave-vlm-cot
export HF_HOME=${CAVE_PROJECT_DIR}/hf_cache
export HF_HUB_CACHE=${CAVE_PROJECT_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAVE_PROJECT_DIR}/hf_cache
export UNSLOTH_COMPILED_CACHE=/projects/cave-vlm-cot/unsloth_compiled_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export UNSLOTH_DISABLE_STATISTICS=1
export TQDM_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MMMU_IMAGE_ROOT=${CAVE_PROJECT_DIR}/processed_images_mmmu

# Full MMMU by default. Set CAVE_MAX_SAMPLES for a smaller diagnostic subset.
export CAVE_MAX_SAMPLES=${CAVE_MAX_SAMPLES:-500}
export CAVE_SAMPLE_SEED=${CAVE_SAMPLE_SEED:-42}
export CAVE_VERIFIER_MODEL=${CAVE_VERIFIER_MODEL:-/projects/cave-vlm-cot/hf_cache/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da}
export CAVE_VERIFIER_QUANTIZATION=${CAVE_VERIFIER_QUANTIZATION:-4bit}
export CAVE_VERIFIER_DEVICE=${CAVE_VERIFIER_DEVICE:-cuda:2}
export CAVE_VERIFIER_DEVICE_MAP=${CAVE_VERIFIER_DEVICE_MAP:-single}
export CAVE_VERIFIER_MAX_PIXELS=${CAVE_VERIFIER_MAX_PIXELS:-768}

if [ -f "${WORKDIR}/.env" ]; then
    set -a
    source "${WORKDIR}/.env"
    set +a
fi

echo "  CaVe-VLM-CoT MMMU run"
echo "  Task ID      : ${TASK_ID}"
echo "  Pipeline     : full"
echo "  Dataset      : mmmu"
echo "  Shard        : ${SHARD_ID} / ${LAST_SHARD}"
echo "  Max samples  : ${CAVE_MAX_SAMPLES}"
echo "  Sample seed  : ${CAVE_SAMPLE_SEED}"
echo "  Verifier     : ${CAVE_VERIFIER_MODEL} (${CAVE_VERIFIER_QUANTIZATION}, ${CAVE_VERIFIER_DEVICE_MAP}, max_pixels=${CAVE_VERIFIER_MAX_PIXELS})"
echo "  Host         : $(hostname)"
echo "  Started      : $(date)"

python - <<'PYEOF' || true
import sys, torch
print("Executable:", sys.executable)
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i}  {props.name}  {total/1e9:.0f} GB total  {free/1e9:.1f} GB free")
PYEOF

sleep 10
python -c "import torch; torch.cuda.init(); print(f'GPUs ready: {torch.cuda.device_count()}')" || sleep 30
sleep $(( (SLURM_ARRAY_TASK_ID % 5) * 120 ))

python -u experiments.py              \
    --num-shards    "${NUM_SHARDS}"   \
    --shard         "${SHARD_ID}"     \
    --pipeline      full              \
    --dataset       mmmu              \
    --max-samples   "${CAVE_MAX_SAMPLES}" \
    --sample-seed   "${CAVE_SAMPLE_SEED}" \
    --verifier-model "${CAVE_VERIFIER_MODEL}" \
    --verifier-quantization "${CAVE_VERIFIER_QUANTIZATION}" \
    --verifier-device "${CAVE_VERIFIER_DEVICE}" \
    --verifier-device-map "${CAVE_VERIFIER_DEVICE_MAP}" \
    --verifier-max-pixels "${CAVE_VERIFIER_MAX_PIXELS}" \
    --no-traces

EXIT_CODE=$?
echo "MMMU shard ${SHARD_ID} finished at $(date) with exit code ${EXIT_CODE}"
exit "${EXIT_CODE}"
