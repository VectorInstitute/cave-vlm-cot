#!/bin/bash

# CaVe-VLM-CoT — parallel 3-shard SLURM array job
#
# Submits three independent jobs, each owning 3 A100 GPUs and processing
# one third of scienceqa_augmented.csv.  Wall-clock time drops from ~6 days
# to ~2 days (3× speedup, limited only by longest shard).
#
# Submit:
#   sbatch cave_array.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/cave_shard_0.out

#SBATCH --job-name=cave-vlm-cot
#SBATCH --array=0-2                   # 3 tasks → SLURM_ARRAY_TASK_ID = 0,1,2
#SBATCH --nodes=1                     # each task runs on its own node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12            # DDG web search uses ThreadPoolExecutor
#SBATCH --gres=gpu:a100:3                  # 3 A100s per task (planner/solver/verifier)
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00             # 3 days — comfortably covers 1 shard
#SBATCH --chdir=/h/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
#SBATCH --output=/h/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/cave_shard_%j_%a.out
#SBATCH --error=/h/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/cave_shard_%j_%a.err

# Environment
mkdir -p /h/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs

module purge
module load cuda/12.4   # CUDA 12.4 matches the toolkit
# Python comes from the virtualenv below — no separate python module needed

# source /fs01/home/sneharao/cbm-vlms/fresh_env/bin/activate
source /h/sneharao/cbm-vlms/fresh_env/bin/activate

export HF_HOME=/fs01/home/sneharao/hf_cache
export TRANSFORMERS_CACHE=/fs01/home/sneharao/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Quick sanity checks ────────────────────────────────────────────────────────
echo "==== PYTHON DIAGNOSTIC ===="
which python && python --version

python - <<'EOF'
import sys, torch
print("Executable:", sys.executable)

for name, mod in [("transformers", "transformers"), ("torch", "torch"), ("unsloth", "unsloth")]:
    try:
        __import__(mod)
        print(f"{name} OK")
    except Exception as e:
        print(f"{name} FAIL: {e}")

print(f"\nGPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i}  {props.name}  {total/1e9:.0f}GB total  {free/1e9:.1f}GB free")
EOF

# Load API keys from .env (dotenv is called inside the script too, but having
# them in the shell environment is a safe fallback)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run
# SLURM_ARRAY_TASK_ID (0, 1, or 2) is picked up automatically by experiments.py
# via os.environ["SLURM_ARRAY_TASK_ID"] — no --shard flag needed.
# Pass --num-shards explicitly so the script knows the total shard count.
echo "Starting shard ${SLURM_ARRAY_TASK_ID} of 3 on $(hostname) at $(date)"

python experiments.py --num-shards 3

EXIT_CODE=$?
echo "Shard ${SLURM_ARRAY_TASK_ID} finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
