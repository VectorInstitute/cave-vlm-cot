#!/bin/bash

# CaVe-VLM-CoT — parallel 3-shard SLURM array job
#
# Submits 5 independent jobs, each owning 4 A100 GPUs and processing
# one fifth of scienceqa_augmented.csv.  Wall-clock time drops from ~6 days
# to ~2 days (3× speedup, limited only by longest shard).
#
# Submit:
#   sbatch cave_array.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/cave_shard_<jobid>_<arrayid>.out

#SBATCH --job-name=cave-vlm-cot
#SBATCH --array=0-4                   # 5 tasks → SLURM_ARRAY_TASK_ID = 0,1,2,3,4
#SBATCH --nodes=1                     # each task runs on its own node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12            # DDG web search uses ThreadPoolExecutor
#SBATCH --gres=gpu:rtx6000:4          # 4 GPUs per task (planner/solver/verifier + spare)
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00             # 3 days — comfortably covers 1 shard
#SBATCH --chdir=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
#SBATCH --output=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/cave_shard_%j_%a.out
#SBATCH --error=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/cave_shard_%j_%a.err

# Environment
# Create log dir before SLURM tries to open the output/error files.
# Must use absolute path — ~ not yet expanded at this point in some shells.
mkdir -p /fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs

# StdEnv must load before cuda — it sets up the toolchain cuda depends on.
module purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2  # CUDA 12.2 matches the toolkit
module load arrow
module load faiss/1.8.0
# Python comes from the virtualenv below — no separate python module needed

which python
python --version

source /h/sneharao/cave-vlm-cot/env/bin/activate
# pip install -r /h/sneharao/cave-vlm-cot/requirements.txt

export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HOME/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load API keys — set -a/source correctly handles values with special
# characters (dashes, equals signs, long tokens). The xargs approach breaks.
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Quick sanity checks 
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

# SLURM_ARRAY_TASK_ID (0-4) is read automatically by experiments.py via
# os.environ["SLURM_ARRAY_TASK_ID"] — no --shard flag needed here.
echo "Starting shard ${SLURM_ARRAY_TASK_ID} of 5 on $(hostname) at $(date)"

python experiments.py --num-shards 5

EXIT_CODE=$?
echo "Shard ${SLURM_ARRAY_TASK_ID} finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
