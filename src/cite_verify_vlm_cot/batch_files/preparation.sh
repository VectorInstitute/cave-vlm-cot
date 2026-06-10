#!/bin/bash
#SBATCH --job-name=cave-data-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1          # 1 GPU for BLIP captioning
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00         # captioning 21k images can take several hours
#SBATCH --chdir=${CAVE_WORKDIR}/src/cite_verify_vlm_cot
#SBATCH --output=/projects/cave-vlm-cot/logs/data_prep_%j.out
#SBATCH --error=/projects/cave-vlm-cot/logs/data_prep_%j.err

set -euo pipefail

mkdir -p /projects/cave-vlm-cot/logs

# Modules
module purge
module load StdEnv/2020
module load tesseract/5.0.1

# Now load the rest under StdEnv/2023
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load arrow
module load python/3.11
module load scipy-stack

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/MPI/gcc12-openmpi4/scipy-stack/2023b/lib/python3.11/site-packages:$PYTHONPATH

source "${CAVE_WORKDIR}/env/bin/activate"

export CAVE_PROJECT_DIR=/projects/cave-vlm-cot
export HF_HOME=${CAVE_PROJECT_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAVE_PROJECT_DIR}/hf_cache
export TQDM_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCIENCEQA_ROOT=${SCIENCEQA_ROOT}

echo "Started: $(date)"
echo "Host: $(hostname)"

python data-preparation.py

echo "Finished: $(date)"
