#!/bin/bash
# cave_postprocess.sh — Aggregate shards + weight sensitivity analysis
#
# Runs AFTER all cave_array.sh tasks complete.  Submit with a dependency:
#
#   ARRAY_JOB_ID=$(sbatch --parsable cave_array.sh)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} cave_postprocess.sh
#
# Or run both in one line:
#   sbatch --dependency=afterok:$(sbatch --parsable cave_array.sh) cave_postprocess.sh
#
# What it does:
#   1. For each experiment label, locates the Phoenix-exported per-shard CSVs
#      and merges them with aggregate_shards.py into a single JSON file.
#   2. Runs sensitivity_analysis.py (Experiment 4) on each merged result file.
#
# This is a lightweight CPU-only job — no GPUs needed.

#SBATCH --job-name=cave-postprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
#SBATCH --output=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/postprocess_%j.out
#SBATCH --error=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot/logs/postprocess_%j.err

set -euo pipefail

# Paths (must match cave_array.sh)
WORKDIR=/fs02/home/sneharao/cave-vlm-cot/src/cite_verify_vlm_cot
LOGDIR=${WORKDIR}/logs
OUTDIR=${WORKDIR}/outputs
mkdir -p "${LOGDIR}" "${OUTDIR}"

# Modules + environment (mirror cave_array.sh)
module purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load scipy-stack
module load arrow

source /fs02/home/sneharao/cave-vlm-cot/env/bin/activate

if [ -f "${WORKDIR}/.env" ]; then
    set -a
    source "${WORKDIR}/.env"
    set +a
fi

echo "  CaVe-VLM-CoT Post-Processing"
echo "  Host    : $(hostname)"
echo "  Started : $(date)"

# Experiment labels (must match cave_array.sh)
EXPERIMENT_LABELS=(
    "baseline-full"
    "ablation1-retrieval-solver"
    "ablation2-solver-only"
    "ablation3-no-citation-injector"
)

NUM_SHARDS=5
OVERALL_EXIT=0

for EXP_LABEL in "${EXPERIMENT_LABELS[@]}"; do
    echo ""
    echo "  Experiment: ${EXP_LABEL}"

    # Step 1: Aggregate shards
    # Look for per-shard Phoenix CSV exports matching this experiment label.
    # Typical naming convention: cave_<exp_label>_shard<N>.csv
    # Adjust the glob pattern below to match your actual export filenames.
    SHARD_CSVS=( ${OUTDIR}/*${EXP_LABEL}*shard*.csv )

    if [ ${#SHARD_CSVS[@]} -eq 0 ] || [ ! -f "${SHARD_CSVS[0]}" ]; then
        echo "  WARNING: No per-shard CSVs found matching *${EXP_LABEL}*shard*.csv"
        echo "           Export CSVs from Phoenix first, then re-run this script."
        echo "           Skipping ${EXP_LABEL}."
        OVERALL_EXIT=1
        continue
    fi

    echo "  Found ${#SHARD_CSVS[@]} shard CSV(s):"
    for f in "${SHARD_CSVS[@]}"; do
        echo "    • $(basename "$f")"
    done

    COMBINED_CSV="${OUTDIR}/cave_results_${EXP_LABEL}.csv"
    COMBINED_JSON="${OUTDIR}/cave_results_${EXP_LABEL}.json"

    echo ""
    echo "  Running aggregate_shards.py ..."
    python aggregate_shards.py "${SHARD_CSVS[@]}" \
        --save-csv "${COMBINED_CSV}"

    if [ $? -ne 0 ]; then
        echo "  ERROR: aggregate_shards.py failed for ${EXP_LABEL}"
        OVERALL_EXIT=1
        continue
    fi
    echo "  Aggregated → ${COMBINED_CSV}"

    # Step 2: Weight sensitivity analysis
    echo ""
    echo "  Running sensitivity_analysis.py ..."
    python sensitivity_analysis.py \
        --results-file "${COMBINED_CSV}" \
        --format csv \
        --out-dir "${OUTDIR}"

    if [ $? -ne 0 ]; then
        echo "  ERROR: sensitivity_analysis.py failed for ${EXP_LABEL}"
        OVERALL_EXIT=1
        continue
    fi

    # Namespace outputs so experiments don't overwrite each other
    mv "${OUTDIR}/weight_sensitivity.json" \
       "${OUTDIR}/weight_sensitivity_${EXP_LABEL}.json"
    mv "${OUTDIR}/weight_sensitivity.csv" \
       "${OUTDIR}/weight_sensitivity_${EXP_LABEL}.csv"

    echo "  Sensitivity analysis complete:"
    echo "    → ${OUTDIR}/weight_sensitivity_${EXP_LABEL}.json"
    echo "    → ${OUTDIR}/weight_sensitivity_${EXP_LABEL}.csv"
done

echo ""
echo "  Post-processing finished at $(date)"
echo "  Exit code: ${OVERALL_EXIT}"

exit "${OVERALL_EXIT}"