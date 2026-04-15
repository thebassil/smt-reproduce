#!/bin/bash
#SBATCH --job-name=ablation-vector
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=04:00:00
#SBATCH --output=logs/vector_%j.out
#SBATCH --error=logs/vector_%j.err
#
# Single REF-VECTOR ablation run on SCRTP taskfarm (CPU).
#
# Usage:
#   sbatch run_vector_single.sbatch                          # defaults
#   sbatch run_vector_single.sbatch QF_BV 123               # override logic + seed
#   sbatch run_vector_single.sbatch QF_LIA 42 5             # + experiment_id for DB write

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

LOGIC="${1:-QF_LIA}"
SEED="${2:-42}"
EXPERIMENT_ID="${3:-}"

# --- Environment ---
REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
# DCS: use system Python (no venv needed)
cd "${REPO_DIR}"
mkdir -p logs

echo "=== Ablation Runner: REF-VECTOR ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Logic:     ${LOGIC}"
echo "Seed:      ${SEED}"
echo "Started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Python:    $(python3 --version)"

# --- Build Hydra overrides ---
OVERRIDES=(
    "reference=ref_vector"
    "data.logic=${LOGIC}"
    "data.db_path=${REPO_DIR}/db/results.sqlite"
    "seed=${SEED}"
)

if [ -n "${EXPERIMENT_ID}" ]; then
    OVERRIDES+=("+experiment_id=${EXPERIMENT_ID}")
    echo "Experiment: ${EXPERIMENT_ID} (DB write enabled)"
else
    echo "Experiment: none (dry run, no DB write)"
fi

# --- Run ---
python3 -m pipeline.runner "${OVERRIDES[@]}"

echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
