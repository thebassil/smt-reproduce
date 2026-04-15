#!/bin/bash
#SBATCH --job-name=ablation-gnn
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=08:00:00
#SBATCH --output=logs/gnn_%j.out
#SBATCH --error=logs/gnn_%j.err
#SBATCH --exclude=gecko-01,gecko-02
#
# Single REF-GNN ablation run on SCRTP Avon (GPU).
# For Blythe, override: --gres=gpu:lovelace_l40:1 --cpus-per-task=10 --mem-per-cpu=5960
#
# Usage:
#   sbatch run_gnn_single.sbatch                             # defaults
#   sbatch run_gnn_single.sbatch QF_BV 123                  # override logic + seed
#   sbatch run_gnn_single.sbatch QF_LIA 42 5                # + experiment_id for DB write

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

LOGIC="${1:-QF_LIA}"
SEED="${2:-42}"
EXPERIMENT_ID="${3:-}"

# --- Environment ---
REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
# DCS gecko: CUDA available natively, no module load needed
cd "${REPO_DIR}"
mkdir -p logs

echo "=== Ablation Runner: REF-GNN ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Logic:     ${LOGIC}"
echo "Seed:      ${SEED}"
echo "Started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Python:    $(python3 --version)"
echo "CUDA:      $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Build Hydra overrides ---
OVERRIDES=(
    "reference=ref_gnn"
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
