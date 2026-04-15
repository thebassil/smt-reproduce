#!/bin/bash
#SBATCH --job-name=sweep-vector
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=12:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/sweep_vector_%A_%a.out
#SBATCH --error=logs/sweep_vector_%A_%a.err
#
# Array job: sweep VECTOR model cards across seeds and logics.
# Each array task runs one (model, seed, logic) combination.
#
# The sweep grid is defined in MODELS/SEEDS/LOGICS below.
# Total tasks = len(MODELS) * len(SEEDS) * len(LOGICS).
# Adjust --array=0-N to match.
#
# Usage:
#   sbatch sweep_vector.sbatch
#   sbatch sweep_vector.sbatch 5              # experiment_id for DB write

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

EXPERIMENT_ID_CLI="${1:-}"

# --- Sweep grid ---
MODELS=(
    "adaboost_ehm"
    "knn"
    "rf"
    "xgboost"
    "lightgbm"
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

# --- Compute (model, seed, logic) from SLURM_ARRAY_TASK_ID ---
N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
N_LOGICS=${#LOGICS[@]}
TOTAL=$((N_MODELS * N_SEEDS * N_LOGICS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "TASK_ID ${TASK_ID} >= TOTAL ${TOTAL}, exiting."
    exit 0
fi

MODEL_IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
LOGIC="${LOGICS[$LOGIC_IDX]}"

# --- Environment ---
REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
# DCS: use system Python (no venv needed)
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== Sweep: VECTOR model=${MODEL} seed=${SEED} logic=${LOGIC} ==="
export CARD_MODEL="${MODEL}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Grid: ${N_MODELS} models x ${N_SEEDS} seeds x ${N_LOGICS} logics = ${TOTAL} tasks"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Resolve experiment_id ---
EXPERIMENT_ID="${EXPERIMENT_ID_CLI}"
if [ -z "${EXPERIMENT_ID}" ]; then
    SWEEP_MAP="${REPO_DIR}/sweep_experiments.json"
    if [ -f "${SWEEP_MAP}" ]; then
        KEY="${MODEL}|${LOGIC}"
        EXPERIMENT_ID=$(python3 -c "import json,sys; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
    fi
fi

# --- Build Hydra overrides ---
OVERRIDES=(
    "reference=ref_vector"
    "model@reference.model=${MODEL}"
    "data.logic=${LOGIC}"
    "data.db_path=/dcs/large/u5573765/db/results.sqlite"
    "data.shard_db=${REPO_DIR:-/dcs/23/u5573765/cs351/smt-ablation-cross-system}/db/shards/shard_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.sqlite"
    "seed=${SEED}"
)

if [ -n "${EXPERIMENT_ID}" ]; then
    OVERRIDES+=("+experiment_id=${EXPERIMENT_ID}")
    echo "Experiment ID: ${EXPERIMENT_ID}"
fi

# --- Run ---
python3 -m pipeline.runner "${OVERRIDES[@]}"

echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
