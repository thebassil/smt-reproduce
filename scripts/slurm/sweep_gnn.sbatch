#!/bin/bash
#SBATCH --job-name=sweep-gnn
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/sweep_gnn_%A_%a.out
#SBATCH --error=logs/sweep_gnn_%A_%a.err
#SBATCH --exclude=gecko-01,gecko-02
#
# Array job: sweep GNN model cards across seeds and logics on Avon GPU.
# Each array task runs one (model, seed, logic) combination.
#
# For Blythe: override --gres=gpu:lovelace_l40:1 --cpus-per-task=10 --mem-per-cpu=5960
#
# Usage:
#   sbatch sweep_gnn.sbatch
#   sbatch sweep_gnn.sbatch 5                 # experiment_id for DB write

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

EXPERIMENT_ID_CLI="${1:-}"

# --- Sweep grid ---
MODELS=(
    "gat_mlp"
    "gcn"
    "gin"
    "graphsage"
    "graph_transformer"
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
# DCS gecko: CUDA available natively, no module load needed
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== Sweep: GNN model=${MODEL} seed=${SEED} logic=${LOGIC} ==="
export CARD_MODEL="${MODEL}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Grid: ${N_MODELS} models x ${N_SEEDS} seeds x ${N_LOGICS} logics = ${TOTAL} tasks"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CUDA: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"

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
    "reference=ref_gnn"
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
