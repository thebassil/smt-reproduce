#!/bin/bash
#SBATCH --job-name=sweep-feat
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=08:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/sweep_feat_%A_%a.out
#SBATCH --error=logs/sweep_feat_%A_%a.err
#
# Array job: sweep VECTOR featuriser cards with fixed model + policy.
# Each array task runs one (featuriser, seed, logic) combination.
#
# Usage:
#   sbatch sweep_featuriser.sbatch
#   sbatch sweep_featuriser.sbatch 5          # experiment_id for DB write

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

EXPERIMENT_ID_CLI="${1:-}"

# --- Sweep grid ---
# Key vector featurisers (subset — extend as needed)
FEATURISERS=(
    "machsmt_162"
    "static_expanded"
    "static_light_bow"
    "entropy"
    "proof_complexity"
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

# --- Compute (featuriser, seed, logic) from SLURM_ARRAY_TASK_ID ---
N_FEATS=${#FEATURISERS[@]}
N_SEEDS=${#SEEDS[@]}
N_LOGICS=${#LOGICS[@]}
TOTAL=$((N_FEATS * N_SEEDS * N_LOGICS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "TASK_ID ${TASK_ID} >= TOTAL ${TOTAL}, exiting."
    exit 0
fi

FEAT_IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

FEAT="${FEATURISERS[$FEAT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
LOGIC="${LOGICS[$LOGIC_IDX]}"

# --- Environment ---
REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
# DCS: use system Python (no venv needed)
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== Sweep: FEATURISER feat=${FEAT} seed=${SEED} logic=${LOGIC} ==="
export CARD_FEAT="${FEAT}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Grid: ${N_FEATS} featurisers x ${N_SEEDS} seeds x ${N_LOGICS} logics = ${TOTAL} tasks"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Resolve experiment_id ---
EXPERIMENT_ID="${EXPERIMENT_ID_CLI}"
if [ -z "${EXPERIMENT_ID}" ]; then
    SWEEP_MAP="${REPO_DIR}/sweep_experiments.json"
    if [ -f "${SWEEP_MAP}" ]; then
        KEY="${FEAT}|${LOGIC}"
        EXPERIMENT_ID=$(python3 -c "import json,sys; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
    fi
fi

# --- Build Hydra overrides ---
# Uses REF-VECTOR policy (select_argmin) + model (adaboost_ehm) as fixed, varies featuriser
OVERRIDES=(
    "reference=ref_vector"
    "featuriser@reference.featuriser=${FEAT}"
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
