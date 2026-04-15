#!/bin/bash
#SBATCH --job-name=sweep-policy
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=08:00:00
#SBATCH --array=0-116
#SBATCH --output=logs/sweep_policy_%A_%a.out
#SBATCH --error=logs/sweep_policy_%A_%a.err
#
# Array job: sweep policy cards with a fixed VECTOR model across seeds and logics.
# Each array task runs one (policy, seed, logic) combination.
#
# Total tasks = len(POLICIES) * len(SEEDS) * len(LOGICS)
# Adjust --array=0-N to match.
#
# Usage:
#   sbatch sweep_policy.sbatch
#   sbatch sweep_policy.sbatch 5              # experiment_id for DB write

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

EXPERIMENT_ID_CLI="${1:-}"

# --- Sweep grid ---
# All 13 policy cards from WT-Policy
POLICIES=(
    "top1_full"
    "top2_split"
    "top2_split_50_50"
    "top2_split_70_30"
    "top2_split_80_20"
    "top3_split"
    "exponential_timer"
    "confidence_gate"
    "softmax_sampling"
    "inverse_runtime_proportional"
    "pairwise_voting"
    "presolver_then_select"
    "cluster_dispatch"
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

# --- Compute (policy, seed, logic) from SLURM_ARRAY_TASK_ID ---
N_POLICIES=${#POLICIES[@]}
N_SEEDS=${#SEEDS[@]}
N_LOGICS=${#LOGICS[@]}
TOTAL=$((N_POLICIES * N_SEEDS * N_LOGICS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "TASK_ID ${TASK_ID} >= TOTAL ${TOTAL}, exiting."
    exit 0
fi

POLICY_IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

POLICY="${POLICIES[$POLICY_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
LOGIC="${LOGICS[$LOGIC_IDX]}"

# --- Environment ---
REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
# DCS: use system Python (no venv needed)
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== Sweep: POLICY policy=${POLICY} seed=${SEED} logic=${LOGIC} ==="
export CARD_POLICY="${POLICY}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Grid: ${N_POLICIES} policies x ${N_SEEDS} seeds x ${N_LOGICS} logics = ${TOTAL} tasks"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Resolve experiment_id ---
EXPERIMENT_ID="${EXPERIMENT_ID_CLI}"
if [ -z "${EXPERIMENT_ID}" ]; then
    SWEEP_MAP="${REPO_DIR}/sweep_experiments.json"
    if [ -f "${SWEEP_MAP}" ]; then
        KEY="${POLICY}|${LOGIC}"
        EXPERIMENT_ID=$(python3 -c "import json,sys; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
    fi
fi

# --- Build Hydra overrides ---
# Uses REF-VECTOR model (adaboost_ehm) as fixed baseline, varies policy
OVERRIDES=(
    "reference=ref_vector"
    "policy@reference.policy=${POLICY}"
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
