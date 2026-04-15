#!/bin/bash
#SBATCH --job-name=abl-sweep
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=08:00:00
#SBATCH --array=0-167
#SBATCH --output=logs/abl_sweep_%A_%a.out
#SBATCH --error=logs/abl_sweep_%A_%a.err
#
# Cross-system ablation sweep on SCRTP taskfarm.
# Same pattern as scrtp_ablation_array.sbatch: each task writes a shard DB.
#
# Grid: 28 cards x 3 seeds x 2 logics = 168 tasks (0-167)
#   - 5 vector models + 5 featurisers + 5 GNN models + 13 policies = 28 cards
#
# Submit from SCRTP:
#   cd /storage/dcs/csuvzr/smt-ablation-cross-system
#   sbatch scripts/slurm/ablation_sweep.sbatch
#
# Results: db/shards_ablation/shard_<JOBID>_<TASKID>.sqlite
# Merge on DCS: bash scripts/pull_and_merge_ablation.sh

set -euo pipefail

export SWEEP_TIER="${SWEEP_TIER:-core}"
export CLUSTER="dcs"

WORKDIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
cd "${WORKDIR}"

# DCS: use system Python (no venv needed)
mkdir -p logs db/shards_ablation

MAIN_DB="/dcs/large/u5573765/db/results.sqlite"

# --- Full sweep grid: (axis, reference, card) tuples ---
# Vector models (ref_vector, model=X)
# Featurisers   (ref_vector, featuriser=X)
# GNN models    (ref_gnn,    model=X)
# Policies      (ref_vector, policy=X)

CARDS=(
    # axis|reference|card
    "model|ref_vector|adaboost_ehm"
    "model|ref_vector|knn"
    "model|ref_vector|rf"
    "model|ref_vector|xgboost"
    "model|ref_vector|lightgbm"
    "featuriser|ref_vector|machsmt_162"
    "featuriser|ref_vector|static_expanded"
    "featuriser|ref_vector|static_light_bow"
    "featuriser|ref_vector|entropy"
    "featuriser|ref_vector|proof_complexity"
    "model|ref_gnn|gat_mlp"
    "model|ref_gnn|gcn"
    "model|ref_gnn|gin"
    "model|ref_gnn|graphsage"
    "model|ref_gnn|graph_transformer"
    "policy|ref_vector|top1_full"
    "policy|ref_vector|top2_split"
    "policy|ref_vector|top2_split_50_50"
    "policy|ref_vector|top2_split_70_30"
    "policy|ref_vector|top2_split_80_20"
    "policy|ref_vector|top3_split"
    "policy|ref_vector|exponential_timer"
    "policy|ref_vector|confidence_gate"
    "policy|ref_vector|softmax_sampling"
    "policy|ref_vector|inverse_runtime_proportional"
    "policy|ref_vector|pairwise_voting"
    "policy|ref_vector|presolver_then_select"
    "policy|ref_vector|cluster_dispatch"
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

N_CARDS=${#CARDS[@]}
N_SEEDS=${#SEEDS[@]}
N_LOGICS=${#LOGICS[@]}
TOTAL=$((N_CARDS * N_SEEDS * N_LOGICS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "TASK_ID ${TASK_ID} >= TOTAL ${TOTAL}, exiting."
    exit 0
fi

# --- Decompose task ID → (card, seed, logic) ---
CARD_IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

IFS='|' read -r AXIS REFERENCE CARD <<< "${CARDS[$CARD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
LOGIC="${LOGICS[$LOGIC_IDX]}"

# --- Shard DB path ---
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
SHARD_DB="${WORKDIR}/db/shards_ablation/shard_${ARRAY_JOB_ID}_${TASK_ID}.sqlite"

echo "=== Ablation: ${AXIS}=${CARD} seed=${SEED} logic=${LOGIC} ==="
echo "Job: ${ARRAY_JOB_ID}_${TASK_ID} on $(hostname)"
echo "Reference: ${REFERENCE}"
echo "Shard DB: ${SHARD_DB}"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Build Hydra overrides ---
OVERRIDES=(
    "reference=${REFERENCE}"
    "${AXIS}@reference.${AXIS}=${CARD}"
    "data.logic=${LOGIC}"
    "data.db_path=${MAIN_DB}"
    "data.shard_db=${SHARD_DB}"
    "seed=${SEED}"
)

# experiment_id from sweep_experiments.json (if it exists)
SWEEP_MAP="${WORKDIR}/sweep_experiments.json"
if [ -f "${SWEEP_MAP}" ]; then
    KEY="${CARD}|${LOGIC}"
    EXP_ID=$(python3 -c "import json; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
    if [ -n "${EXP_ID}" ]; then
        OVERRIDES+=("+experiment_id=${EXP_ID}")
        echo "Experiment ID: ${EXP_ID}"
    fi
fi

# --- Run ---
python3 -m pipeline.runner "${OVERRIDES[@]}"

echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
