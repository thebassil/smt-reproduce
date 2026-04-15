#!/bin/bash
#SBATCH --job-name=ext-model-cpu
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=12:00:00
#SBATCH --output=logs/ext_model_cpu_%A_%a.out
#SBATCH --error=logs/ext_model_cpu_%A_%a.err
#SBATCH --array=0-152

set -euo pipefail
export SWEEP_TIER="extended"
export CLUSTER="dcs"

MODELS=(
    autofolio collaborative_filtering conformal contextual_bandits
    cost_sensitive gp_regression greedy_logic instance_clustering
    lambdarank mosap online_learning ordinal_regression
    pwc quantile_regression solver_logic_pwc stacking survival
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
N_LOGICS=${#LOGICS[@]}
TOTAL=$((N_MODELS * N_SEEDS * N_LOGICS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then exit 0; fi

MODEL_IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
LOGIC="${LOGICS[$LOGIC_IDX]}"

REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== Extended: model=${MODEL} seed=${SEED} logic=${LOGIC} ==="
export CARD_MODEL="${MODEL}"

SWEEP_MAP="${REPO_DIR}/sweep_experiments.json"
EXPERIMENT_ID=""
if [ -f "${SWEEP_MAP}" ]; then
    KEY="${MODEL}|${LOGIC}"
    EXPERIMENT_ID=$(python3 -c "import json; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
fi

OVERRIDES=(
    "reference=ref_vector"
    "model@reference.model=${MODEL}"
    "data.logic=${LOGIC}"
    "data.db_path=/dcs/large/u5573765/db/results.sqlite"
    "data.shard_db=${REPO_DIR:-/dcs/23/u5573765/cs351/smt-ablation-cross-system}/db/shards/shard_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.sqlite"
    "seed=${SEED}"
)
[ -n "${EXPERIMENT_ID}" ] && OVERRIDES+=("+experiment_id=${EXPERIMENT_ID}")

python3 -m pipeline.runner "${OVERRIDES[@]}"
echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
