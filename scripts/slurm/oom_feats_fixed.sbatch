#!/bin/bash
#SBATCH --job-name=oom-feat-fix
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=24:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/oom_feat_fix_%A_%a.out
#SBATCH --error=logs/oom_feat_fix_%A_%a.err

set -euo pipefail
export SWEEP_TIER="extended"
export CLUSTER="dcs"

FEATS=(
    structural_width_approx tda
    wl_kernel wl_kernel_h1 wl_kernel_h2 wl_kernel_h5
    spectral random_walk_kernel
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

N=${#FEATS[@]}; N_SEEDS=${#SEEDS[@]}; N_LOGICS=${#LOGICS[@]}
TOTAL=$((N * N_SEEDS * N_LOGICS))
TASK_ID=${SLURM_ARRAY_TASK_ID}
[ "${TASK_ID}" -ge "${TOTAL}" ] && exit 0

IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

FEAT="${FEATS[$IDX]}"; SEED="${SEEDS[$SEED_IDX]}"; LOGIC="${LOGICS[$LOGIC_IDX]}"
export CARD_FEAT="${FEAT}"

REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== OOM-fix: featuriser=${FEAT} seed=${SEED} logic=${LOGIC} ==="

OVERRIDES=(
    "reference=ref_vector"
    "featuriser@reference.featuriser=${FEAT}"
    "data.logic=${LOGIC}"
    "data.db_path=/dcs/large/u5573765/db/results.sqlite"
    "data.shard_db=${REPO_DIR}/db/shards/shard_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sqlite"
    "seed=${SEED}"
)

python3 -m pipeline.runner "${OVERRIDES[@]}"
echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
