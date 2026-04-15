#!/bin/bash
#SBATCH --job-name=ext-feat-cpu
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=12:00:00
#SBATCH --output=logs/ext_feat_cpu_%A_%a.out
#SBATCH --error=logs/ext_feat_cpu_%A_%a.err
#SBATCH --array=0-260

set -euo pipefail
export SWEEP_TIER="extended"
export CLUSTER="dcs"

FEATS=(
    community community_modularity_only community_spectral
    dynamic_probes dynamic_probes_05s dynamic_probes_1s dynamic_probes_2s dynamic_probes_5s
    entropy_no_mi hash_kernel hypergraph_features
    random_walk_kernel spectral
    static_expanded_k10 static_expanded_k50 static_expanded_k100
    static_expanded_pca_only static_expanded_poly_only
    static_light_bow_binary static_light_bow_no_arith static_light_bow_no_bv static_light_bow_theory
    structural_width structural_width_approx
    tda wl_kernel wl_kernel_h1 wl_kernel_h2 wl_kernel_h5
)
SEEDS=(42 123 456)
LOGICS=("QF_LIA" "QF_BV" "QF_NRA")

N=${#FEATS[@]}
N_SEEDS=${#SEEDS[@]}
N_LOGICS=${#LOGICS[@]}
TOTAL=$((N * N_SEEDS * N_LOGICS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then exit 0; fi

IDX=$((TASK_ID / (N_SEEDS * N_LOGICS)))
REMAINDER=$((TASK_ID % (N_SEEDS * N_LOGICS)))
SEED_IDX=$((REMAINDER / N_LOGICS))
LOGIC_IDX=$((REMAINDER % N_LOGICS))

FEAT="${FEATS[$IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
LOGIC="${LOGICS[$LOGIC_IDX]}"

REPO_DIR="/dcs/23/u5573765/cs351/smt-ablation-cross-system"
cd "${REPO_DIR}"
mkdir -p logs results_csv db/shards

echo "=== Extended: featuriser=${FEAT} seed=${SEED} logic=${LOGIC} ==="
export CARD_FEAT="${FEAT}"

SWEEP_MAP="${REPO_DIR}/sweep_experiments.json"
EXPERIMENT_ID=""
if [ -f "${SWEEP_MAP}" ]; then
    KEY="${FEAT}|${LOGIC}"
    EXPERIMENT_ID=$(python3 -c "import json; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
fi

OVERRIDES=(
    "reference=ref_vector"
    "featuriser@reference.featuriser=${FEAT}"
    "data.logic=${LOGIC}"
    "data.db_path=/dcs/large/u5573765/db/results.sqlite"
    "data.shard_db=${REPO_DIR:-/dcs/23/u5573765/cs351/smt-ablation-cross-system}/db/shards/shard_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.sqlite"
    "seed=${SEED}"
)
[ -n "${EXPERIMENT_ID}" ] && OVERRIDES+=("+experiment_id=${EXPERIMENT_ID}")

python3 -m pipeline.runner "${OVERRIDES[@]}"
echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
