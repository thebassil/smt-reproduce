#!/bin/bash
#SBATCH --job-name=ext-feat-gpu
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=12:00:00
#SBATCH --output=logs/ext_feat_gpu_%A_%a.out
#SBATCH --error=logs/ext_feat_gpu_%A_%a.err
#SBATCH --exclude=gecko-01,gecko-02

set -euo pipefail
export SWEEP_TIER="extended"
export CLUSTER="dcs"

FEATS=(
    aig ast_path ast_path_len5 ast_path_len10 ast_path_len20
    clause_hypergraph contrastive_embed contrastive_embed_supervised_only
    formula2vec lcg lcg_star lig llm_embed llm_embed_finetuned
    neuroback sibyl_ast_ud smt_dag subgraph_pool
    tess transformer_embed transformer_embed_graph_transformer
    tree_lstm tree_lstm_flat_lstm
    tripartite tripartite_no_posenc vcg vcg_star vig
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

echo "=== Extended GPU: featuriser=${FEAT} seed=${SEED} logic=${LOGIC} ==="
echo "CUDA: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"

SWEEP_MAP="${REPO_DIR}/sweep_experiments.json"
EXPERIMENT_ID=""
if [ -f "${SWEEP_MAP}" ]; then
    KEY="${FEAT}|${LOGIC}"
    EXPERIMENT_ID=$(python3 -c "import json; d=json.load(open('${SWEEP_MAP}')); print(d.get('${KEY}',''))")
fi

OVERRIDES=(
    "reference=ref_gnn"
    "featuriser@reference.featuriser=${FEAT}"
    "data.logic=${LOGIC}"
    "data.db_path=/dcs/large/u5573765/db/results.sqlite"
    "data.shard_db=${REPO_DIR:-/dcs/23/u5573765/cs351/smt-ablation-cross-system}/db/shards/shard_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.sqlite"
    "seed=${SEED}"
)
[ -n "${EXPERIMENT_ID}" ] && OVERRIDES+=("+experiment_id=${EXPERIMENT_ID}")

python3 -m pipeline.runner "${OVERRIDES[@]}"
echo "=== Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
