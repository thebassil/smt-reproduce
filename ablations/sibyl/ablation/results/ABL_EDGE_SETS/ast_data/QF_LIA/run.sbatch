#!/bin/bash
# ---------------------------------------------------------------
# Auto-generated sbatch script for Sibyl ablation
# Experiment: ABL_EDGE_SETS
# Knob:       edge_sets = ['AST', 'Data']
# Logic:      QF_LIA
# Generated:  2026-03-04T18:06:14.062393+00:00
# ---------------------------------------------------------------
#SBATCH --job-name=abl_EDGE_SETS_ast_data_QF_LIA
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_EDGE_SETS/ast_data/QF_LIA/slurm_%j.out
#SBATCH --error=/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_EDGE_SETS/ast_data/QF_LIA/slurm_%j.err

set -euo pipefail

echo "[$(date)] Sibyl ablation: ABL_EDGE_SETS | edge_sets=['AST', 'Data'] | QF_LIA"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Environment
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load CUDA/12.6.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /dcs/large/u5573765/env.sh

cd /dcs/23/u5573765/cs351/smt-ablation-sibyl

python3 ablation/runner.py \
    --config "/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_EDGE_SETS/ast_data/QF_LIA/run_config.json" \
    --output-dir "/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_EDGE_SETS/ast_data/QF_LIA" \
    --logic "QF_LIA"

echo "[$(date)] Done."
