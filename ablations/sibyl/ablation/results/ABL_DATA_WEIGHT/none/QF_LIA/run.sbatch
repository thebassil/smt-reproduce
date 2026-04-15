#!/bin/bash
# ---------------------------------------------------------------
# Auto-generated sbatch script for Sibyl ablation
# Experiment: ABL_DATA_WEIGHT
# Knob:       data_weight = none
# Logic:      QF_LIA
# Generated:  2026-03-04T18:06:13.725861+00:00
# ---------------------------------------------------------------
#SBATCH --job-name=abl_ABL_DATA_WEIGHT_none_QF_LIA
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_DATA_WEIGHT/none/QF_LIA/slurm_%j.out
#SBATCH --error=/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_DATA_WEIGHT/none/QF_LIA/slurm_%j.err

set -euo pipefail

echo "[$(date)] Sibyl ablation: ABL_DATA_WEIGHT | data_weight=none | QF_LIA"
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
    --config "/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_DATA_WEIGHT/none/QF_LIA/run_config.json" \
    --output-dir "/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_DATA_WEIGHT/none/QF_LIA" \
    --logic "QF_LIA"

echo "[$(date)] Done."
