#!/bin/bash
# ---------------------------------------------------------------
# Auto-generated sbatch script for Sibyl ablation
# Experiment: ABL_POOL_TYPE
# Knob:       pool_type = add
# Logic:      QF_BV
# Generated:  2026-03-04T18:06:14.657589+00:00
# ---------------------------------------------------------------
#SBATCH --job-name=abl_ABL_POOL_TYPE_add_QF_BV
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_POOL_TYPE/add/QF_BV/slurm_%j.out
#SBATCH --error=/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_POOL_TYPE/add/QF_BV/slurm_%j.err

set -euo pipefail

echo "[$(date)] Sibyl ablation: ABL_POOL_TYPE | pool_type=add | QF_BV"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Environment
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load CUDA/12.6.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /dcs/large/u5573765/env.sh

# Fix: Disable fp16 autocast — add-pooling sums all node embeddings
# which overflows fp16 range (max 65504), producing NaN scores.
# Inject sitecustomize.py that patches autocast(enabled=False).
export PYTHONPATH="/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_POOL_TYPE/add/QF_BV:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/dcs/23/u5573765/.local/lib/python3.9/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

cd /dcs/23/u5573765/cs351/smt-ablation-sibyl

python3 ablation/runner.py \
    --config "/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_POOL_TYPE/add/QF_BV/run_config.json" \
    --output-dir "/dcs/23/u5573765/cs351/smt-ablation-sibyl/ablation/results/ABL_POOL_TYPE/add/QF_BV" \
    --logic "QF_BV"

echo "[$(date)] Done."
