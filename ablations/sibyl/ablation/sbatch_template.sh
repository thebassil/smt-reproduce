#!/bin/bash
# ---------------------------------------------------------------
# Auto-generated sbatch script for Sibyl ablation
# Experiment: {{EXPERIMENT_CODE}}
# Knob:       {{KNOB}} = {{VALUE}}
# Logic:      {{LOGIC}}
# Generated:  {{TIMESTAMP}}
# ---------------------------------------------------------------
#SBATCH --job-name=abl_{{EXPERIMENT_CODE}}_{{VALUE_SLUG}}_{{LOGIC}}
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3000
#SBATCH --time=08:00:00
#SBATCH --output={{RESULTS_DIR}}/slurm_%j.out
#SBATCH --error={{RESULTS_DIR}}/slurm_%j.err

set -euo pipefail

echo "[$(date)] Sibyl ablation: {{EXPERIMENT_CODE}} | {{KNOB}}={{VALUE}} | {{LOGIC}}"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Environment
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load CUDA/12.6.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SLURM sets CUDA_VISIBLE_DEVICES automatically; do NOT override with SLURM_JOB_GPUS
# export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
source /dcs/large/u5573765/env.sh

# Fix: cublasLt for LSTM mode lives in nvidia-cublas-cu12 pip package
export LD_LIBRARY_PATH="/dcs/23/u5573765/.local/lib/python3.9/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

cd {{PROJECT_ROOT}}

python3 ablation/runner.py \
    --config "{{RUN_CONFIG}}" \
    --output-dir "{{RESULTS_DIR}}" \
    --logic "{{LOGIC}}"

echo "[$(date)] Done."
