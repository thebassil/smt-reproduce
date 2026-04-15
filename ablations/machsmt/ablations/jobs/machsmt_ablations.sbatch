#!/bin/bash
#SBATCH --job-name=machsmt-abl
#SBATCH --output=/dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/logs/machsmt-abl_%A_%a.out
#SBATCH --error=/dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/logs/machsmt-abl_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=tiger
#SBATCH --array=0-8

source /dcs/large/u5573765/env.sh
ulimit -s unlimited

TEMPLATES=(
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/feature_timeout_sweep.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/greedy_toggle.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/kfold_sweep.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/logic_filter_toggle.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/max_score_sweep.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/min_datapoints_sweep.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/ml_core_sweep.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/seed_sensitivity.yaml
  /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/selector_swap.yaml
)

YAML="${TEMPLATES[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task $SLURM_ARRAY_TASK_ID: $YAML ==="

cd /dcs/23/u5573765/cs351/smt-ablation-machsmt
python /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablation_harness.py run "$YAML" --execute
