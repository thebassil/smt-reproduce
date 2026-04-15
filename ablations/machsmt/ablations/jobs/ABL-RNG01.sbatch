#!/bin/bash
#SBATCH --job-name=ABL-RNG01
#SBATCH --output=/dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/logs/ABL-RNG01_%j.out
#SBATCH --error=/dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/logs/ABL-RNG01_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=tiger

source /dcs/large/u5573765/env.sh
ulimit -s unlimited

cd /dcs/23/u5573765/cs351/smt-ablation-machsmt
python /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablation_harness.py run /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/templates/seed_sensitivity.yaml --execute
